"""
PAD-position ablation experiment for causal validation of the PAD-bias shortcut.

Correlation story so far: during Stage 1 (114k–126k) L1H0 and L1H1 build a
PAD-token OV bias — L1H0 attends heavily to PAD positions, injecting a constant
offset into the residual stream that boosts accuracy without reading graph
structure.  At Stage 2 (188k–192k) this shortcut is dismantled and replaced by
L0H1's content-sensitive circuit.

This module tests that story causally: if we zero the PAD-position attention
contributions in Layer 1, accuracy should drop most at 126k (shortcut fully built
and load-bearing), barely matter at 100k (not yet built) or 196k (dismantled),
and be intermediate at 160k (inter-stage plateau).

Two ablation variants (see run_pad_ablation for details):
    renormalized  — zero PAD columns, rescale surviving weights to sum-to-1
    no-renorm     — zero PAD columns, leave row sums at ~0.57

Primary entry point:
    run_pad_ablation(checkpoint_paths, val_tokens, val_labels)
"""

import numpy as np

from interp.attention_viz import _layer_norm
from interp.weight_analysis import load_params


_PAD_TOKEN = 6   # matches Vocab(n_nodes=4): tokens 0-3=nodes, 4=EDGE, 5=QUERY, 6=PAD, 7=INF


def _numpy_forward_pass(
    params: dict,
    tokens: np.ndarray,
    n_layers: int = 2,
    n_heads: int = 2,
    ablate_layer: int | None = None,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Full single-example forward pass in numpy.  Returns logits at the last token
    position (the dst token, position seq_len-1), shape (vocab_size,).

    If ablate_layer is set, PAD-position key columns in that layer's attention
    matrix are zeroed before value aggregation.  If renormalize=True the surviving
    weights are rescaled row-by-row to sum to 1; otherwise the row sums shrink in
    proportion to how much attention was directed to PAD positions.

    Replicates the exact einsum sequence from MultiHeadAttention.__call__ and
    TransformerLayer.__call__ in src/model/transformer.py.
    """
    p = params["params"]
    seq_len = len(tokens)
    d_model = int(p["layers_0"]["attn"]["W_Q"]["kernel"].shape[0])
    d_head = d_model // n_heads

    W_E = np.array(p["W_E"]["embedding"])    # (vocab, d_model)
    W_pos = np.array(p["W_pos"]["embedding"])  # (max_seq, d_model)
    W_U = np.array(p["W_U"]["kernel"])         # (d_model, vocab)

    x = W_E[tokens] + W_pos[:seq_len]          # (seq_len, d_model)

    # Boolean mask: True at positions whose token is PAD.
    pad_cols = tokens == _PAD_TOKEN             # (seq_len,)

    for l in range(n_layers):
        lp = p[f"layers_{l}"]
        x_ln = _layer_norm(
            x,
            np.array(lp["ln"]["scale"]),
            np.array(lp["ln"]["bias"]),
        )

        W_Q = np.array(lp["attn"]["W_Q"]["kernel"])
        W_K = np.array(lp["attn"]["W_K"]["kernel"])
        W_V = np.array(lp["attn"]["W_V"]["kernel"])
        W_O = np.array(lp["attn"]["W_O"]["kernel"])

        Q = (x_ln @ W_Q).reshape(seq_len, n_heads, d_head)
        K = (x_ln @ W_K).reshape(seq_len, n_heads, d_head)
        V = (x_ln @ W_V).reshape(seq_len, n_heads, d_head)

        scores = np.einsum("qhd,khd->hqk", Q, K) * (d_head ** -0.5)
        scores -= scores.max(axis=-1, keepdims=True)  # stable softmax
        exp_s = np.exp(scores)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)  # (n_heads, seq, seq)

        if l == ablate_layer:
            # Zero PAD key columns across all heads and all query positions.
            # attn[:, :, pad_cols] broadcasts to shape (n_heads, seq_len, n_pad_positions).
            attn = attn.copy()
            attn[:, :, pad_cols] = 0.0

            if renormalize:
                # Rescale each query row so surviving weights sum to 1.
                # Where the entire row was PAD (extreme edge case), keep zeros.
                row_sum = attn.sum(axis=-1, keepdims=True)         # (n_heads, seq, 1)
                safe_sum = np.where(row_sum == 0.0, 1.0, row_sum)
                attn = attn / safe_sum

        z = np.einsum("hqk,khd->qhd", attn, V).reshape(seq_len, d_model)
        x = x + z @ W_O

    # Final layer norm before W_U, matching transformer.py:
    #   x = self.ln_final(x)
    #   return self.W_U(x[:, seq - 1, :])
    x_final = _layer_norm(
        x[-1:],
        np.array(p["ln_final"]["scale"]),
        np.array(p["ln_final"]["bias"]),
    )
    return x_final[0] @ W_U  # (vocab_size,)


def _evaluate_ablation(
    params: dict,
    val_tokens: np.ndarray,
    val_labels: np.ndarray,
    ablate_layer: int | None,
    renormalize: bool,
) -> float:
    """
    Run the (possibly ablated) forward pass over the full validation set.
    Returns fraction correct.
    """
    n = len(val_labels)
    correct = 0
    for i in range(n):
        logits = _numpy_forward_pass(
            params, val_tokens[i],
            ablate_layer=ablate_layer,
            renormalize=renormalize,
        )
        if int(np.argmax(logits)) == int(val_labels[i]):
            correct += 1
    return correct / n


def run_pad_ablation(
    checkpoint_paths: list[tuple[str, str]],
    val_tokens: np.ndarray,
    val_labels: np.ndarray,
    ablate_layer: int = 1,
) -> None:
    """
    Run the PAD-position ablation experiment across all supplied checkpoints.

    For each checkpoint, three conditions are evaluated on the full validation set:
        baseline   — unmodified forward pass
        renorm     — PAD key columns zeroed, rows rescaled to sum-to-1
        no-renorm  — PAD key columns zeroed, row sums left < 1

    Renorm tests whether PAD routing carries unique value information: if
    redistributing attention to non-PAD tokens compensates for the loss, the
    model is just using PAD positions as a convenient bucket, not for their
    specific content.

    No-renorm tests whether the PAD bias is a magnitude effect: zeroing PAD
    columns shrinks the magnitude of Layer 1's residual-stream contribution by
    ~43% (the PAD fraction), directly removing any constant offset that PAD
    attention was injecting.

    Expected arc across checkpoints:
        100k  — shortcut not yet built; ablation should have little effect
        126k  — shortcut fully built and load-bearing; largest accuracy drop
        160k  — inter-stage plateau; intermediate drop
        196k  — shortcut dismantled; ablation should have little effect

    If this arc is observed, the causal case for the PAD-bias shortcut is airtight.

    checkpoint_paths: list of (display_label, .pkl_path) pairs.
    val_tokens: (n_val, seq_len) int array
    val_labels: (n_val,)        int array
    ablate_layer: which layer to ablate (default 1, where the shortcut lives)
    """
    n_pad = int((val_tokens == _PAD_TOKEN).mean() * 100)
    print(f"\nPAD-position ablation — Layer {ablate_layer}")
    print(f"  val set: {len(val_labels)} examples  |  "
          f"mean PAD fraction: ~{n_pad}% of sequence positions")
    print(f"  ablation: zero attention weight on key positions where token == PAD\n")

    col_w = 10
    hdr  = f"  {'step':<8}"
    hdr += f"  {'baseline':>{col_w}}  {'renorm':>{col_w}}  {'no-renorm':>{col_w}}"
    hdr += f"  {'Δ renorm':>{col_w}}  {'Δ no-renorm':>{col_w+2}}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for ck_label, ck_path in checkpoint_paths:
        print(f"  {ck_label:<8}  ", end="", flush=True)
        params = load_params(ck_path)

        base = _evaluate_ablation(params, val_tokens, val_labels,
                                   ablate_layer=None, renormalize=True)
        print(f"{base:>{col_w}.3f}  ", end="", flush=True)

        renorm = _evaluate_ablation(params, val_tokens, val_labels,
                                     ablate_layer=ablate_layer, renormalize=True)
        print(f"{renorm:>{col_w}.3f}  ", end="", flush=True)

        no_renorm = _evaluate_ablation(params, val_tokens, val_labels,
                                        ablate_layer=ablate_layer, renormalize=False)
        del params

        d_r  = renorm    - base
        d_nr = no_renorm - base
        print(f"{no_renorm:>{col_w}.3f}  {d_r:>+{col_w}.3f}  {d_nr:>+{col_w+2}.3f}")

    print()
