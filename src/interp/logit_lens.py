"""
Logit lens analysis for Phase 3 interpretability.

The logit lens (Nostalgebraist 2020) projects the residual stream at each
layer through ln_final + W_U to read the model's "partial prediction" at
that depth.  For a 2-layer model the depths are:

    embed_only  —  W_E[tokens] + W_pos, no attention
    after_L0    —  above + Layer 0 attention output
    full_model  —  above + Layer 1 attention output

Accuracy is broken down by answer category (distance=1, 2, 3, INF) to
pinpoint which layer is responsible for which part of the computation.

Primary entry point:
    run_logit_lens(checkpoint_paths, val_tokens, val_labels)
"""

import numpy as np

from interp.attention_viz import _layer_norm
from interp.weight_analysis import load_params


_INF_TOKEN  = 7          # vocab index for the INF answer token
_CATEGORIES = [1, 2, 3, _INF_TOKEN]
_CAT_LABELS = {1: "dist=1", 2: "dist=2", 3: "dist=3", _INF_TOKEN: "INF"}


def _forward_to_depth(
    params: dict,
    tokens: np.ndarray,
    stop_after_layer: int | None,
    n_heads: int = 2,
) -> np.ndarray:
    """
    Run the forward pass to a specified depth, then apply ln_final + W_U to
    the residual stream at the dst (last) sequence position.

    stop_after_layer=-1  → embedding only, no attention
    stop_after_layer=0   → Layer 0 only (before Layer 1)
    stop_after_layer=None → full model (all layers)

    Applying ln_final to an intermediate residual stream is the standard
    logit-lens approximation.  It is exact only for the final layer; earlier
    depths are heuristic projections.
    """
    p = params["params"]
    seq_len   = len(tokens)
    d_model   = int(p["layers_0"]["attn"]["W_Q"]["kernel"].shape[0])
    d_head    = d_model // n_heads

    W_E   = np.array(p["W_E"]["embedding"])
    W_pos = np.array(p["W_pos"]["embedding"])
    W_U   = np.array(p["W_U"]["kernel"])

    x = W_E[tokens] + W_pos[:seq_len]          # (seq_len, d_model)

    if stop_after_layer != -1:
        n_run = 2 if stop_after_layer is None else (stop_after_layer + 1)
        for l in range(n_run):
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
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s  = np.exp(scores)
            attn   = exp_s / exp_s.sum(axis=-1, keepdims=True)

            z = np.einsum("hqk,khd->qhd", attn, V).reshape(seq_len, d_model)
            x = x + z @ W_O

    x_final = _layer_norm(
        x[-1:],
        np.array(p["ln_final"]["scale"]),
        np.array(p["ln_final"]["bias"]),
    )
    return x_final[0] @ W_U   # (vocab_size,)


def _eval_by_category(
    params: dict,
    val_tokens: np.ndarray,
    val_labels: np.ndarray,
    stop_after_layer: int | None,
) -> dict[int, tuple[int, int]]:
    """
    Evaluate accuracy at one depth, split by answer category.
    Returns dict: label_value -> (n_correct, n_total).
    """
    counts: dict[int, list[int]] = {cat: [0, 0] for cat in _CATEGORIES}
    for tokens, label in zip(val_tokens, val_labels):
        label = int(label)
        if label not in counts:
            continue
        logits = _forward_to_depth(params, tokens, stop_after_layer)
        pred   = int(np.argmax(logits))
        counts[label][1] += 1
        if pred == label:
            counts[label][0] += 1
    return {cat: (c, t) for cat, (c, t) in counts.items()}


def run_logit_lens(
    checkpoint_paths: list[tuple[str, str]],
    val_tokens: np.ndarray,
    val_labels: np.ndarray,
) -> None:
    """
    For each checkpoint, print a table of per-category accuracy at each layer
    depth: embedding only, after Layer 0, and full model.

    Reading the table:
        - embed_only row is the baseline: what W_E + W_pos alone predicts.
        - after_L0 row shows what Layer 0's attention adds.
        - full_model row is the model's true output.
        - The per-category breakdown (dist=1 vs dist=2 vs dist=3 vs INF)
          reveals which distance cases each layer is responsible for.
    """
    depths = [
        ("embed_only",  -1),
        ("after_L0",     0),
        ("full_model", None),
    ]
    col_w = 10

    print(f"\nLogit lens — accuracy by layer depth and answer category")
    print(f"  val set: {len(val_labels)} examples")
    print(f"  Depths: embed_only (no attention) | after_L0 | full_model\n")

    for ck_label, ck_path in checkpoint_paths:
        print(f"  ── step {ck_label} {'─' * 58}")
        params = load_params(ck_path)

        hdr  = f"  {'depth':<14}  {'overall':>{col_w}}"
        hdr += "".join(f"  {_CAT_LABELS[c]:>{col_w}}" for c in _CATEGORIES)
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        results: dict[str, dict[int, tuple[int, int]]] = {}
        for depth_lbl, depth in depths:
            cats = _eval_by_category(params, val_tokens, val_labels, depth)
            results[depth_lbl] = cats

            total_c = sum(c for c, t in cats.values())
            total_t = sum(t for c, t in cats.values())
            overall = total_c / total_t if total_t > 0 else 0.0

            row  = f"  {depth_lbl:<14}  {overall:>{col_w}.3f}"
            for cat in _CATEGORIES:
                c, t = cats[cat]
                acc  = c / t if t > 0 else float("nan")
                row += f"  {acc:>{col_w}.3f}"
            print(row)

        # Show the Layer 0 → full model delta per category
        l0   = results["after_L0"]
        full = results["full_model"]
        row  = f"  {'Δ L1 adds':<14}  {'':>{col_w}}"
        for cat in _CATEGORIES:
            c0, t0   = l0[cat]
            cf, tf   = full[cat]
            delta    = (cf / tf - c0 / t0) if t0 > 0 and tf > 0 else float("nan")
            row     += f"  {delta:>+{col_w}.3f}"
        print(row)

        # Category sizes (same for all depths)
        cats_full = results["full_model"]
        row  = f"  {'n_examples':<14}  {'':>{col_w}}"
        row += "".join(f"  {cats_full[c][1]:>{col_w}}" for c in _CATEGORIES)
        print(row)
        print()

        del params
