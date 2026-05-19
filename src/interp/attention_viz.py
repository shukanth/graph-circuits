"""
Attention visualization for Phase 3 interpretability.

Primary entry point:
    plot_attention_grid(checkpoint_paths, save_path)

Six diagnostic probe inputs are defined at module level in PROBE_SPECS and
can be imported and reused by other analysis scripts.

Why re-implement the forward pass in numpy:
    Flax Linen has no hook mechanism — model.apply(params, tokens) returns
    only the final logits. Capturing attention weights requires either
    modifying transformer.py (which we won't do) or computing them directly
    from the param matrices. extract_attention_weights does the latter: it
    replicates the exact einsum operations in MultiHeadAttention.__call__ and
    TransformerLayer.__call__, running in numpy on already-loaded param arrays.

Sequence position structure (4-node graphs, max_edges=6, seq_len=21):
    positions  0– 2:  edge 0  [EDGE, u, v]
    positions  3– 5:  edge 1
    positions  6– 8:  edge 2
    positions  9–17:  edges 3-5 or PAD triplets
    positions 18–20:  query    [QUERY, src, dst]
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from interp.weight_analysis import load_params


# ── Vocabulary constants (4-node graphs, vocab_size=8) ───────────────────────
# Mirrors Vocab(n_nodes=4): tokens 0-3 = nodes, 4=EDGE, 5=QUERY, 6=PAD, 7=INF
_EDGE_TOKEN  = 4
_QUERY_TOKEN = 5
_PAD_TOKEN   = 6
_MAX_EDGES   = 6   # floor(0.50 * 4*3) = 6 for 4-node graphs
_SEQ_LEN     = 21  # 3 * _MAX_EDGES + 3


# ── Six diagnostic probes ────────────────────────────────────────────────────

PROBE_SPECS: list[dict] = [
    {
        "label": "P1: 0→1→2→3\nsrc=0 dst=3  ans=3",
        "edges": [(0, 1), (1, 2), (2, 3)],
        "src": 0, "dst": 3,
    },
    {
        "label": "P2: direct 0→3\nsrc=0 dst=3  ans=1",
        "edges": [(0, 3), (0, 1), (1, 2)],
        "src": 0, "dst": 3,
    },
    {
        "label": "P3: edges reversed\nsrc=0 dst=3  ans=INF",
        "edges": [(1, 0), (2, 1), (3, 2)],
        "src": 0, "dst": 3,
    },
    {
        "label": "P4: P1 reordered\nsrc=0 dst=3  ans=3",
        "edges": [(2, 3), (1, 2), (0, 1)],  # same edges as P1, reversed sequence
        "src": 0, "dst": 3,
    },
    {
        "label": "P5: two equal paths\nsrc=0 dst=3  ans=2",
        "edges": [(0, 1), (1, 3), (0, 2), (2, 3)],
        "src": 0, "dst": 3,
    },
    {
        "label": "P6: P1 graph, close dst\nsrc=0 dst=1  ans=1",
        "edges": [(0, 1), (1, 2), (2, 3)],  # same graph as P1, different query
        "src": 0, "dst": 1,
    },
]


def build_probe_tokens(
    edges: list[tuple[int, int]],
    src: int,
    dst: int,
    max_edges: int = _MAX_EDGES,
) -> np.ndarray:
    """
    Encode a probe graph + query into a token array of shape (3*max_edges+3,).

    Layout: [EDGE u v] * n_edges + [PAD PAD PAD] * (max_edges-n_edges) + [QUERY src dst]
    The PAD-initialized buffer means any positions beyond n_edges are padding
    without needing an explicit fill loop.
    """
    seq_len = 3 * max_edges + 3
    tokens = np.full(seq_len, _PAD_TOKEN, dtype=np.int32)
    for i, (u, v) in enumerate(edges):
        tokens[3 * i    ] = _EDGE_TOKEN
        tokens[3 * i + 1] = u
        tokens[3 * i + 2] = v
    tokens[3 * max_edges    ] = _QUERY_TOKEN
    tokens[3 * max_edges + 1] = src
    tokens[3 * max_edges + 2] = dst
    return tokens


def _layer_norm(
    x: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Layer normalization over the last axis, matching flax.linen.LayerNorm defaults."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return scale * (x - mean) / np.sqrt(var + eps) + bias


def extract_attention_weights(
    params: dict,
    tokens: np.ndarray,
    n_layers: int = 2,
    n_heads: int = 2,
) -> list[np.ndarray]:
    """
    Run a single-example forward pass in numpy and return per-layer attention
    weight matrices (post-softmax).

    Returns a list of n_layers arrays, each shape (n_heads, seq_len, seq_len).
    attn[l][h, i, j] = probability that query position i attends to key position j
    in head h of layer l.

    Replicates MultiHeadAttention.__call__ and TransformerLayer.__call__ exactly:

        x_ln = LayerNorm(x)                              # pre-norm
        Q = (x_ln @ W_Q).reshape(seq, n_heads, d_head)  # "sqhd"
        K = (x_ln @ W_K).reshape(seq, n_heads, d_head)
        scores = einsum("qhd,khd->hqk", Q, K) / sqrt(d_head)
        attn = softmax(scores, axis=-1)

    To get layer 1's correct input, we must also compute V, W_O, and the
    residual update after layer 0.  The layer 0 output feeds layer 1's
    LayerNorm, so skipping this step would give wrong attention weights for
    layer 1.
    """
    p = params["params"]
    seq_len = len(tokens)
    d_model = int(p["layers_0"]["attn"]["W_Q"]["kernel"].shape[0])
    d_head  = d_model // n_heads

    W_E   = np.array(p["W_E"]["embedding"])    # (vocab_size, d_model)
    W_pos = np.array(p["W_pos"]["embedding"])  # (max_seq, d_model)
    x = W_E[tokens] + W_pos[:seq_len]          # (seq_len, d_model)

    all_weights: list[np.ndarray] = []

    for l in range(n_layers):
        lp = p[f"layers_{l}"]

        # Pre-norm: LayerNorm applied before attention (pre-norm architecture)
        x_ln = _layer_norm(
            x,
            np.array(lp["ln"]["scale"]),
            np.array(lp["ln"]["bias"]),
        )

        W_Q = np.array(lp["attn"]["W_Q"]["kernel"])  # (d_model, d_model)
        W_K = np.array(lp["attn"]["W_K"]["kernel"])
        W_V = np.array(lp["attn"]["W_V"]["kernel"])
        W_O = np.array(lp["attn"]["W_O"]["kernel"])

        # Dense with no bias = plain matmul; reshape splits d_model into heads
        Q = (x_ln @ W_Q).reshape(seq_len, n_heads, d_head)  # (seq, heads, d_head)
        K = (x_ln @ W_K).reshape(seq_len, n_heads, d_head)
        V = (x_ln @ W_V).reshape(seq_len, n_heads, d_head)

        # Attention scores: einsum contracts over d only
        # Q[q,h,d] * K[k,h,d] -> scores[h,q,k]
        scores = np.einsum("qhd,khd->hqk", Q, K) * (d_head ** -0.5)

        # Stable softmax: subtract row-max before exp to avoid overflow
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s  = np.exp(scores)
        attn   = exp_s / exp_s.sum(axis=-1, keepdims=True)  # (n_heads, seq, seq)
        all_weights.append(attn)

        # Residual update: required so layer l+1 receives the correct x
        # einsum contracts over k and h: attn[h,q,k] * V[k,h,d] -> z[q,h,d]
        z = np.einsum("hqk,khd->qhd", attn, V).reshape(seq_len, d_model)
        x = x + z @ W_O

    return all_weights


def plot_attention_grid(
    checkpoint_paths: list[tuple[str, str]],
    save_path: str = "notebooks/attention_patterns.png",
    probe_specs: list[dict] | None = None,
    n_layers: int = 2,
    n_heads: int = 2,
) -> None:
    """
    Plot post-softmax attention weights for all probes at all checkpoints.

    Layout:
      Rows:    one per probe (6 rows for the default PROBE_SPECS)
      Columns: n_checkpoints * n_layers * n_heads, grouped as
               [ckpt0: L0H0 L0H1 L1H0 L1H1 | ckpt1: ... | ckpt2: ...]

    Each cell is a (seq_len x seq_len) heatmap.  Thin gray lines mark the
    3-token EDGE triplet boundaries; a red line marks the QUERY boundary at
    position 17.5 (separating edge/PAD tokens from the [QUERY src dst] triplet).

    vmin=0, vmax=1 is shared across the entire figure so intensity is
    comparable between heads and checkpoints.

    checkpoint_paths: list of (display_label, .pkl path) pairs.
    """
    if probe_specs is None:
        probe_specs = PROBE_SPECS

    n_probes    = len(probe_specs)
    n_ckpts     = len(checkpoint_paths)
    n_heads_tot = n_layers * n_heads   # 4 heads per checkpoint
    n_cols      = n_ckpts * n_heads_tot

    # ── Pre-compute all attention weights ─────────────────────────────────────
    # Shape: all_attn[ckpt_idx][probe_idx] = list[np.ndarray (n_heads, seq, seq)]
    all_attn: list[list] = []
    for ck_label, ck_path in checkpoint_paths:
        print(f"  Loading {ck_label} ...", end="", flush=True)
        params = load_params(ck_path)
        probe_weights = [
            extract_attention_weights(params, build_probe_tokens(
                spec["edges"], spec["src"], spec["dst"]
            ), n_layers, n_heads)
            for spec in probe_specs
        ]
        all_attn.append(probe_weights)
        del params
        print(" done")

    # ── Build figure ───────────────────────────────────────────────────────────
    cell = 1.45  # inches per heatmap cell
    fig, axes = plt.subplots(
        n_probes, n_cols,
        figsize=(n_cols * cell + 2.2, n_probes * cell + 1.0),
        squeeze=False,
    )
    fig.suptitle(
        "Post-softmax Attention Weights — Run 5 (n4-wd0.4-200k-seed0)\n"
        "Rows: probe inputs  |  Columns: checkpoint × head  "
        "|  Red line: QUERY boundary (pos 17.5)",
        fontsize=8.5, y=0.999,
    )

    # Column headers on the top row
    head_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    for ci, (ck_label, _) in enumerate(checkpoint_paths):
        for hi, hl in enumerate(head_labels):
            axes[0, ci * n_heads_tot + hi].set_title(
                f"step {ck_label}\n{hl}", fontsize=6.5, pad=2
            )

    # Fill heatmaps row by row
    for pi, spec in enumerate(probe_specs):
        axes[pi, 0].set_ylabel(spec["label"], fontsize=6.5, labelpad=4, va="center")
        for ci in range(n_ckpts):
            layer_weights = all_attn[ci][pi]   # list of n_layers arrays
            for l in range(n_layers):
                for h in range(n_heads):
                    col = ci * n_heads_tot + l * n_heads + h
                    ax  = axes[pi, col]
                    ax.imshow(
                        layer_weights[l][h],
                        vmin=0, vmax=1,
                        cmap="Blues",
                        aspect="auto",
                        interpolation="nearest",
                    )
                    # Triplet boundary lines (subtle gray)
                    for pos in [2.5, 5.5, 8.5, 11.5, 14.5]:
                        ax.axhline(pos, color="gray", linewidth=0.3, alpha=0.35)
                        ax.axvline(pos, color="gray", linewidth=0.3, alpha=0.35)
                    # QUERY boundary (more prominent red)
                    ax.axhline(17.5, color="tomato", linewidth=0.6, alpha=0.85)
                    ax.axvline(17.5, color="tomato", linewidth=0.6, alpha=0.85)
                    ax.set_xticks([])
                    ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Probe comparison: directionality and position-vs-content tests ────────────

_QUERY_START = _MAX_EDGES * 3   # = 18: first QUERY-row index in the seq


def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """
    Symmetric KL divergence (Jensen-Shannon style): mean of KL(p‖q) and KL(q‖p).

    Each input is a 1-D probability vector (one attention row, already sums to 1).
    eps is added before log to prevent log(0) on near-zero softmax weights.
    """
    p = p + eps
    q = q + eps
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return (kl_pq + kl_qp) / 2.0


def compare_probe_pair(
    params: dict,
    spec_a: dict,
    spec_b: dict,
    layer: int,
    head: int,
    n_layers: int = 2,
    n_heads: int = 2,
) -> dict:
    """
    Quantitatively compare the QUERY-row attention distributions of two probes
    for one (layer, head) pair.

    'QUERY rows' = positions _QUERY_START through seq_len-1 (positions 18-20
    for the default 4-node, max_edges=6 layout).  These are the positions whose
    attention patterns most directly encode which edges the model uses to route
    toward its answer.

    Returns a dict with:
        kl_sym   float  symmetric KL divergence, averaged over the 3 QUERY rows
        cos_sim  float  cosine similarity of the flattened QUERY-row submatrices
                        (shape 3*seq_len → scalar), so the entire QUERY block is
                        treated as one vector; 1 = identical direction, 0 = orthogonal
    """
    tokens_a = build_probe_tokens(spec_a["edges"], spec_a["src"], spec_a["dst"])
    tokens_b = build_probe_tokens(spec_b["edges"], spec_b["src"], spec_b["dst"])

    # Both probes must have the same seq_len (guaranteed by fixed max_edges)
    attn_a = extract_attention_weights(params, tokens_a, n_layers, n_heads)[layer][head]
    attn_b = extract_attention_weights(params, tokens_b, n_layers, n_heads)[layer][head]

    # Extract QUERY rows only: shape (3, seq_len)
    qrows_a = attn_a[_QUERY_START:]   # rows 18, 19, 20
    qrows_b = attn_b[_QUERY_START:]

    # Per-row symmetric KL, then average
    kl_vals = [_kl_div(qrows_a[r], qrows_b[r]) for r in range(len(qrows_a))]
    kl_sym  = float(np.mean(kl_vals))

    # Cosine similarity over flattened QUERY block
    va = qrows_a.ravel()
    vb = qrows_b.ravel()
    cos_sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    return {"kl_sym": kl_sym, "cos_sim": cos_sim}


def run_probe_comparisons(
    checkpoint_paths: list[tuple[str, str]],
    comparisons: list[tuple[str, int, int, int, int]] | None = None,
    n_layers: int = 2,
    n_heads: int = 2,
) -> None:
    """
    Print a table of KL divergence and cosine similarity for probe pairs across
    checkpoints.

    comparisons: list of (name, probe_idx_a, probe_idx_b, layer, head).
    Defaults to the two primary diagnostic comparisons:
        P1 vs P3 on L0H0  — directionality test
        P1 vs P4 on L0H0  — position-vs-content test
    """
    if comparisons is None:
        comparisons = [
            ("P1 vs P3  L0H0  (directionality)", 0, 2, 0, 0),
            ("P1 vs P4  L0H0  (pos vs content)", 0, 3, 0, 0),
        ]

    ck_labels = [label for label, _ in checkpoint_paths]
    col_w = 14

    # Header
    header = f"  {'Comparison':<36}" + "".join(
        f"{'step '+lbl:^{col_w}}" for lbl in ck_labels
    )
    metric_header = f"  {'':36}" + "".join(
        f"{'KL    cos':^{col_w}}" for _ in ck_labels
    )
    print(header)
    print(metric_header)
    print("  " + "─" * (36 + col_w * len(ck_labels)))

    for name, ai, bi, layer, head in comparisons:
        row = f"  {name:<36}"
        for _, ck_path in checkpoint_paths:
            params = load_params(ck_path)
            result = compare_probe_pair(
                params, PROBE_SPECS[ai], PROBE_SPECS[bi], layer, head,
                n_layers, n_heads,
            )
            del params
            cell = f"{result['kl_sym']:5.3f}  {result['cos_sim']:5.3f}"
            row += f"{cell:^{col_w}}"
        print(row)
