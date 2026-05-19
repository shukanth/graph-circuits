"""
OV circuit analysis for Phase 3 interpretability.

Primary entry point:
    plot_ov_comparison(checkpoint_paths, save_path)

The OV circuit of head (l, h) is the composition W_V_h @ W_O_h — a
(d_model × d_model) linear map from the residual stream at an attended
position to what gets written back.  Projecting through the token embedding
and unembedding matrices gives the full token-to-logit circuit:

    OV_full = W_E  (vocab × d_model)
            @ W_V_h (d_model × d_head)
            @ W_O_h (d_head × d_model)
            @ W_U   (d_model × vocab)
          → shape (vocab, vocab)

Entry [i, j]: if this head attends to a position whose token is i,
              how much does it push toward outputting vocab token j?

Because vocab_size=8, this is a readable 8×8 matrix in token space.
It is independent of what the head actually attends to (that is QK) —
it captures purely what information the head can copy or transform.

Why this matters for Stage 2:
    L1H0 and L1H1 both settled into near-uniform attention after Stage 2,
    meaning their QK routing did not change.  If Layer 1 is responsible for
    the Stage 2 accuracy jump, the change must be in their OV circuits.
    A structured OV matrix (e.g., strong output-answer logit amplification)
    at step 196k that is absent at step 126k would confirm this.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from interp.weight_analysis import load_params


_D_HEAD    = 64   # d_model (128) // n_heads (2)
_VOCAB_LABELS = ["0", "1", "2", "3", "EDGE", "QRY", "PAD", "INF"]


def compute_ov_circuit(
    params: dict,
    layer: int,
    head: int,
) -> np.ndarray:
    """
    Compute the full token-to-logit OV circuit matrix for one head.

    Returns a (vocab_size, vocab_size) array where entry [i, j] is the
    contribution to output logit j when this head attends to a token-i position.

    Slicing rules (same as weight_analysis.py):
        W_V_h = W_V kernel[:, lo:hi]   column slice — (d_model, d_head)
        W_O_h = W_O kernel[lo:hi, :]   row slice    — (d_head, d_model)
    """
    p   = params["params"]
    lo, hi = head * _D_HEAD, (head + 1) * _D_HEAD

    W_E   = np.array(p["W_E"]["embedding"])                              # (vocab, d_model)
    W_V_h = np.array(p[f"layers_{layer}"]["attn"]["W_V"]["kernel"])[:, lo:hi]  # (d_model, d_head)
    W_O_h = np.array(p[f"layers_{layer}"]["attn"]["W_O"]["kernel"])[lo:hi, :]  # (d_head, d_model)
    W_U   = np.array(p["W_U"]["kernel"])                                 # (d_model, vocab)

    return W_E @ W_V_h @ W_O_h @ W_U   # (vocab, vocab)


def plot_ov_comparison(
    checkpoint_paths: list[tuple[str, str]],
    save_path: str = "notebooks/ov_circuits.png",
    n_layers: int = 2,
    n_heads: int = 2,
) -> None:
    """
    Plot the full OV circuit matrices for all heads at all checkpoints,
    plus the per-entry difference between the last and first checkpoint.

    Layout: rows = checkpoints + one difference row
            cols = n_layers * n_heads (L0H0, L0H1, L1H0, L1H1)

    Each cell is an 8×8 heatmap (vocab × vocab).  A diverging colormap
    (RdBu_r) is used with vmin/vmax set symmetrically to the 99th-percentile
    absolute value across all cells, so the zero point (no contribution)
    always maps to white.

    The difference row uses the same colormap/scale so that the magnitude of
    the Stage-2 change is directly comparable to the raw circuit values.
    """
    n_ckpts   = len(checkpoint_paths)
    n_cols    = n_layers * n_heads
    head_keys = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    head_lbls = [f"L{l}H{h}" for l, h in head_keys]

    # ── Load all OV circuits ──────────────────────────────────────────────────
    # ov[ckpt_idx][head_idx] = (vocab, vocab) array
    all_ov: list[list[np.ndarray]] = []
    for ck_label, ck_path in checkpoint_paths:
        print(f"  Loading {ck_label} ...", end="", flush=True)
        params = load_params(ck_path)
        all_ov.append([
            compute_ov_circuit(params, l, h) for l, h in head_keys
        ])
        del params
        print(" done")

    # ── Symmetric color scale from 99th-percentile absolute value ────────────
    all_vals = np.concatenate([m.ravel() for row in all_ov for m in row])
    vlim = float(np.percentile(np.abs(all_vals), 99))

    # ── Build figure ──────────────────────────────────────────────────────────
    n_rows = n_ckpts + 1   # checkpoints + difference row
    cell   = 2.2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell + 1.5, n_rows * cell + 1.2),
        squeeze=False,
    )
    fig.suptitle(
        "Full OV Circuit  W_E W_V W_O W_U  — Run 5 (n4-wd0.4-200k-seed0)\n"
        "Rows: checkpoints + difference (last − first)  "
        "|  Cols: heads  |  Axes: source token → output logit",
        fontsize=8.5, y=1.001,
    )

    def _draw_cell(ax, mat, vl, title, row_lbl=None):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vl, vmax=vl,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(_VOCAB_LABELS, fontsize=5.5, rotation=45, ha="right")
        ax.set_yticklabels(_VOCAB_LABELS, fontsize=5.5)
        ax.set_title(title, fontsize=7, pad=3)
        if row_lbl:
            ax.set_ylabel(row_lbl, fontsize=6.5, labelpad=4)
        # grid lines between cells
        for k in np.arange(-0.5, 8, 1):
            ax.axhline(k, color="white", linewidth=0.4)
            ax.axvline(k, color="white", linewidth=0.4)
        return im

    # Checkpoint rows
    for ci, (ck_label, _) in enumerate(checkpoint_paths):
        for hi, (hl, (l, h)) in enumerate(zip(head_lbls, head_keys)):
            row_lbl = f"step {ck_label}" if hi == 0 else None
            title   = hl if ci == 0 else ""
            _draw_cell(axes[ci, hi], all_ov[ci][hi], vlim, title, row_lbl)

    # Difference row: last checkpoint − first checkpoint
    diff_label = f"diff  ({checkpoint_paths[-1][0]} − {checkpoint_paths[0][0]})"
    for hi, hl in enumerate(head_lbls):
        diff = all_ov[-1][hi] - all_ov[0][hi]
        row_lbl = diff_label if hi == 0 else None
        _draw_cell(axes[-1, hi], diff, vlim, "", row_lbl)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=plt.Normalize(vmin=-vlim, vmax=vlim))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("logit contribution", fontsize=7)

    plt.tight_layout(rect=[0, 0, 0.97, 0.998])
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def print_ov_frobenius(
    checkpoint_paths: list[tuple[str, str]],
    n_layers: int = 2,
    n_heads: int = 2,
) -> None:
    """
    Print Frobenius norms of the full OV circuit matrices and their differences
    across checkpoints, to quantify how much each head's OV circuit changed.

    The difference norm  ‖OV_last − OV_first‖_F  relative to  ‖OV_first‖_F
    gives a scale-free measure of how much the OV circuit changed at Stage 2.
    """
    head_keys = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    ov_by_step: dict[str, list[np.ndarray]] = {
        f"L{l}H{h}": [] for l, h in head_keys
    }

    for ck_label, ck_path in checkpoint_paths:
        params = load_params(ck_path)
        for l, h in head_keys:
            ov_by_step[f"L{l}H{h}"].append(compute_ov_circuit(params, l, h))
        del params

    ck_labels = [lbl for lbl, _ in checkpoint_paths]
    col_w = 10

    header = f"  {'Head':<8}" + "".join(f"{'‖OV_'+lbl+'‖':>{col_w}}" for lbl in ck_labels)
    header += f"  {'‖diff‖_F':>{col_w}}  {'rel. change':>12}"
    print(header)
    print("  " + "─" * len(header))

    for key in [f"L{l}H{h}" for l, h in head_keys]:
        matrices = ov_by_step[key]
        norms    = [float(np.linalg.norm(m)) for m in matrices]
        diff_norm = float(np.linalg.norm(matrices[-1] - matrices[0]))
        rel       = diff_norm / norms[0] if norms[0] > 0 else float("nan")
        row  = f"  {key:<8}"
        row += "".join(f"{n:>{col_w}.4f}" for n in norms)
        row += f"  {diff_norm:>{col_w}.4f}  {rel:>11.3f}x"
        print(row)
