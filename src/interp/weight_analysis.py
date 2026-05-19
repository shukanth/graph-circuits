"""
Weight-space analysis for the Phase 3 interpretability toolkit.

Primary entry point: scan_checkpoints(checkpoint_dir) scans all checkpoints
and returns per-head Frobenius norm trajectories. Then print_norm_table(data)
shows which heads changed at the grokking transition.

Params tree access paths (confirmed against step_0040000.pkl):
    params['params']['layers_0']['attn']['W_Q']['kernel']  shape (128, 128)
    params['params']['layers_0']['attn']['W_K']['kernel']  shape (128, 128)
    params['params']['layers_0']['attn']['W_V']['kernel']  shape (128, 128)
    params['params']['layers_0']['attn']['W_O']['kernel']  shape (128, 128)
    params['params']['layers_1']['attn']['...']['kernel']  (same for layer 1)

Head slicing (n_heads=2, d_model=128, d_head=64):
    W_Q, W_K, W_V  kernel[:, h*64 : (h+1)*64]   column slice  shape (128, 64)
    W_O            kernel[h*64 : (h+1)*64, :]    row slice     shape (64, 128)

The column slice for Q/K/V matches the forward pass: Dense output of shape
(batch, seq, 128) is reshaped to (batch, seq, n_heads, d_head), so head h
occupies output columns h*64:(h+1)*64. W_O receives the concatenated head
outputs in the same layout, so head h's contribution maps through input rows
h*64:(h+1)*64 — hence the row slice.
"""

import os
import re
import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


_D_HEAD = 64  # d_model (128) // n_heads (2) — fixed for this model


@dataclass
class HeadNorms:
    """Frobenius norms of all four weight sub-matrices for one attention head."""
    WQ: float
    WK: float
    WV: float
    WO: float


def load_params(path: str) -> dict:
    """Load a checkpoint .pkl file and return the Flax Linen params dict."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    leaves = [jnp.array(leaf) for leaf in data["leaves"]]
    return jax.tree_util.tree_unflatten(data["treedef"], leaves)


def extract_head_norms(params: dict, layer: int, head: int) -> HeadNorms:
    """
    Return Frobenius norms of the four weight sub-matrices for one head.

    W_Q, W_K, W_V are column-sliced: the forward pass reshapes Dense output
    (batch, seq, d_model) to (batch, seq, n_heads, d_head), placing head h
    at output columns [h*d_head : (h+1)*d_head].

    W_O is row-sliced: it receives the concatenated head outputs, so head h
    occupies input rows [h*d_head : (h+1)*d_head].
    """
    attn = params["params"][f"layers_{layer}"]["attn"]
    lo, hi = head * _D_HEAD, (head + 1) * _D_HEAD
    return HeadNorms(
        WQ=float(jnp.linalg.norm(attn["W_Q"]["kernel"][:, lo:hi])),
        WK=float(jnp.linalg.norm(attn["W_K"]["kernel"][:, lo:hi])),
        WV=float(jnp.linalg.norm(attn["W_V"]["kernel"][:, lo:hi])),
        WO=float(jnp.linalg.norm(attn["W_O"]["kernel"][lo:hi, :])),
    )


def _step_from_path(path: str) -> int | None:
    """
    Parse the step number from a checkpoint filename.
    Returns None for _final files, which duplicate the corresponding numbered
    checkpoint and should not be counted twice.
    """
    name = os.path.basename(path)
    if "_final" in name:
        return None
    m = re.search(r"step_(\d+)\.pkl$", name)
    return int(m.group(1)) if m else None


def scan_checkpoints(
    checkpoint_dir: str,
    n_layers: int = 2,
    n_heads: int = 2,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Scan all checkpoints in a directory and return per-head Frobenius norm
    trajectories over the course of training.

    Loads one checkpoint at a time (~560 KB each for this model) and discards
    params immediately after norm extraction. Peak memory is bounded at one
    checkpoint regardless of total count.

    Returns a dict mapping string keys to 1-D numpy arrays (one value per step):
        'steps'         step numbers in ascending order
        'L{l}H{h}_WQ'  norm of head h's W_Q sub-matrix in layer l
        'L{l}H{h}_WK'  norm of head h's W_K sub-matrix
        'L{l}H{h}_WV'  norm of head h's W_V sub-matrix
        'L{l}H{h}_WO'  norm of head h's W_O sub-matrix
        'WE'            norm of the token embedding matrix W_E
        'WU'            norm of the unembedding matrix W_U
    """
    entries: list[tuple[int, str]] = []
    for fname in os.listdir(checkpoint_dir):
        step = _step_from_path(fname)
        if step is not None:
            entries.append((step, os.path.join(checkpoint_dir, fname)))
    entries.sort()

    if not entries:
        raise FileNotFoundError(f"No step_*.pkl files found in {checkpoint_dir!r}")

    accum: dict[str, list] = {"steps": []}
    for l in range(n_layers):
        for h in range(n_heads):
            for mat in ("WQ", "WK", "WV", "WO"):
                accum[f"L{l}H{h}_{mat}"] = []
    accum["WE"] = []
    accum["WU"] = []

    n = len(entries)
    for i, (step, path) in enumerate(entries):
        if verbose:
            print(f"  [{i+1:3d}/{n}]  step {step:>7,d}", end="\r", flush=True)

        params = load_params(path)

        accum["steps"].append(step)
        for l in range(n_layers):
            for h in range(n_heads):
                norms = extract_head_norms(params, l, h)
                accum[f"L{l}H{h}_WQ"].append(norms.WQ)
                accum[f"L{l}H{h}_WK"].append(norms.WK)
                accum[f"L{l}H{h}_WV"].append(norms.WV)
                accum[f"L{l}H{h}_WO"].append(norms.WO)

        accum["WE"].append(float(jnp.linalg.norm(params["params"]["W_E"]["embedding"])))
        accum["WU"].append(float(jnp.linalg.norm(params["params"]["W_U"]["kernel"])))

        del params  # discard immediately — bounds peak memory at ~560 KB

    if verbose:
        print(f"\n  Done — scanned {n} checkpoints.")

    return {k: np.array(v) for k, v in accum.items()}


def print_norm_table(
    data: dict[str, np.ndarray],
    key_steps: list[int] | None = None,
    n_layers: int = 2,
    n_heads: int = 2,
    file=None,
) -> None:
    """
    Print a formatted table of per-head norms at selected steps.

    Pass file=<open file handle> to write to disk instead of stdout.

    Defaults to the eight Run 5 (wd=0.4) anchor checkpoints spanning the
    two-stage staircase grokking transition:
        100k  pre-Stage 1 plateau
        116k  Stage 1 onset
        120k  Stage 1 midpoint
        126k  post-Stage 1, entering plateau 2
        160k  plateau 2 anchor
        186k  pre-Stage 2
        190k  Stage 2 transition peak
        196k  post-Stage 2 settled
    """
    if key_steps is None:
        key_steps = [100_000, 116_000, 120_000, 126_000, 160_000, 186_000, 190_000, 196_000]

    steps = data["steps"]
    cols: list[tuple[int, int]] = []
    for s in key_steps:
        matches = np.where(steps == s)[0]
        if len(matches) == 0:
            print(f"  Warning: step {s:,} not found in data — skipping.", file=file)
        else:
            cols.append((s, int(matches[0])))

    col_w = 9
    # Right-justify each header to col_w so it aligns with the 9-char data columns.
    header = f"{'':14}" + "".join(f"{'step'+str(s//1000)+'k':>{col_w}}" for s, _ in cols)
    print(header, file=file)
    print("─" * len(header), file=file)

    for l in range(n_layers):
        for h in range(n_heads):
            for mat in ("WQ", "WK", "WV", "WO"):
                key = f"L{l}H{h}_{mat}"
                vals = "".join(f"{data[key][idx]:>{col_w}.3f}" for _, idx in cols)
                print(f"  L{l} H{h}  {mat}  {vals}", file=file)
        if l < n_layers - 1:
            print(file=file)

    print(file=file)
    we_row = "".join(f"{data['WE'][idx]:>{col_w}.3f}" for _, idx in cols)
    wu_row = "".join(f"{data['WU'][idx]:>{col_w}.3f}" for _, idx in cols)
    print(f"  WE          {we_row}", file=file)
    print(f"  WU          {wu_row}", file=file)


def plot_norm_trajectories(
    data: dict[str, np.ndarray],
    save_path: str = "notebooks/weight_norms.png",
    stage1_window: tuple[int, int] = (114_000, 125_000),
    stage2_window: tuple[int, int] = (188_000, 192_000),
) -> None:
    """
    Plot per-head Frobenius norm trajectories over all checkpoints.

    Six panels (3 rows x 2 cols): WQ, WK, WV, WO, WE, WU.
    Each attention panel has four lines: L0H0, L0H1, L1H0, L1H1.
    Shaded bands mark the two grokking transition windows.

    Color scheme: blue = Layer 0, orange = Layer 1.
    Line style:   solid = Head 0, dashed = Head 1.
    """
    steps_k = data["steps"] / 1_000  # x-axis in thousands

    # (layer, head) -> (color, linestyle, legend label)
    head_style: dict[tuple[int, int], tuple[str, str, str]] = {
        (0, 0): ("tab:blue",   "-",  "L0 H0"),
        (0, 1): ("tab:blue",   "--", "L0 H1"),
        (1, 0): ("tab:orange", "-",  "L1 H0"),
        (1, 1): ("tab:orange", "--", "L1 H1"),
    }

    s1_lo, s1_hi = stage1_window[0] / 1_000, stage1_window[1] / 1_000
    s2_lo, s2_hi = stage2_window[0] / 1_000, stage2_window[1] / 1_000

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    fig.suptitle("Weight Frobenius Norm Trajectories — Run 5 (n4-wd0.4-200k-seed0)",
                 fontsize=12, y=0.98)

    attn_panels = [
        ("WQ", axes[0, 0]),
        ("WK", axes[0, 1]),
        ("WV", axes[1, 0]),
        ("WO", axes[1, 1]),
    ]

    def _add_bands(ax: plt.Axes) -> None:
        ax.axvspan(s1_lo, s1_hi, alpha=0.15, color="gold",   zorder=0, label="_nolegend_")
        ax.axvspan(s2_lo, s2_hi, alpha=0.20, color="tomato", zorder=0, label="_nolegend_")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_ylabel("‖W‖_F", fontsize=9)

    for mat, ax in attn_panels:
        for (l, h), (color, ls, lbl) in head_style.items():
            key = f"L{l}H{h}_{mat}"
            ax.plot(steps_k, data[key], color=color, linestyle=ls,
                    linewidth=1.2, label=lbl)
        _add_bands(ax)
        ax.set_title(mat, fontsize=10)

    # Legend for head styles — top-right panel (WK)
    axes[0, 1].legend(fontsize=8, loc="upper right", framealpha=0.8)

    # WE and WU — single-line panels
    for key, ax, title in [("WE", axes[2, 0], "WE"), ("WU", axes[2, 1], "WU")]:
        ax.plot(steps_k, data[key], color="tab:green", linewidth=1.4)
        _add_bands(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training step (thousands)", fontsize=9)

    # Shared legend for transition bands at figure bottom
    patch1 = mpatches.Patch(color="gold",   alpha=0.5, label="Stage 1  114k–125k")
    patch2 = mpatches.Patch(color="tomato", alpha=0.5, label="Stage 2  188k–192k")
    fig.legend(handles=[patch1, patch2], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.00))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")
