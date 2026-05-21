"""
L0H1 full OV circuit analysis at step 126k and 196k.

Computes W_E @ W_V_h @ W_O_h @ W_U for Layer 0 Head 1 at both anchor
checkpoints to characterise what L0H1 writes to the residual stream when
it attends to edge-triplet positions containing the dst token.

Usage (from repo root):
    python scripts/run_l0h1_ov.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["JAX_PLATFORMS"] = "cpu"

from interp.ov_circuit import print_ov_full, plot_ov_comparison

CHECKPOINT_DIR = "checkpoints"
CHECKPOINTS = [
    ("126k", os.path.join(CHECKPOINT_DIR, "step_0126000.pkl")),
    ("196k", os.path.join(CHECKPOINT_DIR, "step_0196000.pkl")),
]


def main() -> None:
    print("L0H1 OV circuit analysis\n")

    # Numerical table for all rows
    print_ov_full(CHECKPOINTS, layer=0, head=1)

    # Visual comparison: L0 heads only, 2 checkpoints + diff
    print("Generating L0 OV comparison plot...")
    plot_ov_comparison(
        checkpoint_paths=CHECKPOINTS,
        save_path="notebooks/ov_l0_circuits.png",
        n_layers=1,
        n_heads=2,
    )


if __name__ == "__main__":
    main()
