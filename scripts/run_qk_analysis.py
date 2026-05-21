"""
Run the L0H1 QK circuit analysis for Run 5 (n4-wd0.4-200k-seed0).

Compares the token-type attention affinity matrix W_E W_Q W_K^T W_E^T
at step 126k (post-Stage-1, shortcut fully built) and step 196k
(post-Stage-2, shortcut dismantled).

Saves a 3-row visual (126k, 196k, diff) to notebooks/qk_circuits.png
and prints the L0 head matrices numerically.

Usage (from repo root):
    python scripts/run_qk_analysis.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["JAX_PLATFORMS"] = "cpu"

from interp.ov_circuit import plot_qk_comparison, print_qk_analysis

CHECKPOINT_DIR = "checkpoints"
CHECKPOINTS = [
    ("126k", os.path.join(CHECKPOINT_DIR, "step_0126000.pkl")),
    ("196k", os.path.join(CHECKPOINT_DIR, "step_0196000.pkl")),
]


def main() -> None:
    print("QK circuit analysis — L0H1 at 126k vs 196k\n")

    print("Generating comparison plot...")
    plot_qk_comparison(
        checkpoint_paths=CHECKPOINTS,
        save_path="notebooks/qk_circuits.png",
    )

    print_qk_analysis(
        checkpoint_paths=CHECKPOINTS,
        layers=[0, 1],
    )


if __name__ == "__main__":
    main()
