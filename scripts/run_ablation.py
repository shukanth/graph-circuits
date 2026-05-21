"""
Run the PAD-position ablation experiment for Run 5 (n4-wd0.4-200k-seed0).

Evaluates Layer 1 PAD ablation at four checkpoints spanning both grokking
transitions: 100k (pre-Stage-1), 126k (post-Stage-1), 160k (inter-stage),
196k (post-Stage-2).

Usage (from repo root):
    python scripts/run_ablation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["JAX_PLATFORMS"] = "cpu"

from data.dataset import generate_dataset
from interp.ablation import run_pad_ablation

CHECKPOINT_DIR = "checkpoints"
CHECKPOINTS = [
    ("100k", os.path.join(CHECKPOINT_DIR, "step_0100000.pkl")),
    ("126k", os.path.join(CHECKPOINT_DIR, "step_0126000.pkl")),
    ("160k", os.path.join(CHECKPOINT_DIR, "step_0160000.pkl")),
    ("196k", os.path.join(CHECKPOINT_DIR, "step_0196000.pkl")),
]


def main() -> None:
    print("Generating dataset (n_nodes=4, n_graphs=200, seed=42)...")
    ds = generate_dataset(n_nodes=4, n_graphs=200, seed=42)
    print(f"  val: {len(ds.val_labels)} examples")

    run_pad_ablation(
        checkpoint_paths=CHECKPOINTS,
        val_tokens=ds.val_tokens,
        val_labels=ds.val_labels,
        ablate_layer=1,
    )


if __name__ == "__main__":
    main()
