"""
Run the logit lens analysis for Run 5 (n4-wd0.4-200k-seed0).

Compares per-category accuracy (dist=1, dist=2, dist=3, INF) at three
depths: embedding only, after Layer 0, and full model.  Tests the
hypothesis that Layer 0 handles distance-1 cases and Layer 1 extends
to multi-hop paths.

Usage (from repo root):
    python scripts/run_logit_lens.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["JAX_PLATFORMS"] = "cpu"

from data.dataset import generate_dataset
from interp.logit_lens import run_logit_lens

CHECKPOINT_DIR = "checkpoints"
CHECKPOINTS = [
    ("126k", os.path.join(CHECKPOINT_DIR, "step_0126000.pkl")),
    ("196k", os.path.join(CHECKPOINT_DIR, "step_0196000.pkl")),
]


def main() -> None:
    print("Generating dataset (n_nodes=4, n_graphs=200, seed=42)...")
    ds = generate_dataset(n_nodes=4, n_graphs=200, seed=42)
    run_logit_lens(CHECKPOINTS, ds.val_tokens, ds.val_labels)


if __name__ == "__main__":
    main()
