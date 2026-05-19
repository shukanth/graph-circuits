"""
Entry point for training ShortestPathTransformer.

Every argument here is a meaningful experimental variable. The three most
important for grokking research are:
  --weight_decay  the primary grokking dial — controls when/whether it happens
  --n_graphs      dataset size — smaller datasets grok faster
  --seed          run multiple seeds to verify results aren't seed-dependent

dataset_seed and seed are intentionally separate: dataset_seed controls which
graphs are generated; seed controls model initialisation and batch order. This
lets you hold the dataset fixed across model seeds (reproducibility check) or
hold the model seed fixed across datasets (dataset-size sweep).
"""

import argparse
import os
import sys

# Fallback in case the package is not installed via pip install -e .
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Force CPU unconditionally. jax-metal (Apple Silicon GPU backend) has XLA bugs
# that produce shape-check aborts (e.g. "(0 <= -55) is statically false") during
# JIT compilation. This must be a hard assignment, not setdefault, because
# jax-metal may register itself as the default backend even without an explicit
# env var — setdefault would not override that registration.
os.environ["JAX_PLATFORMS"] = "cpu"

from data.dataset import generate_dataset
from model.transformer import ModelConfig
from train.trainer import TrainConfig, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ShortestPathTransformer on shortest-path classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset ---
    p.add_argument(
        "--n_nodes", type=int, default=6,
        help="Nodes per graph. Controls task complexity, vocab size, and sequence length.",
    )
    p.add_argument(
        "--n_graphs", type=int, default=1000,
        help="Number of graphs. Total samples = n_graphs * n_nodes * (n_nodes - 1).",
    )
    p.add_argument(
        "--dataset_seed", type=int, default=42,
        help="Seed for dataset generation. Separate from --seed so you can hold the "
             "dataset fixed while varying model initialisation.",
    )

    # --- Optimizer ---
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="AdamW learning rate.",
    )
    p.add_argument(
        "--weight_decay", type=float, default=1.0,
        help="AdamW weight decay. ~1.0 is the grokking sweet spot; sweep this first "
             "when diagnosing missing transitions.",
    )

    # --- Training schedule ---
    p.add_argument(
        "--n_steps", type=int, default=100_000,
        help="Total gradient steps. Grokking may need 50k–200k steps; set generously.",
    )
    p.add_argument(
        "--batch_size", type=int, default=128,
        help="Samples per gradient step. JAX 0.10.0 on Apple Silicon CPU aborts "
             "for batch >= 256; 128 is the confirmed-stable maximum.",
    )

    # --- Logging and checkpointing ---
    p.add_argument(
        "--log_every", type=int, default=100,
        help="Steps between W&B log entries.",
    )
    p.add_argument(
        "--checkpoint_every", type=int, default=2_000,
        help="Steps between checkpoint saves.",
    )
    p.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory for checkpoint .pkl files.",
    )

    # --- Experiment tracking ---
    p.add_argument(
        "--wandb_project", type=str, default="mechinterp",
        help="W&B project name.",
    )
    p.add_argument(
        "--run_name", type=str, default="",
        help="W&B run name. Empty string lets W&B auto-generate one.",
    )
    p.add_argument(
        "--no_wandb", action="store_true",
        help="Disable W&B logging (useful for quick local debugging runs).",
    )

    # --- Reproducibility ---
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for model initialisation and batch sampling.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Generating dataset: n_nodes={args.n_nodes}, n_graphs={args.n_graphs}, "
          f"dataset_seed={args.dataset_seed}")
    ds = generate_dataset(
        n_nodes=args.n_nodes,
        n_graphs=args.n_graphs,
        seed=args.dataset_seed,
    )
    seq_len = 3 * ds.max_edges + 3
    print(
        f"  train: {len(ds.train_tokens):,} samples | "
        f"val: {len(ds.val_tokens):,} samples | "
        f"vocab size: {ds.vocab.size} | seq len: {seq_len}"
    )

    model_cfg = ModelConfig(
        vocab_size=ds.vocab.size,
        seq_len=seq_len,
    )

    train_cfg = TrainConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )

    print(
        f"Training: lr={args.lr}, wd={args.weight_decay}, "
        f"n_steps={args.n_steps:,}, seed={args.seed}"
    )
    train(train_cfg, ds, model_cfg)


if __name__ == "__main__":
    main()
