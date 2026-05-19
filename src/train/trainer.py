"""
Training loop for ShortestPathTransformer.

Uses Flax Linen's functional API: model.init(key, tokens) returns a params
dict; model.apply(params, tokens) runs the forward pass. Both are pure
functions that compose naturally with @jax.jit and jax.value_and_grad.

Why Linen instead of nnx:
  In Flax 0.11.0, nnx.merge inside any @jax.jit context (whether @nnx.jit or
  @jax.jit) triggers an XLA shape-inference abort: "(0 <= -55) is statically
  false". The root cause is that nnx.merge performs graph reconstruction that
  conflicts with XLA's static shape tracing. Linen's model.apply has no such
  issue — it is the canonical pattern for functional JAX and has been stable
  for years.

Why model is captured as a closure variable in train_step:
  model is a Linen module — a static Python object (no JAX arrays). JAX treats
  non-array closure variables as compile-time constants, so @jax.jit reuses the
  compiled binary every step without retracing.

AdamW with weight_decay~1.0 is the known-good setting for grokking
(Power et al., 2022). Weight decay creates pressure toward lower-norm
solutions: the model first memorizes (large, complex weights) then, after
extended training, weight decay wins and forces a lower-norm generalizing
solution. This is the grokking mechanism.
"""

import os
import pickle
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from data.dataset import Dataset
from model.transformer import ModelConfig, ShortestPathTransformer


@dataclass
class TrainConfig:
    # Optimizer — weight_decay~1.0 is the grokking sweet spot
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    # Training schedule
    n_steps: int = 100_000
    # JAX 0.10.0 XLA shape-inference abort on Apple Silicon CPU backend:
    # the combined model+optimizer JIT fails for batch >= 256.
    # 128 is the confirmed-stable maximum.
    batch_size: int = 128
    # Logging — every 500 steps captures fine-grained grokking dynamics
    log_every: int = 500
    # Checkpointing — every 2k steps gives dense coverage of the transition
    checkpoint_every: int = 2_000
    checkpoint_dir: str = "checkpoints"
    # Experiment tracking
    wandb_project: str = "mechinterp"
    run_name: str = ""  # empty = W&B auto-generates a name
    seed: int = 0
    use_wandb: bool = True


def save_checkpoint(params: dict, path: str) -> None:
    """Save params dict to disk as a pickle of numpy arrays.

    params is a plain JAX pytree (nested dict from Linen). Converts JAX arrays
    to numpy for portability — interpretability scripts can load this without
    a GPU. The treedef preserves the pytree structure for correct round-tripping.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    np_leaves = [np.array(leaf) for leaf in leaves]
    with open(path, "wb") as f:
        pickle.dump({"leaves": np_leaves, "treedef": treedef}, f)


def load_checkpoint(path: str) -> dict:
    """Load a params dict from a checkpoint file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    jax_leaves = [jnp.array(leaf) for leaf in data["leaves"]]
    return jax.tree_util.tree_unflatten(data["treedef"], jax_leaves)


def train(
    cfg: TrainConfig,
    ds: Dataset,
    model_cfg: ModelConfig,
) -> tuple[ShortestPathTransformer, dict]:
    """Run the full training loop. Returns (model, params) for interpretability use."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model = ShortestPathTransformer(cfg=model_cfg)

    # Initialize params with a dummy forward pass.
    # model.init returns {'params': {...}} — a nested dict of all weight arrays.
    dummy_tokens = jnp.zeros((1, model_cfg.seq_len), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(cfg.seed), dummy_tokens)

    tx = optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, tokens, labels):
        def loss_fn(p):
            logits = model.apply(p, tokens)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss, logits

        # has_aux=True: loss_fn returns (scalar, logits); grad is wrt the scalar only
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # params must be passed to tx.update for AdamW's weight decay term:
        # weight_decay gradient = wd * params, added before the Adam step
        updates, new_opt_state = tx.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)

        preds = jnp.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        grad_norm = optax.global_norm(grads)

        return new_params, new_opt_state, loss, acc, grad_norm

    @jax.jit
    def eval_step(params, tokens, labels):
        logits = model.apply(params, tokens)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return loss, acc

    def compute_val_metrics(params, tokens, labels):
        """Compute loss and accuracy over an entire split by iterating in batches."""
        n = len(tokens)
        total_loss = 0.0
        total_correct = 0
        for i in range(0, n, cfg.batch_size):
            t = tokens[i : i + cfg.batch_size]
            lbl = labels[i : i + cfg.batch_size]
            loss, acc = eval_step(params, t, lbl)
            bsz = len(t)
            total_loss += float(loss) * bsz
            total_correct += round(float(acc) * bsz)
        return total_loss / n, total_correct / n

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name or None,
            config={
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "n_steps": cfg.n_steps,
                "batch_size": cfg.batch_size,
                "d_model": model_cfg.d_model,
                "n_heads": model_cfg.n_heads,
                "n_layers": model_cfg.n_layers,
                "vocab_size": model_cfg.vocab_size,
                "seq_len": model_cfg.seq_len,
                "seed": cfg.seed,
            },
        )

    # Move dataset to JAX arrays once — avoids repeated numpy→jax copies per step
    train_tokens = jnp.array(ds.train_tokens)
    train_labels = jnp.array(ds.train_labels)
    val_tokens = jnp.array(ds.val_tokens)
    val_labels = jnp.array(ds.val_labels)
    n_train = len(train_tokens)

    # numpy rng for batch sampling — kept separate from JAX rng
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    for step in range(1, cfg.n_steps + 1):
        idx = rng.integers(0, n_train, size=cfg.batch_size)
        params, opt_state, loss, acc, grad_norm = train_step(
            params, opt_state, train_tokens[idx], train_labels[idx]
        )

        log_interval = 10 if step < 1000 else cfg.log_every
        if step % log_interval == 0:
            val_loss, val_acc = compute_val_metrics(params, val_tokens, val_labels)
            elapsed = time.time() - t0
            print(
                f"step {step:>6d} | "
                f"train loss {float(loss):.4f} | train acc {float(acc):.3f} | "
                f"val loss {val_loss:.4f} | val acc {val_acc:.3f} | "
                f"‖g‖ {float(grad_norm):.4f} | {elapsed:.0f}s"
            )
            if cfg.use_wandb:
                wandb.log(
                    {
                        "train/loss": float(loss),
                        "train/acc": float(acc),
                        "train/grad_norm": float(grad_norm),
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                    },
                    step=step,
                )

        if step % cfg.checkpoint_every == 0:
            ckpt = os.path.join(cfg.checkpoint_dir, f"step_{step:07d}.pkl")
            save_checkpoint(params, ckpt)

    # Final checkpoint
    save_checkpoint(
        params,
        os.path.join(cfg.checkpoint_dir, f"step_{cfg.n_steps:07d}_final.pkl"),
    )

    if cfg.use_wandb:
        wandb.finish()

    return model, params
