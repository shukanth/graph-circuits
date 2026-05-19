"""
Diagnostic script — run inside the venv:
    python scripts/diagnose.py

Tests each piece of the training pipeline in isolation. The first test that
fails identifies the exact source of the (0 <= -55) XLA abort.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import optax
import flax
print("=" * 60)
print(f"JAX     version: {jax.__version__}")
print(f"Flax    version: {flax.__version__}")
print(f"Optax   version: {optax.__version__}")
print(f"Devices: {jax.devices()}")
print("=" * 60)

from data.dataset import generate_dataset
from model.transformer import ModelConfig, ShortestPathTransformer

ds = generate_dataset(n_nodes=6, n_graphs=5, seed=42)
seq_len = 3 * ds.max_edges + 3
print(f"\nDataset: max_edges={ds.max_edges}, seq_len={seq_len}")
print(f"  train_tokens: shape={ds.train_tokens.shape}, dtype={ds.train_tokens.dtype}")
print(f"  train_labels: shape={ds.train_labels.shape}, dtype={ds.train_labels.dtype}")
print(f"  token value range: [{ds.train_tokens.min()}, {ds.train_tokens.max()}]")
print(f"  label value range: [{ds.train_labels.min()}, {ds.train_labels.max()}]")

cfg = ModelConfig(vocab_size=ds.vocab.size, seq_len=seq_len)
model = ShortestPathTransformer(cfg=cfg)
dummy = jnp.zeros((1, seq_len), dtype=jnp.int32)
params = model.init(jax.random.PRNGKey(0), dummy)
print(f"\nmodel.init: OK (seq_len={seq_len}, vocab_size={ds.vocab.size})")

tokens = jnp.array(ds.train_tokens[:4], dtype=jnp.int32)
labels = jnp.array(ds.train_labels[:4])
print(f"Batch: tokens={tokens.shape} dtype={tokens.dtype}  labels={labels.shape} dtype={labels.dtype}")

# ── Test 1: forward pass, no JIT ─────────────────────────────────────────────
print("\n[1] Forward pass WITHOUT jit ...", end=" ", flush=True)
logits = model.apply(params, tokens)
print(f"OK  logits={logits.shape}")

# ── Test 2: forward pass, WITH JIT ───────────────────────────────────────────
print("[2] Forward pass WITH jit ...", end=" ", flush=True)
jit_fwd = jax.jit(model.apply)
logits_jit = jit_fwd(params, tokens)
print(f"OK  logits={logits_jit.shape}")

# ── Test 3: value_and_grad, no JIT ───────────────────────────────────────────
print("[3] value_and_grad WITHOUT jit ...", end=" ", flush=True)
def loss_fn(p):
    logits = model.apply(p, tokens)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, logits
(loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
print(f"OK  loss={float(loss):.4f}")

# ── Test 4: value_and_grad, WITH JIT (no optimizer) ──────────────────────────
print("[4] value_and_grad WITH jit (no optimizer) ...", end=" ", flush=True)
@jax.jit
def jit_grad_step(p, t, lbl):
    def lf(p):
        logits = model.apply(p, t)
        return optax.softmax_cross_entropy_with_integer_labels(logits, lbl).mean(), logits
    (loss, logits), _ = jax.value_and_grad(lf, has_aux=True)(p)
    preds = jnp.argmax(logits, axis=-1)
    return loss, preds
loss_jit, preds = jit_grad_step(params, tokens, labels)
print(f"OK  loss={float(loss_jit):.4f}")

# ── Test 5: full train_step with optimizer ────────────────────────────────────
print("[5] Full train_step WITH jit + optimizer ...", end=" ", flush=True)
tx = optax.adamw(learning_rate=1e-3, weight_decay=1.0)
opt_state = tx.init(params)

@jax.jit
def full_train_step(params, opt_state, tokens, labels):
    def lf(p):
        logits = model.apply(p, tokens)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(lf, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    preds = jnp.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    grad_norm = optax.global_norm(grads)
    return new_params, new_opt_state, loss, acc, grad_norm

new_params, new_opt_state, loss, acc, grad_norm = full_train_step(
    params, opt_state, tokens, labels
)
print(f"OK  loss={float(loss):.4f}  acc={float(acc):.3f}")

# ── Test 6a: value_and_grad WITH jit, batch=512 (model only, no optimizer) ───
print("[6a] value_and_grad WITH jit, batch=512 (no optimizer) ...", end=" ", flush=True)
tokens_512 = jnp.zeros((512, seq_len), dtype=jnp.int32)
labels_512 = jnp.zeros((512,), dtype=jnp.int32)

@jax.jit
def jit_grad_512(p, t, lbl):
    def lf(p):
        logits = model.apply(p, t)
        return optax.softmax_cross_entropy_with_integer_labels(logits, lbl).mean(), logits
    (loss, logits), _ = jax.value_and_grad(lf, has_aux=True)(p)
    return loss, jnp.argmax(logits, axis=-1)

loss_512a, _ = jit_grad_512(params, tokens_512, labels_512)
print(f"OK  loss={float(loss_512a):.4f}")

# ── Test 6b: find max stable batch size (blocking on every result) ─────────
# NOTE: ds here has only 120 train samples (n_graphs=5). We use jnp.zeros to
# test genuine batch sizes larger than the dataset, so compilation sees the
# real shape. float(loss) blocks until the XLA computation completes.
print("[6b] Finding max stable batch size (blocking) ...")
working_bs = None
for bs in [8, 32, 64, 128]:
    print(f"     batch={bs} ...", end=" ", flush=True)
    t_bs = jnp.zeros((bs, seq_len), dtype=jnp.int32)
    l_bs = jnp.zeros((bs,), dtype=jnp.int32)
    try:
        _, _, loss_bs, acc_bs, gnorm_bs = full_train_step(params, opt_state, t_bs, l_bs)
        _ = float(loss_bs), float(acc_bs), float(gnorm_bs)  # force completion
        print(f"OK  loss={float(loss_bs):.4f}")
        working_bs = bs
    except Exception as e:
        print(f"FAIL ({e})")
        break
print(f"     → max confirmed-stable batch size: {working_bs}")

print("\n" + "=" * 60)
print("All tests passed.")
