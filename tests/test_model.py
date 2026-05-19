import jax
import jax.numpy as jnp
import pytest

from data.dataset import generate_dataset
from model.transformer import ModelConfig, ShortestPathTransformer

VOCAB_SIZE = 10  # n_nodes=6 → vocab_size = n_nodes + 4
SEQ_LEN = 48    # 3 * max_edges + 3 = 3 * 15 + 3


def make_model_and_params(seed: int = 0):
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    model = ShortestPathTransformer(cfg=cfg)
    dummy = jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(seed), dummy)
    return model, params


def test_output_shape():
    model, params = make_model_and_params()
    tokens = jnp.zeros((4, SEQ_LEN), dtype=jnp.int32)
    logits = model.apply(params, tokens)
    assert logits.shape == (4, VOCAB_SIZE)


def test_output_is_finite():
    """A freshly initialised model should produce finite logits."""
    model, params = make_model_and_params()
    tokens = jnp.zeros((2, SEQ_LEN), dtype=jnp.int32)
    logits = model.apply(params, tokens)
    assert jnp.all(jnp.isfinite(logits))


def test_logits_vary_with_input():
    """Different token sequences must produce different logits."""
    model, params = make_model_and_params()
    tokens_a = jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)
    tokens_b = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    assert not jnp.allclose(model.apply(params, tokens_a), model.apply(params, tokens_b))


def test_parameter_count():
    """Expected ~140k params for d_model=128, n_heads=2, n_layers=2."""
    model, params = make_model_and_params()
    n_params = sum(x.size for x in jax.tree.leaves(params))
    assert 100_000 <= n_params <= 200_000, f"Unexpected param count: {n_params:,}"


def test_model_accepts_dataset_tokens():
    """Model input/output shapes must be compatible with the dataset generator."""
    ds = generate_dataset(n_nodes=6, n_graphs=5, seed=0)
    cfg = ModelConfig(vocab_size=ds.vocab.size, seq_len=3 * ds.max_edges + 3)
    model = ShortestPathTransformer(cfg=cfg)
    dummy = jnp.zeros((1, cfg.seq_len), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy)
    tokens = jnp.array(ds.train_tokens[:8])
    logits = model.apply(params, tokens)
    assert logits.shape == (8, ds.vocab.size)
