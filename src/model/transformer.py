from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 2


class MultiHeadAttention(nnx.Module):
    """
    Multi-head self-attention with no causal mask.

    Full bidirectional attention because this is a classification task — every
    edge token needs to be able to influence the output position.

    No biases on Q/K/V/O projections — bias-free QKV is standard in
    mechanistic interpretability work because it keeps the circuit equations
    clean (no additive offset terms to track).
    """

    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs):
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.W_K = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.W_V = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.W_O = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, seq, _ = x.shape

        def split_heads(z: jnp.ndarray) -> jnp.ndarray:
            # (batch, seq, d_model) → (batch, n_heads, seq, d_head)
            return z.reshape(batch, seq, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        Q = split_heads(self.W_Q(x))
        K = split_heads(self.W_K(x))
        V = split_heads(self.W_V(x))

        scale = self.d_head ** -0.5
        scores = jnp.einsum("bhqd,bhkd->bhqk", Q, K) * scale  # (batch, heads, seq, seq)
        attn = jax.nn.softmax(scores, axis=-1)

        z = jnp.einsum("bhqk,bhkd->bhqd", attn, V)            # (batch, heads, seq, d_head)
        z = z.transpose(0, 2, 1, 3).reshape(batch, seq, -1)   # (batch, seq, d_model)
        return self.W_O(z)


class TransformerLayer(nnx.Module):
    """
    Pre-norm transformer layer: LayerNorm → attention → residual add. No MLP.

    Attention-only keeps this at the right complexity level for full circuit
    analysis (QK/OV decomposition, composition between layers).

    Pre-norm (normalize before attention, not after) is more stable during
    training than post-norm and is the modern standard.
    """

    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs):
        self.attn = MultiHeadAttention(d_model, n_heads, rngs)
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.attn(self.ln(x))


class ShortestPathTransformer(nnx.Module):
    """
    2-layer attention-only transformer for shortest-path length classification.

    Reads the full token sequence and predicts the answer from the final token
    position (the dst node token). The dst position can attend to all edge
    tokens across both layers to accumulate path-length information.
    """

    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.W_E = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.W_pos = nnx.Embed(cfg.seq_len, cfg.d_model, rngs=rngs)
        self.layers = nnx.List([TransformerLayer(cfg.d_model, cfg.n_heads, rngs) for _ in range(cfg.n_layers)])
        self.ln_final = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.W_U = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            tokens: integer array of shape (batch, seq_len)
        Returns:
            logits: float array of shape (batch, vocab_size)
        """
        seq = tokens.shape[1]
        x = self.W_E(tokens) + self.W_pos(jnp.arange(seq))

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.W_U(x[:, -1, :])  # read from the dst token position
