"""
Shortest-path transformer defined with Flax Linen.

Linen (flax.linen) is the functional-JAX API: the forward pass is a pure
function `model.apply(params, tokens)` where params is an explicit dict
argument rather than state living inside the module. This fits @jax.jit
naturally — pass params in, get outputs out, no mutation.

The nnx API (flax.nnx) is the stateful API designed to feel like PyTorch.
It wraps params inside module objects and uses @nnx.jit to extract/re-inject
them. In Flax 0.11.0, nnx.merge inside @jax.jit triggers an XLA shape
inference abort (`(0 <= -55) is statically false`). Linen has no such issue.

Architecture choices:
- Attention-only (no MLP): right complexity level for full circuit analysis.
- No causal mask: bidirectional attention because this is classification, not
  generation — every edge token needs to influence the output position.
- No bias on Q/K/V/O: bias-free keeps circuit equations clean for Phase 3
  interpretability work.
- Pre-norm: LayerNorm before attention, not after. More training-stable.
- Read output from the last token position (the dst node token).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclass(frozen=True)
class ModelConfig:
    """
    frozen=True makes ModelConfig hashable, which is required when passing it
    as an attribute of a Flax Linen module (Linen modules need hashable fields).
    """
    vocab_size: int
    seq_len: int
    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 2


class MultiHeadAttention(nn.Module):
    d_model: int
    n_heads: int

    def setup(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        )
        self.W_Q = nn.Dense(self.d_model, use_bias=False)
        self.W_K = nn.Dense(self.d_model, use_bias=False)
        self.W_V = nn.Dense(self.d_model, use_bias=False)
        self.W_O = nn.Dense(self.d_model, use_bias=False)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, seq, _ = x.shape
        d_head = self.d_model // self.n_heads

        # Reshape only — no transpose at all. Q/K/V stay in (batch, seq, heads, d_head)
        # order ("bqhd") throughout. All transposes are eliminated to work around a
        # JAX 0.10.0 XLA bug where transpose→reshape (in either direction) during the
        # backward pass produces a negative slice start index at batch >= 512 on CPU,
        # aborting the process with "(0 <= -N) is statically false".
        Q = self.W_Q(x).reshape(batch, seq, self.n_heads, d_head)  # "bqhd"
        K = self.W_K(x).reshape(batch, seq, self.n_heads, d_head)  # "bkhd"
        V = self.W_V(x).reshape(batch, seq, self.n_heads, d_head)  # "bkhd"

        scale = d_head ** -0.5
        # Q "bqhd", K "bkhd" → scores "bhqk": contracts over d only.
        scores = jnp.einsum("bqhd,bkhd->bhqk", Q, K) * scale
        attn = jax.nn.softmax(scores, axis=-1)

        # attn "bhqk", V "bkhd" → z "bqhd": contracts over k only.
        z = jnp.einsum("bhqk,bkhd->bqhd", attn, V)
        z = z.reshape(batch, seq, self.d_model)
        return self.W_O(z)


class TransformerLayer(nn.Module):
    d_model: int
    n_heads: int

    def setup(self):
        self.attn = MultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads)
        self.ln = nn.LayerNorm()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.attn(self.ln(x))


class ShortestPathTransformer(nn.Module):
    cfg: ModelConfig

    def setup(self):
        self.W_E = nn.Embed(self.cfg.vocab_size, self.cfg.d_model)
        self.W_pos = nn.Embed(self.cfg.seq_len, self.cfg.d_model)
        # List of submodules — Linen names them layers_0, layers_1, ... in the
        # params dict, which is exactly the structure we want for Phase 3 analysis.
        self.layers = [
            TransformerLayer(d_model=self.cfg.d_model, n_heads=self.cfg.n_heads)
            for _ in range(self.cfg.n_layers)
        ]
        self.ln_final = nn.LayerNorm()
        self.W_U = nn.Dense(self.cfg.vocab_size, use_bias=False)

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
        return self.W_U(x[:, seq - 1, :])  # read from the dst token position
