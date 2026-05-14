"""
Transformer definition in Flax.

Will contain:
- ShortestPathTransformer: 2-layer, attention-only, ~100k params
- Embedding, positional encoding, attention head, and output projection modules
- No MLPs — attention-only is the right complexity for full circuit analysis
"""
