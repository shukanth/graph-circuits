"""
Correctness tests.

Priority tests:
- Dataset generator: ground-truth labels match NetworkX BFS
- No graph leaks across train/val split
- Tokenization round-trips (encode → decode → same graph)
- Model output shape sanity checks
"""
