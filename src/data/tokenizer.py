from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Vocab:
    """
    Token vocabulary for shortest-path sequences.

    Layout: [0..n_nodes-1 = node IDs] [n_nodes = EDGE] [n_nodes+1 = QUERY]
            [n_nodes+2 = PAD] [n_nodes+3 = INF]

    INF is an output-only label — it never appears in input sequences.
    Keeping it in the same vocab makes the output head a single linear layer
    over vocab_size logits, which simplifies the model definition.
    """

    n_nodes: int

    @property
    def edge(self) -> int:
        return self.n_nodes

    @property
    def query(self) -> int:
        return self.n_nodes + 1

    @property
    def pad(self) -> int:
        return self.n_nodes + 2

    @property
    def inf(self) -> int:
        return self.n_nodes + 3

    @property
    def size(self) -> int:
        return self.n_nodes + 4


def encode(
    G: nx.DiGraph,
    src: int,
    dst: int,
    vocab: Vocab,
    max_edges: int,
    edge_order: np.ndarray,
) -> np.ndarray:
    """
    Encode a directed graph and (src, dst) query as a flat integer token sequence.

    Format: [EDGE u v] * n_edges + [PAD] * padding + [QUERY src dst]
    Length is always 3 * max_edges + 3 — fixed length is required for JIT compilation
    since JAX traces shapes statically.

    edge_order is a permutation of range(n_edges) passed by the caller so that
    randomness is explicit (JAX/pure-function convention).
    """
    edge_list = list(G.edges())
    assert len(edge_list) <= max_edges, (
        f"Graph has {len(edge_list)} edges but max_edges={max_edges}"
    )

    tokens = np.full(3 * max_edges + 3, vocab.pad, dtype=np.int32)

    for i, idx in enumerate(edge_order):
        u, v = edge_list[idx]
        tokens[3 * i] = vocab.edge
        tokens[3 * i + 1] = u
        tokens[3 * i + 2] = v

    query_start = 3 * max_edges
    tokens[query_start] = vocab.query
    tokens[query_start + 1] = src
    tokens[query_start + 2] = dst

    return tokens


def decode_edges(tokens: np.ndarray, vocab: Vocab, max_edges: int) -> list[tuple[int, int]]:
    """Extract the edge list from an encoded token sequence."""
    edges = []
    for i in range(0, 3 * max_edges, 3):
        if tokens[i] == vocab.edge:
            edges.append((int(tokens[i + 1]), int(tokens[i + 2])))
    return edges


def decode_query(tokens: np.ndarray, max_edges: int) -> tuple[int, int]:
    """Extract the (src, dst) query from an encoded token sequence."""
    query_start = 3 * max_edges
    return int(tokens[query_start + 1]), int(tokens[query_start + 2])
