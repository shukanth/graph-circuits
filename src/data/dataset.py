from dataclasses import dataclass

import numpy as np

from data.graphs import all_pair_distances, sample_graph
from data.tokenizer import Vocab, encode


@dataclass
class Dataset:
    train_tokens: np.ndarray     # (n_train, seq_len)
    train_labels: np.ndarray     # (n_train,)
    train_graph_ids: np.ndarray  # (n_train,) — which graph each sample came from
    val_tokens: np.ndarray       # (n_val, seq_len)
    val_labels: np.ndarray       # (n_val,)
    val_graph_ids: np.ndarray    # (n_val,)
    vocab: Vocab
    max_edges: int


def generate_dataset(
    n_nodes: int,
    n_graphs: int,
    density_range: tuple[float, float] = (0.27, 0.50),
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Dataset:
    """
    Generate a dataset of shortest-path queries on random directed graphs.

    The train/val split is by graph identity — all queries for a given graph go
    to the same split. This tests generalization to unseen graph topologies, not
    just unseen query orderings on seen graphs.

    Edges are shuffled independently per sample so the model cannot rely on a
    consistent edge ordering as a shortcut to the answer.
    """
    rng = np.random.default_rng(seed)
    vocab = Vocab(n_nodes)
    max_edges = int(density_range[1] * n_nodes * (n_nodes - 1))

    # Assign graphs to splits before generating any data to prevent leakage.
    graph_indices = rng.permutation(n_graphs)
    n_val = max(1, int(val_fraction * n_graphs))
    val_graph_set = set(graph_indices[:n_val].tolist())

    train_tokens, train_labels, train_graph_ids = [], [], []
    val_tokens, val_labels, val_graph_ids = [], [], []

    for graph_id in range(n_graphs):
        density = rng.uniform(*density_range)
        G = sample_graph(n_nodes, density, rng)
        distances = all_pair_distances(G)
        n_edges = G.number_of_edges()

        is_val = graph_id in val_graph_set

        for src in range(n_nodes):
            for dst in range(n_nodes):
                if src == dst:
                    continue

                dist = distances[(src, dst)]
                label = vocab.inf if dist is None else dist

                edge_order = rng.permutation(n_edges)
                tokens = encode(G, src, dst, vocab, max_edges, edge_order)

                if is_val:
                    val_tokens.append(tokens)
                    val_labels.append(label)
                    val_graph_ids.append(graph_id)
                else:
                    train_tokens.append(tokens)
                    train_labels.append(label)
                    train_graph_ids.append(graph_id)

    return Dataset(
        train_tokens=np.stack(train_tokens),
        train_labels=np.array(train_labels, dtype=np.int32),
        train_graph_ids=np.array(train_graph_ids, dtype=np.int32),
        val_tokens=np.stack(val_tokens),
        val_labels=np.array(val_labels, dtype=np.int32),
        val_graph_ids=np.array(val_graph_ids, dtype=np.int32),
        vocab=vocab,
        max_edges=max_edges,
    )
