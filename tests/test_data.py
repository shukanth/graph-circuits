import numpy as np
import networkx as nx
import pytest

from data.graphs import all_pair_distances, sample_graph
from data.tokenizer import Vocab, decode_edges, decode_query, encode
from data.dataset import generate_dataset


# ── graph generation ──────────────────────────────────────────────────────────

def test_sample_graph_node_count():
    rng = np.random.default_rng(0)
    G = sample_graph(6, 0.4, rng)
    assert G.number_of_nodes() == 6


def test_sample_graph_no_self_loops():
    rng = np.random.default_rng(0)
    G = sample_graph(6, 0.4, rng)
    assert all(u != v for u, v in G.edges())


def test_sample_graph_edge_count():
    rng = np.random.default_rng(0)
    n_nodes, density = 6, 0.4
    G = sample_graph(n_nodes, density, rng)
    expected = round(density * n_nodes * (n_nodes - 1))
    assert G.number_of_edges() == expected


# ── ground-truth labels ───────────────────────────────────────────────────────

def test_labels_match_networkx():
    """Our distance function must agree with nx.shortest_path_length on every pair."""
    rng = np.random.default_rng(1)
    G = sample_graph(6, 0.4, rng)
    distances = all_pair_distances(G)

    for src in range(6):
        nx_lengths = nx.single_source_shortest_path_length(G, src)
        for dst in range(6):
            if src == dst:
                continue
            expected = nx_lengths.get(dst, None)
            assert distances[(src, dst)] == expected


def test_unreachable_returns_none():
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    distances = all_pair_distances(G)
    assert all(v is None for v in distances.values())


def test_known_path_length():
    """Hand-constructed graph with a known shortest path."""
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])  # chain: 0→1→2→3
    distances = all_pair_distances(G)
    assert distances[(0, 3)] == 3
    assert distances[(3, 0)] is None  # directed — no path back


# ── tokenization ──────────────────────────────────────────────────────────────

def test_sequence_length():
    vocab = Vocab(6)
    max_edges = 15
    rng = np.random.default_rng(2)
    G = sample_graph(6, 0.4, rng)
    edge_order = rng.permutation(G.number_of_edges())
    tokens = encode(G, 0, 1, vocab, max_edges, edge_order)
    assert len(tokens) == 3 * max_edges + 3


def test_tokenization_roundtrip():
    """Encode then decode must recover the original edge set and query."""
    vocab = Vocab(6)
    max_edges = 15
    rng = np.random.default_rng(3)
    G = sample_graph(6, 0.4, rng)
    src, dst = 0, 3
    edge_order = np.arange(G.number_of_edges())  # identity — deterministic order
    tokens = encode(G, src, dst, vocab, max_edges, edge_order)

    assert set(decode_edges(tokens, vocab, max_edges)) == set(G.edges())

    decoded_src, decoded_dst = decode_query(tokens, max_edges)
    assert decoded_src == src
    assert decoded_dst == dst


def test_node_tokens_are_valid_node_ids():
    """The u, v positions in every EDGE triple must be valid node IDs."""
    vocab = Vocab(6)
    max_edges = 15
    rng = np.random.default_rng(4)
    G = sample_graph(6, 0.4, rng)
    edge_order = rng.permutation(G.number_of_edges())
    tokens = encode(G, 0, 1, vocab, max_edges, edge_order)

    for i in range(0, 3 * max_edges, 3):
        if tokens[i] == vocab.edge:
            assert tokens[i + 1] < vocab.n_nodes
            assert tokens[i + 2] < vocab.n_nodes


# ── dataset ───────────────────────────────────────────────────────────────────

def test_no_graph_leaks():
    """No graph ID should appear in both train and val splits."""
    ds = generate_dataset(n_nodes=6, n_graphs=20, seed=0)
    train_ids = set(ds.train_graph_ids.tolist())
    val_ids = set(ds.val_graph_ids.tolist())
    assert train_ids.isdisjoint(val_ids)


def test_label_values_in_range():
    """Every label must be either a valid path length (0..n_nodes-1) or INF."""
    ds = generate_dataset(n_nodes=6, n_graphs=10, seed=0)
    vocab = ds.vocab
    all_labels = np.concatenate([ds.train_labels, ds.val_labels])
    valid = ((all_labels >= 0) & (all_labels < vocab.n_nodes)) | (all_labels == vocab.inf)
    assert valid.all()


def test_inf_ratio():
    """
    INF fraction should be in a healthy range.
    Too low → model ignores reachability. Too high → model ignores path lengths.
    """
    ds = generate_dataset(n_nodes=6, n_graphs=100, seed=0)
    all_labels = np.concatenate([ds.train_labels, ds.val_labels])
    inf_ratio = (all_labels == ds.vocab.inf).mean()
    assert 0.10 <= inf_ratio <= 0.60, f"INF ratio {inf_ratio:.2f} outside expected [0.10, 0.60]"


def test_dataset_tokens_shape():
    n_nodes, n_graphs = 6, 20
    ds = generate_dataset(n_nodes=n_nodes, n_graphs=n_graphs, seed=0)
    seq_len = 3 * ds.max_edges + 3
    assert ds.train_tokens.shape[1] == seq_len
    assert ds.val_tokens.shape[1] == seq_len
    assert ds.train_tokens.shape[0] == len(ds.train_labels)
    assert ds.val_tokens.shape[0] == len(ds.val_labels)


def test_total_sample_count():
    """Total samples = n_graphs * n_nodes * (n_nodes - 1) (all pairs, src != dst)."""
    n_nodes, n_graphs = 6, 20
    ds = generate_dataset(n_nodes=n_nodes, n_graphs=n_graphs, seed=0)
    total = len(ds.train_labels) + len(ds.val_labels)
    assert total == n_graphs * n_nodes * (n_nodes - 1)
