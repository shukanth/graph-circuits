from typing import Optional

import networkx as nx
import numpy as np


def sample_graph(n_nodes: int, density: float, rng: np.random.Generator) -> nx.DiGraph:
    """
    Sample a random directed graph with approximately `density` * max_edges edges.

    Using density (fraction of max possible edges) rather than a fixed count so the
    generator scales cleanly when n_nodes increases without re-tuning the range.
    """
    max_edges = n_nodes * (n_nodes - 1)  # directed, no self-loops
    n_edges = max(1, round(density * max_edges))

    all_edges = [(u, v) for u in range(n_nodes) for v in range(n_nodes) if u != v]
    chosen_idx = rng.choice(len(all_edges), size=n_edges, replace=False)

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from([all_edges[i] for i in chosen_idx])
    return G


def all_pair_distances(G: nx.DiGraph) -> dict[tuple[int, int], Optional[int]]:
    """
    Compute shortest-path length for every (src, dst) pair where src != dst.
    Returns None for unreachable pairs.

    This is our ground truth — correctness here is critical since a silent bug
    poisons every label in the dataset.
    """
    n = G.number_of_nodes()
    distances: dict[tuple[int, int], Optional[int]] = {}

    for src in range(n):
        lengths = nx.single_source_shortest_path_length(G, src)
        for dst in range(n):
            if dst != src:
                distances[(src, dst)] = lengths.get(dst, None)

    return distances
