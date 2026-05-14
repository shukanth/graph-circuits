"""
Graph generation and tokenization.

Will contain:
- generate_graphs: sample random directed graphs via NetworkX
- shortest_path_labels: compute ground-truth BFS distances
- tokenize: encode edge sequences + (src, dst) query into integer token arrays
- dataset split: train/val split by graph structure, never by edge permutation
"""
