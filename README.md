# Mechanistic Interpretability of Shortest-Path Reasoning

A small transformer trained from scratch to compute shortest-path lengths on directed graphs, then fully reverse-engineered using mechanistic interpretability techniques.

The central question: **what algorithm does the model actually learn?**

---

## Finding

The fully trained model implements a **partial two-step backward BFS from the destination node**, one BFS step per layer.

- **Layer 0** (head L0H1): a token-identity QK circuit routes the destination position's attention toward every edge triplet in which `dst` appears. The OV contribution writes a neighborhood signal — where in the sequence dst's edges sit — into the residual stream.
- **Layer 1**: reads that signal and checks whether `src` can reach any of `dst`'s direct predecessors. This handles distance-1 (direct connection confirmed) and distance-2 (one intermediate hop confirmed) well, and correctly identifies unreachability (INF). Distance-3 paths exceed the architecture's depth and are systematically misclassified.

The training curve shows two accuracy jumps (a double-grokking). These are mechanistically distinct:
- **Stage 1** (steps 114k–126k): Layer 1 discovers a statistical shortcut — attending to PAD-token positions injects a constant logit bias toward longer distances and INF. This is a matrix-alignment event invisible in individual weight norms.
- **Stage 2** (steps 188k–196k): the shortcut is partially dismantled and replaced by the genuine BFS circuit above.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+. GPU training uses Apple Silicon Metal via `jax-metal`; all interpretability scripts run on CPU.

---

## Repository structure

```
src/
  data/          graph generation, tokenization, dataset assembly
  model/         2-layer attention-only transformer (Flax Linen)
  train/         JAX/Optax training loop, checkpointing, W&B logging
  interp/        interpretability toolkit (Phase 3)
    weight_analysis.py   Frobenius norm trajectories across checkpoints
    attention_viz.py     attention pattern visualization, probe comparisons
    ov_circuit.py        OV circuit matrices, PAD value analysis, QK circuits
    ablation.py          PAD-position ablation experiment (numpy forward pass)
    logit_lens.py        per-layer accuracy by answer category

scripts/
  train.py              training entry point
  run_ablation.py       PAD ablation across 4 checkpoints
  run_qk_analysis.py    QK circuit analysis (L0H1 at 126k vs 196k)
  run_logit_lens.py     logit lens — per-category accuracy breakdown
  run_l0h1_ov.py        L0H1 full OV circuit analysis

notebooks/
  documentation.tex     full technical writeup (the single source of truth)
  weight_norms.png      norm trajectory plot
  ov_circuits.png       OV circuit heatmaps (Stage 2 window)
  qk_circuits.png       QK circuit heatmaps (126k vs 196k)
  ov_l0_circuits.png    L0 OV circuit heatmaps

tests/
  test_data.py          14 data pipeline tests
  test_model.py         5 model tests
```

---

## Training

```bash
# Train Run 5 (n_nodes=4, weight_decay=0.4, 200k steps)
python scripts/train.py

# Checkpoints are saved to checkpoints/ every 2000 steps (gitignored)
```

The model is a 2-layer, 2-head attention-only transformer: `d_model=128`, `d_head=64`, 140,288 parameters. Target task: given a shuffled sequence of directed edge tokens plus a source and destination node, predict the shortest-path length (1, 2, 3, or INF).

---

## Interpretability pipeline

Run these from the repository root after training.

```bash
# Causal validation: does PAD-position attention carry the shortcut?
python scripts/run_ablation.py

# What does L0H1 attend to? (QK circuit)
python scripts/run_qk_analysis.py

# Which layer handles which distance category? (logit lens)
python scripts/run_logit_lens.py

# What does L0H1 write? (OV circuit)
python scripts/run_l0h1_ov.py
```

The full technical documentation — circuit derivations, all 31 numbered findings, tables, and the algorithmic narrative — is in `notebooks/documentation.tex`.

---

## Task

**Input:** a directed graph on 4 nodes, encoded as a shuffled sequence of `[EDGE, src, dst]` triplets padded to fixed length, followed by `[QRY, src, dst]`.

**Output:** the shortest-path length from src to dst (tokens `1`, `2`, `3`, or `INF`).

Directed graphs are used deliberately: asymmetric reachability (`A→B` does not imply `B→A`) blocks symmetry shortcuts and forces the model to learn a richer algorithm.

---

## References

- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177, 2022.
- Elhage et al., *A Mathematical Framework for Transformer Circuits*, Transformer Circuits Thread, 2021.
- nostalgebraist, *Interpreting GPT: the logit lens*, LessWrong, 2020.
