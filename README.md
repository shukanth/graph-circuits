# Mechanistic Interpretability of Shortest-Path Reasoning

A small transformer trained from scratch to compute shortest-path lengths on directed graphs, then fully reverse-engineered using mechanistic interpretability techniques.

The central question: **what algorithm does the model actually learn?**

---

## Finding

The fully trained model implements a **partial two-step backward BFS from the destination node**, one BFS step per layer.

- **Layer 0** (head L0H1): a token-identity QK circuit routes the destination position's attention toward every edge triplet containing `dst`. The OV contribution writes neighborhood information into the residual stream, allowing the model to answer ~78% of distance-1 queries correctly after Layer 0 alone.
- **Layer 1**: reads that signal and checks whether `src` can reach any of `dst`'s direct predecessors in one hop. This handles distance-1 (99.4% val accuracy) and distance-2 (87.0%) well, and correctly identifies unreachability (INF, 96.2%). Distance-3 paths (23.8%) exceed the architecture's depth and are systematically degraded by Layer 1, which cannot perform a third BFS step.

Training shows two accuracy jumps — a double-grokking. These are mechanistically distinct:

- **Stage 1** (steps 114k–126k): Layer 1 discovers a statistical shortcut — attending to PAD-token positions injects a constant logit bias toward longer distances and INF. This is a matrix-alignment event, invisible in individual weight norms, and causally confirmed by ablation (−2.9% accuracy when PAD-position attention is zeroed at step 126k).
- **Stage 2** (steps 188k–196k): the shortcut is partially dismantled and replaced by the genuine BFS circuit above. L0H1's OV circuit norm grows +97% during this window.

The research paper writeup is in `notebooks/writeup.tex`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+. Training runs on CPU (Apple Silicon Metal backend disabled due to XLA compatibility). All interpretability scripts run on CPU.

---

## Repository structure

```
src/
  data/
    graphs.py         random directed graph generation, NetworkX BFS ground truth
    tokenizer.py      Vocab, encode/decode, sequence format
    dataset.py        generate_dataset, Dataset dataclass, train/val split
  model/
    transformer.py    ModelConfig, MultiHeadAttention, ShortestPathTransformer
  train/
    trainer.py        TrainConfig, train_step, eval_step, checkpointing, W&B
  interp/
    weight_analysis.py    Frobenius norm trajectories across all checkpoints
    attention_viz.py      attention pattern visualization, probe comparisons
    ov_circuit.py         OV and QK circuit matrices, PAD value analysis
    ablation.py           PAD-position ablation (numpy forward pass)
    logit_lens.py         per-layer accuracy by answer category

scripts/
  train.py              training entry point
  diagnose.py           pipeline diagnostic (6 isolated tests for XLA bugs)
  run_ablation.py       PAD ablation across 4 checkpoints
  run_qk_analysis.py    QK circuit analysis (L0H1 at 126k vs 196k)
  run_logit_lens.py     logit lens — per-category accuracy breakdown
  run_l0h1_ov.py        L0H1 full OV circuit analysis

notebooks/
  writeup.tex               research paper (Phase 4)
  documentation.tex         full technical documentation (31 findings, all derivations)
  weight_norms.png          norm trajectory plot (100 checkpoints, all heads)
  weight_norms_table.txt    anchor checkpoint norm table
  attention_patterns.png    6-probe × 3-checkpoint × 4-head attention grid
  attention_190k_zoom.png   attention patterns zoomed to step 190k
  ov_circuits.png           OV circuit heatmaps — Stage 2 window (126k/190k/196k)
  ov_circuits_stage1.png    OV circuit heatmaps — Stage 1 window (100k/116k/126k)
  ov_l0_circuits.png        L0 OV circuit heatmaps (126k vs 196k)
  qk_circuits.png           QK circuit heatmaps (126k vs 196k)

tests/
  test_data.py     14 data pipeline tests
  test_model.py    5 model tests

refs/
  Grokking.pdf
  A Mathematical Framework for Transformer Circuits.pdf
```

---

## Training

```bash
python scripts/train.py --n_nodes 4 --n_graphs 200 --weight_decay 0.4 --n_steps 200000
```

The analyzed run (`Run 5`) used `n_nodes=4`, `n_graphs=200`, `weight_decay=0.4`, `seed=0`, 200k steps. Checkpoints saved every 2,000 steps to `checkpoints/` (gitignored).

**Model:** 2-layer, 2-head attention-only transformer. `d_model=128`, `d_head=64`, ~140k parameters. No MLP layers — every computation is an attention head, which makes full circuit-level analysis tractable.

**Task:** given a shuffled sequence of directed edge tokens `[EDGE, u, v]` padded to fixed length, followed by `[QRY, src, dst]`, predict the shortest-path length from `src` to `dst` (tokens `1`, `2`, `3`, or `INF`).

---

## Interpretability pipeline

Run from the repository root after training.

```bash
# Causal validation: does PAD-position attention carry the shortcut?
python scripts/run_ablation.py

# What does L0H1 attend to? (QK circuit — token-identity routing)
python scripts/run_qk_analysis.py

# Which layer handles which distance category? (logit lens)
python scripts/run_logit_lens.py

# What does L0H1 write to the residual stream? (OV circuit)
python scripts/run_l0h1_ov.py
```

All four scripts load saved checkpoints from `checkpoints/` and write results to stdout. No GPU required.

The full technical documentation — all 31 numbered findings, circuit derivations, weight norm tables, and the complete algorithmic narrative — is in `notebooks/documentation.tex`.

---

## Key results

| Category | After Layer 0 | Full model | Layer 1 Δ |
|---|---|---|---|
| dist=1 (n=181) | 77.9% | 99.4% | +21.5 pp |
| dist=2 (n=92) | 17.4% | 87.0% | **+69.6 pp** |
| dist=3 (n=21) | 47.6% | 23.8% | **−23.8 pp** |
| INF (n=186) | 58.6% | 96.2% | +37.6 pp |
| Overall | 57.5% | 92.5% | +35.0 pp |

Step 196k, 480 validation examples. The −23.8 pp on dist=3 is an architectural constraint, not a training failure: a two-layer model cannot perform three BFS steps.

---

## References

- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177, 2022.
- Elhage et al., *A Mathematical Framework for Transformer Circuits*, Transformer Circuits Thread, 2021.
- Cohen et al., *Spectral Journey: How Transformers Predict the Shortest Path*, 2025.
- nostalgebraist, *Interpreting GPT: the logit lens*, LessWrong, 2020.
