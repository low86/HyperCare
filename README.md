# HyperCare: Longitudinal Heterogeneous Hypergraph Neural Network for Healthcare Prediction

![Pipeline](source/pipeline.jpg)

## Overview
HyperCare is a longitudinal heterogeneous hypergraph framework for healthcare prediction. It builds patient-specific hypergraphs from visit sequences and performs message passing over events and time hyperedges.

## Repository Structure
- `benchmarks/` Baseline models and task wrappers.
- `core/` Training loop, task specs, and utilities.
- `models/` HyperCare model, encoder, and offline graph construction.
- `preprocess/` Data preprocessing utilities (note embeddings, code embeddings, co-occurrence stats).
- `graph/` Cached hypergraph data (generated at runtime).
- `logs/` Training outputs and cached artifacts.
- `source/` Figures and assets (pipeline image).

## Supported Tasks and Models

| Task | Datasets | Models |
|---|---|---|
| Diagnosis prediction | `mimic3`, `mimic4` | `HyperCare`, `Transformer`, `RETAIN`, `StageNet`, `KAME`, `TRANS` |
| Drug recommendation | `mimic3`, `mimic4` | `HyperCare`, `Transformer`, `RETAIN`, `StageNet`, `Deepr`, `AdaCare`, `ConCare`, `GRASP`, `DrAgent`, `SafeDrug`, `GAMENet`, `MICRON`, `MoleRec`, `TRANS` |

## Quick Start

### 1) Diagnosis prediction
```bash
python main.py \
  --task diag \
  --dataset mimic3 \
  --model HyperCare \
  --data_root <PATH_TO_MIMIC3> \
  --note_emb_path <PATH_TO_NOTE_EMB_NPY> \
  --node_emb_path <PATH_TO_NODE_EMB_NPY>
```

### 2) Drug recommendation
```bash
python main.py \
  --task drug \
  --dataset mimic3 \
  --model HyperCare \
  --data_root <PATH_TO_MIMIC3> \
  --note_emb_path <PATH_TO_NOTE_EMB_NPY> \
  --node_emb_path <PATH_TO_NODE_EMB_NPY>
```

### 3) Run a baseline (example)
```bash
python main.py \
  --task drug \
  --dataset mimic3 \
  --model Transformer \
  --data_root <PATH_TO_MIMIC3>
```

## Notes
- `note_emb_path` is required for HyperCare. It should point to the offline note embedding file (e.g., `note_emb_dict.npy`).
- `node_emb_path` is optional. If provided, code embeddings are initialized from semantic vectors (e.g., 768-dim). If not provided, embeddings are randomly initialized.
- Hypergraph caches are stored under `logs/` and `graph/` by default. If you change `time_dim`, a new cache file will be generated.

## Requirements
- PyTorch 2.1.2
- PyTorch Geometric 2.3.1
- torch-scatter 2.1.2
- torch-sparse 0.6.18
- PyHealth 1.1.6

## Clinical Summaries
> **300,000 high-quality clinical summaries**  
> Download and node descriptions: **[Link to be added]**

## Reproducibility
- Set `--seed` for deterministic training.
- Use the same dataset split and cache files for fair comparison.
