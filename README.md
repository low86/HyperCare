# HyperCare: Longitudinal Heterogeneous Hypergraph Neural Network for Healthcare Prediction

[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-blue)](#)

This repository contains the official implementation of **HyperCare**, accepted by **ACM Multimedia 2026 (ACM MM 2026)**.

![Pipeline](source/pipeline.jpg)

## Overview
HyperCare is a longitudinal heterogeneous hypergraph neural network for modeling patient health trajectories from multimodal EHRs. It integrates guided hypergraph message passing to capture subtle risk signals and clinically meaningful longitudinal health dynamics.

## Repository Structure
- `benchmarks/` Baseline models and task wrappers.
- `core/` Training loop, task specs, and utilities.
- `models/` HyperCare model, encoder, and offline graph construction.
- `logs/` Training outputs and cached artifacts.

## Supported Tasks and Models

| Task | Datasets | Models |
|---|---|---|
| Diagnosis prediction | `mimic3`, `mimic4` | `HyperCare`, `Transformer`, `RETAIN`, `StageNet`, `KAME`, `TRANS` |
| Drug recommendation | `mimic3`, `mimic4` | `HyperCare`, `Transformer`, `RETAIN`, `StageNet`, `Deepr`, `AdaCare`, `ConCare`, `GRASP`, `DrAgent`, `SafeDrug`, `GAMENet`, `MICRON`, `MoleRec`, `TRANS` |

## Datasets
HyperCare is evaluated on MIMIC-III and MIMIC-IV. Structured EHR records are loaded with PyHealth from the corresponding hospital module directories.

Clinical notes are used only to generate local summary embeddings:
- MIMIC-IV notes: `discharge.csv` from MIMIC-IV-Note.
- MIMIC-III notes: `NOTEEVENTS.csv` from MIMIC-III.

The repository does not redistribute MIMIC notes, generated clinical summaries, or note embeddings. Users must obtain authorized access to the MIMIC datasets through PhysioNet and generate these artifacts locally.

## Quick Start

Create the environment first:

```bash
conda env create -f environment.yml
conda activate hypercare
```

This environment was prepared for experiments on an NVIDIA RTX 4090 with CUDA 11.8 and PyTorch 2.1.2.

### 1) Diagnosis prediction
```bash
python main.py \
  --task diag \
  --dataset mimic3 \
  --model HyperCare \
  --data_root <PATH_TO_MIMIC3_HOSP> \
  --note_emb_path <PATH_TO_NOTE_EMB_NPY> \
  --node_emb_path <PATH_TO_NODE_EMB_NPY>
```

### 2) Drug recommendation
```bash
python main.py \
  --task drug \
  --dataset mimic3 \
  --model HyperCare \
  --data_root <PATH_TO_MIMIC3_HOSP> \
  --note_emb_path <PATH_TO_NOTE_EMB_NPY> \
  --node_emb_path <PATH_TO_NODE_EMB_NPY>
```

### 3) Run a baseline (example)
```bash
python main.py \
  --task drug \
  --dataset mimic3 \
  --model Transformer \
  --data_root <PATH_TO_MIMIC3_HOSP>
```

## Notes
- `data_root` is required. Use the path to the corresponding MIMIC-III or MIMIC-IV hospital module directory.
- `note_emb_path` is required for HyperCare. It should point to the offline note embedding file (e.g., `note_emb_dict.npy`).
- `node_emb_path` is optional. If provided, code embeddings are initialized from semantic vectors (e.g., 768-dim). If not provided, embeddings are randomly initialized.

## Clinical Note Summaries
The note pipeline is optional and produces the `note_emb_path` artifact used by HyperCare. It has two steps: summarize selected discharge-note sections with Qwen2.5, then encode each summary with a clinical/text encoder.

Install optional note-processing dependencies:

```bash
pip install -r requirements-notes.txt
```

Run a Qwen2.5 OpenAI-compatible server separately, for example on `http://127.0.0.1:8000/v1`. One common setup is:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000
```

Then generate summaries:

```bash
python preprocess/notes/build_note_summaries.py \
  --dataset mimic4 \
  --note_csv <PATH_TO_MIMIC_IV_NOTE>/discharge.csv \
  --output_jsonl ./logs/mimic4_note_summaries.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --resume
```

For MIMIC-III, use `--dataset mimic3` and `--note_csv <PATH_TO_MIMIC_III>/NOTEEVENTS.csv`.

The summary script extracts these sections when available:
- Chief Complaint
- Past Medical History
- Discharge Condition
- Discharge Diagnosis
- Brief Hospital Course
- Discharge Medications

Encode summaries into a `{hadm_id: embedding}` `.npy` dictionary:

```bash
python preprocess/notes/embed_note_summaries.py \
  --summary_jsonl ./logs/mimic4_note_summaries.jsonl \
  --output_npy ./logs/mimic4_emb_dict.npy \
  --encoder bioclinicalbert \
  --resume
```

Supported encoder presets are `clinical-longformer`, `bioclinicalbert`, `bge-m3`, and `jina-v3`. You can also pass a local Hugging Face model path or model ID through `--encoder`. The script uses the `[CLS]` representation directly.

## Population-Level Priors
Population-level priors can be generated from the structured EHR trajectories without using clinical notes. The script builds two kinds of priors for each clinical type (`cond`, `proc`, `drug`):
- Co-occurrence priors: patient-level co-occurrence matrix, Jaccard top-k graph, frequency, centrality, and specificity.
- Temporal progression priors: visit-ordered directional transition scores with temporal decay.

Example:

```bash
python preprocess/build_population_priors.py \
  --task diag \
  --dataset mimic3 \
  --data_root <PATH_TO_MIMIC3_HOSP> \
  --output_path ./logs/diag_mimic3_population_priors.pt
```

The saved artifact is a PyTorch dictionary with `metadata` and one entry per clinical type. Dense scores and sparse top-k tensors are both included so downstream guidance modules can choose the most convenient representation.

## Environment
The recommended environment is provided in `environment.yml`.

Key versions:
- Python 3.10
- CUDA 11.8
- PyTorch 2.1.2
- PyTorch Geometric 2.3.1
- torch-scatter 2.1.2
- torch-sparse 0.6.18
- PyHealth 1.1.6

## Reproducibility
- Set `--seed` for deterministic training.
- Use the same dataset split and cache files for fair comparison.
