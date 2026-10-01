import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from core.registry import TASK_SPECS
from core.utils import get_init_tokenizers, load_dataset


TYPE_KEYS = {
    "diag": {"cond": "cond_hist", "proc": "procedures", "drug": "drugs"},
    "drug": {"cond": "conditions", "proc": "procedures", "drug": "drugs_hist"},
}


def select_longest_patient_trajectories(samples):
    patient_to_sample = {}
    for sample in samples:
        patient_id = sample["patient_id"]
        visit_count = len(sample.get("visit_history", []))
        if patient_id not in patient_to_sample:
            patient_to_sample[patient_id] = sample
            continue
        prev_count = len(patient_to_sample[patient_id].get("visit_history", []))
        if visit_count > prev_count:
            patient_to_sample[patient_id] = sample
    return list(patient_to_sample.values())


def select_prior_cohort(patient_samples, cohort, train_ratio, seed):
    if cohort == "all":
        return patient_samples
    if cohort != "train":
        raise ValueError(f"Unknown cohort: {cohort}")
    patient_samples = list(patient_samples)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(patient_samples))
    train_size = int(len(patient_samples) * train_ratio)
    if patient_samples:
        train_size = max(1, train_size)
    return [patient_samples[int(i)] for i in indices[:train_size]]


def encode_trajectory(sample, tokenizers, key_map):
    visits = {}
    for type_name, feature_key in key_map.items():
        visit_tokens = sample.get(feature_key, [])
        visits[type_name] = tokenizers[feature_key].batch_encode_2d(
            visit_tokens,
            padding=False,
            truncation=False,
        )
    return visits


def compute_structural_prior(patient_sets, vocab_size, topk, eps):
    cooc = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    freq = np.zeros(vocab_size, dtype=np.float32)

    for token_set in tqdm(patient_sets, desc="cooccurrence", leave=False):
        tokens = sorted(token_set)
        for i in tokens:
            freq[i] += 1.0
        for i in tokens:
            for j in tokens:
                if i != j:
                    cooc[i, j] += 1.0

    cooc_sum = cooc.sum(axis=1)
    structural = np.divide(
        cooc_sum,
        cooc_sum.max() + eps,
        out=np.zeros_like(cooc_sum),
        where=cooc_sum.max() > 0,
    )

    return {
        "cooccurrence_count": torch.tensor(cooc, dtype=torch.float32),
        "cooccurrence_topk": dense_to_row_topk_sparse(cooc, topk=topk),
        "patient_count": torch.tensor(freq, dtype=torch.float32),
        "structural_prior": torch.tensor(structural, dtype=torch.float32),
        "centrality": torch.tensor(structural, dtype=torch.float32),
    }


def compute_directional_prior(trajectories, patient_counts, num_patients, vocab_size, topk, eps):
    before_count = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for visits in tqdm(trajectories, desc="directional", leave=False):
        first_visit = {}
        last_visit = {}
        for visit_idx, visit in enumerate(visits):
            for token in set(visit):
                first_visit.setdefault(token, visit_idx)
                last_visit[token] = visit_idx

        tokens = sorted(first_visit)
        for i in tokens:
            for j in tokens:
                if i == j:
                    continue
                if first_visit[i] < last_visit[j]:
                    before_count[i, j] += 1.0

    denom = patient_counts[:, None] * patient_counts[None, :] + eps
    directional = np.log1p(before_count * float(num_patients) / denom)
    directional[before_count <= 0] = 0.0

    return {
        "directional_topk": dense_to_row_topk_sparse(directional, topk=topk),
        "directional_score": torch.tensor(directional, dtype=torch.float32),
        "directional_count": torch.tensor(before_count, dtype=torch.float32),
    }


def dense_to_row_topk_sparse(matrix, topk):
    rows = []
    cols = []
    values = []
    for row in range(matrix.shape[0]):
        nonzero = np.flatnonzero(matrix[row] > 0)
        if nonzero.size == 0:
            continue
        if topk > 0 and nonzero.size > topk:
            local = nonzero[np.argsort(matrix[row, nonzero])[-topk:]]
        else:
            local = nonzero
        for col in local:
            rows.append(row)
            cols.append(int(col))
            values.append(float(matrix[row, col]))

    if values:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        vals = torch.tensor(values, dtype=torch.float32)
    else:
        indices = torch.empty((2, 0), dtype=torch.long)
        vals = torch.empty((0,), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, vals, size=matrix.shape).coalesce()


def tokenizer_vocab(tokenizer):
    vocab = tokenizer.vocabulary
    if isinstance(vocab, dict):
        token_to_index = vocab
    elif hasattr(vocab, "token2idx"):
        token_to_index = vocab.token2idx
    elif hasattr(vocab, "_token2idx"):
        token_to_index = vocab._token2idx
    else:
        size = tokenizer.get_vocabulary_size()
        return [tokenizer.convert_indices_to_tokens([i])[0] for i in range(size)]

    index_to_token = [None] * tokenizer.get_vocabulary_size()
    for token, idx in token_to_index.items():
        if 0 <= idx < len(index_to_token):
            index_to_token[idx] = token
    return index_to_token


def build_priors(args):
    if args.task not in TASK_SPECS:
        raise ValueError(f"Unknown task: {args.task}")
    spec = TASK_SPECS[args.task]
    if args.dataset not in spec["task_fn"]:
        raise ValueError(f"Task {args.task} does not support dataset {args.dataset}")

    task_dataset = load_dataset(
        args.dataset,
        root=args.data_root,
        task_fn=spec["task_fn"][args.dataset],
        dev=args.devm,
    )
    tokenizers = get_init_tokenizers(task_dataset, keys=spec["feature_keys"])
    all_patient_samples = select_longest_patient_trajectories(task_dataset.samples)
    patient_samples = select_prior_cohort(
        all_patient_samples,
        cohort=args.cohort,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    key_map = TYPE_KEYS[args.task]

    patient_sets = {
        "cond": [],
        "proc": [],
        "drug": [],
    }
    directional_trajectories = {
        "cond": [],
        "proc": [],
        "drug": [],
    }

    for sample in tqdm(patient_samples, desc="encoding trajectories"):
        visits_by_type = encode_trajectory(sample, tokenizers, key_map)
        for type_name, visits in visits_by_type.items():
            token_set = set()
            for visit in visits:
                token_set.update(visit)
            patient_sets[type_name].append(token_set)
            directional_trajectories[type_name].append(visits)

    out = {
        "metadata": {
            "task": args.task,
            "dataset": args.dataset,
            "cohort": args.cohort,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "num_all_patients": len(all_patient_samples),
            "num_prior_patients": len(patient_samples),
            "topk": args.topk,
            "formula": {
                "structural_prior": "gamma_i=sum_j C_ij / (max_k sum_j C_kj + eps)",
                "directional_score": "S_ij=log(1 + n(i->j) * N / (n_i * n_j + eps))",
            },
        }
    }

    for type_name, feature_key in key_map.items():
        vocab_size = tokenizers[feature_key].get_vocabulary_size()
        print(f"[prior] {type_name}: vocab={vocab_size}")
        structural = compute_structural_prior(
            patient_sets[type_name],
            vocab_size=vocab_size,
            topk=args.topk,
            eps=args.eps,
        )
        directional = compute_directional_prior(
            directional_trajectories[type_name],
            patient_counts=structural["patient_count"].numpy(),
            num_patients=len(patient_samples),
            vocab_size=vocab_size,
            topk=args.topk,
            eps=args.eps,
        )
        out[type_name] = {
            "feature_key": feature_key,
            "index_to_token": tokenizer_vocab(tokenizers[feature_key]),
            **structural,
            **directional,
        }

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build population-level cooccurrence and progression priors."
    )
    parser.add_argument("--task", required=True, choices=["diag", "drug"])
    parser.add_argument("--dataset", required=True, choices=["mimic3", "mimic4"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_path", default="./results/population_priors.pt")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--cohort", choices=["train", "all"], default="train")
    parser.add_argument("--train_ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--devm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    priors = build_priors(args)
    torch.save(priors, args.output_path)
    print(f"[done] saved population priors to {args.output_path}")


if __name__ == "__main__":
    main()
