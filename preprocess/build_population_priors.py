import argparse
import math
import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from core.registry import TASK_SPECS
from core.utils import get_init_tokenizers, load_dataset


TYPE_KEYS = {
    "diag": {"cond": "cond_hist", "proc": "procedures", "drug": "drugs"},
    "drug": {"cond": "conditions", "proc": "procedures", "drug": "drugs_hist"},
}


def parse_time(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


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


def encode_trajectory(sample, tokenizers, key_map):
    visits = {}
    for type_name, feature_key in key_map.items():
        visit_tokens = sample.get(feature_key, [])
        visits[type_name] = tokenizers[feature_key].batch_encode_2d(
            visit_tokens,
            padding=False,
            truncation=False,
        )
    times = sample.get("adm_time", [])
    parsed_times = [parse_time(t) for t in times]
    return visits, parsed_times


def compute_cooccurrence_prior(patient_sets, vocab_size, topk, eps):
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

    denom = freq[:, None] + freq[None, :] - cooc + eps
    jaccard = np.divide(cooc, denom, out=np.zeros_like(cooc), where=denom > 0)
    centrality = cooc.sum(axis=1)
    if centrality.max() > 0:
        centrality = centrality / centrality.max()

    degree = (jaccard > 0).sum(axis=1).astype(np.float32)
    specificity = 1.0 / np.sqrt(degree + 1.0)
    if specificity.max() > specificity.min():
        specificity = (specificity - specificity.min()) / (
            specificity.max() - specificity.min()
        )

    return {
        "cooccurrence": torch.tensor(cooc, dtype=torch.float32),
        "jaccard_topk": dense_to_row_topk_sparse(jaccard, topk=topk),
        "frequency": torch.tensor(freq, dtype=torch.float32),
        "centrality": torch.tensor(centrality, dtype=torch.float32),
        "specificity": torch.tensor(specificity, dtype=torch.float32),
    }


def compute_directional_prior(trajectories, vocab_size, topk, tau_days, tau_visits, eps):
    score = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    count = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for visits, times in tqdm(trajectories, desc="directional", leave=False):
        n_visits = len(visits)
        for t in range(n_visits):
            src_tokens = set(visits[t])
            if not src_tokens:
                continue
            for u in range(t + 1, n_visits):
                dst_tokens = set(visits[u])
                if not dst_tokens:
                    continue
                weight = temporal_weight(times, t, u, tau_days, tau_visits)
                for i in src_tokens:
                    for j in dst_tokens:
                        if i == j:
                            continue
                        score[i, j] += weight
                        count[i, j] += 1.0

    row_sum = score.sum(axis=1, keepdims=True)
    directional = np.divide(
        score,
        row_sum + eps,
        out=np.zeros_like(score),
        where=row_sum > 0,
    )

    return {
        "directional_topk": dense_to_row_topk_sparse(directional, topk=topk),
        "directional_score": torch.tensor(directional, dtype=torch.float32),
        "transition_count": torch.tensor(count, dtype=torch.float32),
    }


def temporal_weight(times, src_idx, dst_idx, tau_days, tau_visits):
    src_time = times[src_idx] if src_idx < len(times) else None
    dst_time = times[dst_idx] if dst_idx < len(times) else None
    if src_time is not None and dst_time is not None:
        delta_days = max((dst_time - src_time).days, 0)
        return math.exp(-float(delta_days) / max(tau_days, 1e-6))
    gap = max(dst_idx - src_idx, 1)
    return math.exp(-float(gap) / max(tau_visits, 1e-6))


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
        use_icd_raw=False,
    )
    tokenizers = get_init_tokenizers(task_dataset, keys=spec["feature_keys"])
    patient_samples = select_longest_patient_trajectories(task_dataset.samples)
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
        visits_by_type, times = encode_trajectory(sample, tokenizers, key_map)
        for type_name, visits in visits_by_type.items():
            token_set = set()
            for visit in visits:
                token_set.update(visit)
            patient_sets[type_name].append(token_set)
            directional_trajectories[type_name].append((visits, times))

    out = {
        "metadata": {
            "task": args.task,
            "dataset": args.dataset,
            "num_patients": len(patient_samples),
            "topk": args.topk,
            "tau_days": args.tau_days,
            "tau_visits": args.tau_visits,
        }
    }

    for type_name, feature_key in key_map.items():
        vocab_size = tokenizers[feature_key].get_vocabulary_size()
        print(f"[prior] {type_name}: vocab={vocab_size}")
        cooc = compute_cooccurrence_prior(
            patient_sets[type_name],
            vocab_size=vocab_size,
            topk=args.topk,
            eps=args.eps,
        )
        directional = compute_directional_prior(
            directional_trajectories[type_name],
            vocab_size=vocab_size,
            topk=args.topk,
            tau_days=args.tau_days,
            tau_visits=args.tau_visits,
            eps=args.eps,
        )
        out[type_name] = {
            "feature_key": feature_key,
            "index_to_token": tokenizer_vocab(tokenizers[feature_key]),
            **cooc,
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
    parser.add_argument("--output_path", default="./logs/population_priors.pt")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--tau_days", type=float, default=365.0)
    parser.add_argument("--tau_visits", type=float, default=4.0)
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
