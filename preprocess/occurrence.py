
import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_cooccurrence_centrality(cond_sets, proc_sets, drug_sets, topk_show=10, sample_matrix=50, tau=0.005):
    print("=== 1) Range check ===")
    def check_range(c_tensor, name):
        print(f"{name}: min={c_tensor.min():.4f}, max={c_tensor.max():.4f}")

    # vocab size
    def safe_max(s):
        s_non_empty = [x for x in s if len(x) > 0]
        if len(s_non_empty) == 0:
            return 0
        return max(max(x) for x in s_non_empty)

    cond_vocab = safe_max(cond_sets) + 1
    proc_vocab = safe_max(proc_sets) + 1
    drug_vocab = safe_max(drug_sets) + 1

    cond_c = _compute_single_type_centrality(cond_sets, cond_vocab, tau=tau)
    proc_c = _compute_single_type_centrality(proc_sets, proc_vocab, tau=tau)
    drug_c = _compute_single_type_centrality(drug_sets, drug_vocab, tau=tau)

    check_range(cond_c, "cond_centrality")
    check_range(proc_c, "proc_centrality")
    check_range(drug_c, "drug_centrality")

    print("\n=== 2) Top-k tokens ===")
    def topk_tokens(c_tensor, name):
        top_idx = torch.topk(c_tensor, k=min(topk_show, len(c_tensor)))[1].tolist()
        print(f"Top {topk_show} {name} tokens: {top_idx}")

    topk_tokens(cond_c, "cond")
    topk_tokens(proc_c, "proc")
    topk_tokens(drug_c, "drug")

    print("\n=== 3) Co-occurrence matrix visualization (sample) ===")
    def plot_cooc_matrix(sets, name, sample_size=sample_matrix):
        non_empty_sets = [s for s in sets if len(s) > 0]
        if len(non_empty_sets) == 0:
            print(f"{name}: no data to visualize")
            return
        vocab_size = max(max(s) for s in non_empty_sets) + 1
        C = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        for tokens in non_empty_sets:
            for i in tokens:
                for j in tokens:
                    if i != j:
                        C[i,j] += 1
        sample_size = min(sample_size, vocab_size)
        plt.figure(figsize=(6,6))
        plt.imshow(C[:sample_size, :sample_size], cmap="hot", interpolation="nearest")
        plt.title(f"{name} co-occurrence (top {sample_size})")
        plt.colorbar()
        plt.show()

    plot_cooc_matrix(cond_sets, "cond")
    plot_cooc_matrix(proc_sets, "proc")
    plot_cooc_matrix(drug_sets, "drug")

    print("\n=== 4) Sparsification check ===")
    def sparse_check(patient_sets, token_type="", tau=tau):
        non_empty_sets = [s for s in patient_sets if len(s) > 0]
        if len(non_empty_sets) == 0:
            print(f"{token_type}: No tokens, skipping sparse check.")
            return
        vocab_size = max(max(s) for s in non_empty_sets) + 1
        C = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        for tokens in non_empty_sets:
            for i in tokens:
                for j in tokens:
                    if i != j:
                        C[i,j] += 1
        row_sum = C.sum(axis=1)
        row_sum[row_sum==0] = 1.0
        P = C / row_sum[:, None]
        P[P<tau] = 0.0
        non_zero = np.count_nonzero(P)
        print(f"{token_type}: Non-zero entries after tau={tau}: {non_zero} / {P.size} ({non_zero/P.size:.4%})")

    sparse_check(cond_sets, "cond")
    sparse_check(proc_sets, "proc")
    sparse_check(drug_sets, "drug")
    print("\nCo-occurrence centrality check complete.")

# def aggregate_patient_tokens_from_graphs(graphs):
#     """
#     Aggregate tokens per patient from graph_data.
#     """
#     from collections import defaultdict
#     
#     patient_tokens = defaultdict(lambda: {"cond": set(), "proc": set(), "drug": set()})
#     
#     for g in tqdm(graphs, desc="Aggregate patient tokens"):
#         patient_id = g["patient_id"]  # group by patient
#         
#         # Aggregate tokens across all visits for this sample
#         for visit in g.get("cond_tokens", []):
#             patient_tokens[patient_id]["cond"].update(visit)
#         for visit in g.get("proc_tokens", []):
#             patient_tokens[patient_id]["proc"].update(visit)
#         for visit in g.get("drug_tokens", []):
#             patient_tokens[patient_id]["drug"].update(visit)
#     
#     # Convert to lists (one set per patient)
#     cond_sets = [tokens["cond"] for tokens in patient_tokens.values()]
#     proc_sets = [tokens["proc"] for tokens in patient_tokens.values()]
#     drug_sets = [tokens["drug"] for tokens in patient_tokens.values()]
#     
#     print(f"Total unique patients: {len(patient_tokens)}")
#     return cond_sets, proc_sets, drug_sets


# Absolute co-occurrence code

def aggregate_patient_tokens_from_graphs(graphs):
    """
    graphs: list of dicts, each dict is an element of MMDataset.graph_data
    Returns token sets per type (one set per patient)
    """
    cond_sets = []
    proc_sets = []
    drug_sets = []

    for g in tqdm(graphs, desc="Aggregate patient tokens"):
        if not isinstance(g, dict):
            raise TypeError(f"Expected dict, got {type(g)}")
        
        cond_set = set()
        proc_set = set()
        drug_set = set()

        # Token list for each visit
        for visit in g.get("cond_tokens", []):
            cond_set.update(visit)
        for visit in g.get("proc_tokens", []):
            proc_set.update(visit)
        for visit in g.get("drug_tokens", []):
            drug_set.update(visit)

        cond_sets.append(cond_set)
        proc_sets.append(proc_set)
        drug_sets.append(drug_set)

    return cond_sets, proc_sets, drug_sets



# def _compute_single_type_centrality(patient_token_sets, vocab_size, tau=None, topk=None):
#     """
#     Simplified version: only absolute co-occurrence counts, chronic codes will be high.
#     """
#     C = np.zeros((vocab_size, vocab_size), dtype=np.float32)

#     for tokens in tqdm(patient_token_sets, desc="Counting co-occurrence"):
#         uniq = list(tokens)
#         for i in uniq:
#             for j in uniq:
#                 if i != j:
#                     C[i, j] += 1

#     # Direct sum (no sparsification)
#     c = C.sum(axis=1)
#     
#     if c.max() > 0:
#         c = c / c.max()

#     return torch.tensor(c, dtype=torch.float32)

# def compute_cooccurrence_centrality(mdataset, tokenizers, dataset_name, tau=0.005, topk=24, save_dir="./logs"):
#     """
#     mdataset: MMDataset instance
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Use MMDataset.graph_data
#     if hasattr(mdataset, "graph_data"):
#         graphs = mdataset.graph_data
#     else:
#         raise AttributeError("mdataset does not have graph_data; pass a MMDataset instance")

#     cond_sets, proc_sets, drug_sets = aggregate_patient_tokens_from_graphs(graphs)

#     # vocab size from tokenizer or semantic table
#     cond_vocab = len(tokenizers['cond_hist'].vocabulary)
#     proc_vocab = len(tokenizers['procedures'].vocabulary)
#     drug_vocab = len(tokenizers['drugs'].vocabulary)

#     cond_c = _compute_single_type_centrality(cond_sets, cond_vocab, tau, topk)
#     proc_c = _compute_single_type_centrality(proc_sets, proc_vocab, tau, topk)
#     drug_c = _compute_single_type_centrality(drug_sets, drug_vocab, tau, topk)

#     save_path = os.path.join(save_dir, f"cooc_centrality_{dataset_name}.pt")
#     torch.save({"cond": cond_c, "proc": proc_c, "drug": drug_c}, save_path)
#     print(f"Saved co-occurrence centrality to {save_path}")

#     # check_cooccurrence_centrality(cond_sets, proc_sets, drug_sets)
#     return cond_c, proc_c, drug_c

# # Functions to compute co-occurrence matrix and metrics (legacy, commented out)

def compute_cooccurrence_matrix(patient_token_sets, vocab_size):
    """
    Returns:
    C: [V, V] co-occurrence matrix
    freq: [V] frequency
    """
    C = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    freq = np.zeros(vocab_size, dtype=np.float32)

    for tokens in tqdm(patient_token_sets, desc="Counting co-occurrence"):
        uniq = list(tokens)
        for i in uniq:
            freq[i] += 1
        for i in uniq:
            for j in uniq:
                if i != j:
                    C[i, j] += 1
    return C, freq

def compute_jaccard_cooccurrence(C, freq, eps=1e-8):
    """
    Returns P: [V, V], range [0,1]
    """
    freq_i = freq[:, None]
    freq_j = freq[None, :]
    denom = freq_i + freq_j - C + eps
    P = C / denom
    P[np.isnan(P)] = 0.0
    return P

def compute_complexity(P, eps=0.01):
    """
    complexity_i = log(1 + number of significant co-occurrences)
    """
    # count how many j are structurally related to i
    breadth = (P > eps).sum(axis=1).astype(np.float32)
    complexity = np.log1p(breadth)

    # normalize to [0,1]
    if complexity.max() > 0:
        complexity = complexity / complexity.max()
    return complexity

def compute_absolute_centrality(C):
    c = C.sum(axis=1)
    if c.max() > 0:
        c = c / c.max()
    return c

def compute_type_cooccurrence_prior(
    patient_token_sets,
    vocab_size,
    save_path=None
):
    C, freq = compute_cooccurrence_matrix(patient_token_sets, vocab_size)
    P = compute_jaccard_cooccurrence(C, freq)
    complexity = compute_complexity(P)
    centrality = compute_absolute_centrality(C)

    out = {
        "C": torch.tensor(C, dtype=torch.float32),
        "P": torch.tensor(P, dtype=torch.float32),
        "complexity": torch.tensor(complexity, dtype=torch.float32),
        "centrality": torch.tensor(centrality, dtype=torch.float32),
    }

    if save_path is not None:
        torch.save(out, save_path)
        print(f"Saved co-occurrence prior to {save_path}")

    return out

def compute_cooccurrence_centrality(mdataset, tokenizers, dataset_name, tau=0.005, topk=24, save_dir="./logs"):
    """
    mdataset: MMDataset instance
    """
    os.makedirs(save_dir, exist_ok=True)

    # Use MMDataset.graph_data
    if hasattr(mdataset, "graph_data"):
        graphs = mdataset.graph_data
    else:
        raise AttributeError("mdataset does not have graph_data; pass a MMDataset instance")

    cond_sets, proc_sets, drug_sets = aggregate_patient_tokens_from_graphs(graphs)

    # vocab size from tokenizer or semantic table
    cond_vocab = len(tokenizers['cond_hist'].vocabulary)
    proc_vocab = len(tokenizers['procedures'].vocabulary)
    drug_vocab = len(tokenizers['drugs'].vocabulary)

    cond_c = compute_type_cooccurrence_prior(cond_sets, cond_vocab)
    proc_c = compute_type_cooccurrence_prior(proc_sets, proc_vocab)
    drug_c = compute_type_cooccurrence_prior(drug_sets, drug_vocab)

    save_path = os.path.join(save_dir, f"cooc_centrality_{dataset_name}.pt")
    torch.save({"cond": cond_c, "proc": proc_c, "drug": drug_c}, save_path)
    print(f"Saved co-occurrence centrality to {save_path}")

    # check_cooccurrence_centrality(cond_sets, proc_sets, drug_sets)
    return cond_c, proc_c, drug_c


# def compute_cooccurrence_matrix(patient_token_sets, vocab_size): 
#     """
#     Returns:
#     C: [V, V] co-occurrence matrix
#     freq: [V] frequency
#     """
#     C = np.zeros((vocab_size, vocab_size), dtype=np.float32)
#     freq = np.zeros(vocab_size, dtype=np.float32)
#     for tokens in tqdm(patient_token_sets, desc="Counting co-occurrence"):
#         uniq = list(tokens)
#         for i in uniq:
#             freq[i] += 1
#         for i in uniq:
#             for j in uniq:
#                 if i != j:
#                     C[i, j] += 1
#     return C, freq


# def sparsify_topk(P, k=20):
#     """
#     Change 1: keep only top-k co-occurrences to increase contrast and sparsity.
#     """
#     P_sparse = np.zeros_like(P)
#     for i in range(P.shape[0]):
#         if np.count_nonzero(P[i]) > 0:
#             # Keep top-k per row
#             idx = np.argsort(P[i])[-k:]
#             P_sparse[i, idx] = P[i, idx]
#     return P_sparse


# def compute_jaccard_cooccurrence(C, freq, eps=1e-8, topk=20):
#     """
#     Returns P: [V, V], range [0,1]
#     Change: add top-k sparsification
#     """
#     freq_i = freq[:, None]
#     freq_j = freq[None, :]
#     denom = freq_i + freq_j - C + eps
#     P = C / denom
#     P[np.isnan(P)] = 0.0
#     
#     # Key: sparsify, keep only top-k
#     P = sparsify_topk(P, k=topk)
#     
#     return P


# def compute_complexity(P, eps=0.01, mode='inverse'):
#     """
#     Change 2: complexity rewards "specific" rather than "common".
#     
#     Args:
#         mode: 
#             - 'inverse': inverse degree (recommended)
#             - 'idf': IDF-like
#             - 'log': original log (not recommended)
#     """
#     breadth = (P > eps).sum(axis=1).astype(np.float32)
#     
#     if mode == 'inverse':
#         # Recommended: fewer strong relations => higher complexity
#         complexity = 1.0 / np.sqrt(breadth + 1.0)
#     elif mode == 'idf':
#         # IDF-like: log(total / breadth)
#         total = P.shape[0]
#         complexity = np.log(total / (breadth + 1.0))
#     else:  # 'log' (not recommended)
#         complexity = np.log1p(breadth)
#     
#     # normalize to [0,1]
#     if complexity.max() > complexity.min():
#         complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min())
#     
#     return complexity


# def compute_absolute_centrality(C, zero_mean=True):
#     """
#     Change 3: zero-mean centrality to avoid all-positive bias.
#     """
#     c = C.sum(axis=1)
#     if c.max() > 0:
#         c = c / c.max()
#     
#     if zero_mean:
#         # zero-mean normalization
#         c_ = c - c.mean()
#     
#     return c, c_


# def compute_type_cooccurrence_prior(
#     patient_token_sets,
#     vocab_size,
#     save_path=None,
#     topk=20,              # new parameter
#     complexity_mode='inverse',  # new parameter
#     zero_mean_centrality=True   # new parameter
# ):
#     """
#     Full revised version
#     """
#     C, freq = compute_cooccurrence_matrix(patient_token_sets, vocab_size)
#     
#     # Change 1: sparsified Jaccard
#     P = compute_jaccard_cooccurrence(C, freq, topk=topk)
#     
#     # Change 2: inverse-degree complexity
#     complexity = compute_complexity(P, mode=complexity_mode)
#     
#     # Change 3: zero-mean centrality
#     centrality, centrality_zero_mean = compute_absolute_centrality(C, zero_mean=zero_mean_centrality)
#     
#     out = {
#         "C": torch.tensor(C, dtype=torch.float32),
#         "P": torch.tensor(P, dtype=torch.float32),
#         "complexity": torch.tensor(complexity, dtype=torch.float32),
#         "centrality": torch.tensor(centrality, dtype=torch.float32),
#         "centrality_zero_mean": torch.tensor(centrality_zero_mean, dtype=torch.float32),
#     }
#     
#     if save_path is not None:
#         torch.save(out, save_path)
#         print(f"Saved co-occurrence prior to {save_path}")
#     
#     # New: print diagnostic info
#     print_prior_stats(out, vocab_size)
#     
#     return out


# def print_prior_stats(prior_dict, vocab_size):
#     """
#     Print summary stats for verification
#     """
#     P = prior_dict['P'].numpy()
#     complexity = prior_dict['complexity'].numpy()
#     centrality = prior_dict['centrality'].numpy()
#     
#     print("\n=== Prior Statistics ===")
#     print(f"Vocab size: {vocab_size}")
#     
#     # P sparsity
#     P_nonzero = (P > 0).sum()
#     P_density = P_nonzero / (vocab_size * vocab_size)
#     print(f"P density: {P_density:.4f} ({P_nonzero} / {vocab_size**2})")
#     print(f"P avg nonzero per row: {(P > 0).sum(axis=1).mean():.2f}")
#     
#     # Complexity distribution
#     print(f"Complexity: min={complexity.min():.4f}, max={complexity.max():.4f}, "
#           f"mean={complexity.mean():.4f}, std={complexity.std():.4f}")
#     
#     # Centrality distribution (should be close to zero-mean)
#     print(f"Centrality: min={centrality.min():.4f}, max={centrality.max():.4f}, "
#           f"mean={centrality.mean():.4f}, std={centrality.std():.4f}")
#     
#     print("========================\n")


# def compute_cooccurrence_centrality(mdataset, tokenizers, dataset_name, 
#                                      topk=20, complexity_mode='inverse',
#                                      save_dir="./logs"):
#     """
#     Revised version
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     
#     if hasattr(mdataset, "graph_data"):
#         graphs = mdataset.graph_data
#     else:
#         raise AttributeError("mdataset does not have graph_data")
#     
#     cond_sets, proc_sets, drug_sets = aggregate_patient_tokens_from_graphs(graphs)
#     
#     cond_vocab = len(tokenizers['cond_hist'].vocabulary)
#     proc_vocab = len(tokenizers['procedures'].vocabulary)
#     drug_vocab = len(tokenizers['drugs'].vocabulary)
#     
#     print("Computing COND prior...")
#     cond_c = compute_type_cooccurrence_prior(
#         cond_sets, cond_vocab, 
#         topk=topk, complexity_mode=complexity_mode
#     )
#     
#     print("Computing PROC prior...")
#     proc_c = compute_type_cooccurrence_prior(
#         proc_sets, proc_vocab,
#         topk=topk, complexity_mode=complexity_mode
#     )
#     
#     print("Computing DRUG prior...")
#     drug_c = compute_type_cooccurrence_prior(
#         drug_sets, drug_vocab,
#         topk=topk, complexity_mode=complexity_mode
#     )
#     
#     save_path = os.path.join(save_dir, f"cooc_centrality_{dataset_name}.pt")
#     torch.save({"cond": cond_c, "proc": proc_c, "drug": drug_c}, save_path)
#     print(f"Saved co-occurrence centrality to {save_path}")
#     
#     return cond_c, proc_c, drug_c
