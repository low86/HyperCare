
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
    print("=== 1️⃣ 数值范围检查 ===")
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

    print("\n=== 2️⃣ Top-k token 查看 ===")
    def topk_tokens(c_tensor, name):
        top_idx = torch.topk(c_tensor, k=min(topk_show, len(c_tensor)))[1].tolist()
        print(f"Top {topk_show} {name} tokens: {top_idx}")

    topk_tokens(cond_c, "cond")
    topk_tokens(proc_c, "proc")
    topk_tokens(drug_c, "drug")

    print("\n=== 3️⃣ 共现矩阵可视化 (sample) ===")
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

    print("\n=== 4️⃣ 稀疏化检查 ===")
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
    print("\n✅ 共现中心度检查完成！")

# def aggregate_patient_tokens_from_graphs(graphs):
#     """
#     从 graph_data 中按患者去重聚合 tokens
#     """
#     from collections import defaultdict
    
#     patient_tokens = defaultdict(lambda: {"cond": set(), "proc": set(), "drug": set()})
    
#     for g in tqdm(graphs, desc="Aggregate patient tokens"):
#         patient_id = g["patient_id"]  # ← 关键：按患者分组
        
#         # 聚合该样本的所有 visit tokens（样本内可能有多个 visit）
#         for visit in g.get("cond_tokens", []):
#             patient_tokens[patient_id]["cond"].update(visit)
#         for visit in g.get("proc_tokens", []):
#             patient_tokens[patient_id]["proc"].update(visit)
#         for visit in g.get("drug_tokens", []):
#             patient_tokens[patient_id]["drug"].update(visit)
    
#     # 转换为 list（每个患者一个 set）
#     cond_sets = [tokens["cond"] for tokens in patient_tokens.values()]
#     proc_sets = [tokens["proc"] for tokens in patient_tokens.values()]
#     drug_sets = [tokens["drug"] for tokens in patient_tokens.values()]
    
#     print(f"Total unique patients: {len(patient_tokens)}")
#     return cond_sets, proc_sets, drug_sets


# #绝对共现的代码
def aggregate_patient_tokens_from_graphs(graphs):
    """
    graphs: list of dicts, 每个 dict 是 MMDataset.graph_data 的元素
    返回每种类型的 token 集合列表（每个患者一个 set）
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

        # 每个 visit 的 token 列表
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
#     简化版：只用绝对共现次数,都是慢性病会高
#     """
#     C = np.zeros((vocab_size, vocab_size), dtype=np.float32)

#     for tokens in tqdm(patient_token_sets, desc="Counting co-occurrence"):
#         uniq = list(tokens)
#         for i in uniq:
#             for j in uniq:
#                 if i != j:
#                     C[i, j] += 1

#     # 直接求和（不做稀疏化，慢性病自然会高）
#     c = C.sum(axis=1)
    
#     if c.max() > 0:
#         c = c / c.max()

#     return torch.tensor(c, dtype=torch.float32)

# def compute_cooccurrence_centrality(mdataset, tokenizers, dataset_name, tau=0.005, topk=24, save_dir="./logs"):
#     """
#     mdataset: MMDataset 实例
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # 使用 MMDataset 的 graph_data
#     if hasattr(mdataset, "graph_data"):
#         graphs = mdataset.graph_data
#     else:
#         raise AttributeError("mdataset 没有 graph_data 属性，请确保传入的是 MMDataset 实例")

#     cond_sets, proc_sets, drug_sets = aggregate_patient_tokens_from_graphs(graphs)

#     # vocab size 推荐从 tokenizer 或 semantic table 获取，这里用简单 max+1
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

# # 计算共现矩阵及相关指标的函数（被注释掉的旧版）
def compute_cooccurrence_matrix(patient_token_sets, vocab_size):
    """
    返回：
    C: [V, V] 共现矩阵
    freq: [V] 出现频率
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
    返回 P: [V, V]，值域 [0,1]
    """
    freq_i = freq[:, None]
    freq_j = freq[None, :]
    denom = freq_i + freq_j - C + eps
    P = C / denom
    P[np.isnan(P)] = 0.0
    return P

def compute_complexity(P, eps=0.01):
    """
    complexity_i = log(1 + 与多少疾病有显著共现)
    """
    # 每个 i 与多少 j 有结构性关联
    breadth = (P > eps).sum(axis=1).astype(np.float32)
    complexity = np.log1p(breadth)

    # 归一化到 [0,1]
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
    mdataset: MMDataset 实例
    """
    os.makedirs(save_dir, exist_ok=True)

    # 使用 MMDataset 的 graph_data
    if hasattr(mdataset, "graph_data"):
        graphs = mdataset.graph_data
    else:
        raise AttributeError("mdataset 没有 graph_data 属性，请确保传入的是 MMDataset 实例")

    cond_sets, proc_sets, drug_sets = aggregate_patient_tokens_from_graphs(graphs)

    # vocab size 推荐从 tokenizer 或 semantic table 获取，这里用简单 max+1
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
#     返回：
#     C: [V, V] 共现矩阵
#     freq: [V] 出现频率
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
#     ✅ 改动 1：只保留 top-k 共现，增加对比性和稀疏性
#     """
#     P_sparse = np.zeros_like(P)
#     for i in range(P.shape[0]):
#         if np.count_nonzero(P[i]) > 0:
#             # 只保留每行的 top-k
#             idx = np.argsort(P[i])[-k:]
#             P_sparse[i, idx] = P[i, idx]
#     return P_sparse


# def compute_jaccard_cooccurrence(C, freq, eps=1e-8, topk=20):
#     """
#     返回 P: [V, V]，值域 [0,1]
#     ✅ 修改：增加 top-k 稀疏化
#     """
#     freq_i = freq[:, None]
#     freq_j = freq[None, :]
#     denom = freq_i + freq_j - C + eps
#     P = C / denom
#     P[np.isnan(P)] = 0.0
    
#     # ✅ 关键：稀疏化，只保留 top-k
#     P = sparsify_topk(P, k=topk)
    
#     return P


# def compute_complexity(P, eps=0.01, mode='inverse'):
#     """
#     ✅ 改动 2：complexity 奖励"专一"而非"常见"
    
#     Args:
#         mode: 
#             - 'inverse': 反 degree（推荐）
#             - 'idf': IDF-like
#             - 'log': 原始 log (不推荐)
#     """
#     breadth = (P > eps).sum(axis=1).astype(np.float32)
    
#     if mode == 'inverse':
#         # ✅ 推荐：与越少 code 强相关的，complexity 越高
#         complexity = 1.0 / np.sqrt(breadth + 1.0)
#     elif mode == 'idf':
#         # IDF-like: log(total / breadth)
#         total = P.shape[0]
#         complexity = np.log(total / (breadth + 1.0))
#     else:  # 'log' (原始，不推荐)
#         complexity = np.log1p(breadth)
    
#     # 归一化到 [0,1]
#     if complexity.max() > complexity.min():
#         complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min())
    
#     return complexity


# def compute_absolute_centrality(C, zero_mean=True):
#     """
#     ✅ 改动 3：centrality 做 zero-mean，避免全正偏置
#     """
#     c = C.sum(axis=1)
#     if c.max() > 0:
#         c = c / c.max()
    
#     if zero_mean:
#         # ✅ 关键：zero-mean normalization
#         c_ = c - c.mean()
    
#     return c, c_


# def compute_type_cooccurrence_prior(
#     patient_token_sets,
#     vocab_size,
#     save_path=None,
#     topk=20,              # ← 新增参数
#     complexity_mode='inverse',  # ← 新增参数
#     zero_mean_centrality=True   # ← 新增参数
# ):
#     """
#     ✅ 完整修复版本
#     """
#     C, freq = compute_cooccurrence_matrix(patient_token_sets, vocab_size)
    
#     # ✅ 修改 1: 稀疏化的 Jaccard
#     P = compute_jaccard_cooccurrence(C, freq, topk=topk)
    
#     # ✅ 修改 2: 反 degree complexity
#     complexity = compute_complexity(P, mode=complexity_mode)
    
#     # ✅ 修改 3: zero-mean centrality
#     centrality, centrality_zero_mean = compute_absolute_centrality(C, zero_mean=zero_mean_centrality)
    
#     out = {
#         "C": torch.tensor(C, dtype=torch.float32),
#         "P": torch.tensor(P, dtype=torch.float32),
#         "complexity": torch.tensor(complexity, dtype=torch.float32),
#         "centrality": torch.tensor(centrality, dtype=torch.float32),
#         "centrality_zero_mean": torch.tensor(centrality_zero_mean, dtype=torch.float32),
#     }
    
#     if save_path is not None:
#         torch.save(out, save_path)
#         print(f"Saved co-occurrence prior to {save_path}")
    
#     # ✅ 新增：打印诊断信息
#     print_prior_stats(out, vocab_size)
    
#     return out


# def print_prior_stats(prior_dict, vocab_size):
#     """
#     打印统计信息，用于验证修复效果
#     """
#     P = prior_dict['P'].numpy()
#     complexity = prior_dict['complexity'].numpy()
#     centrality = prior_dict['centrality'].numpy()
    
#     print("\n=== Prior Statistics ===")
#     print(f"Vocab size: {vocab_size}")
    
#     # P 的稀疏性
#     P_nonzero = (P > 0).sum()
#     P_density = P_nonzero / (vocab_size * vocab_size)
#     print(f"P density: {P_density:.4f} ({P_nonzero} / {vocab_size**2})")
#     print(f"P avg nonzero per row: {(P > 0).sum(axis=1).mean():.2f}")
    
#     # Complexity 分布
#     print(f"Complexity: min={complexity.min():.4f}, max={complexity.max():.4f}, "
#           f"mean={complexity.mean():.4f}, std={complexity.std():.4f}")
    
#     # Centrality 分布（应该接近 zero-mean）
#     print(f"Centrality: min={centrality.min():.4f}, max={centrality.max():.4f}, "
#           f"mean={centrality.mean():.4f}, std={centrality.std():.4f}")
    
#     print("========================\n")


# def compute_cooccurrence_centrality(mdataset, tokenizers, dataset_name, 
#                                      topk=20, complexity_mode='inverse',
#                                      save_dir="./logs"):
#     """
#     ✅ 修复版本
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     if hasattr(mdataset, "graph_data"):
#         graphs = mdataset.graph_data
#     else:
#         raise AttributeError("mdataset 没有 graph_data 属性")
    
#     cond_sets, proc_sets, drug_sets = aggregate_patient_tokens_from_graphs(graphs)
    
#     cond_vocab = len(tokenizers['cond_hist'].vocabulary)
#     proc_vocab = len(tokenizers['procedures'].vocabulary)
#     drug_vocab = len(tokenizers['drugs'].vocabulary)
    
#     print("Computing COND prior...")
#     cond_c = compute_type_cooccurrence_prior(
#         cond_sets, cond_vocab, 
#         topk=topk, complexity_mode=complexity_mode
#     )
    
#     print("Computing PROC prior...")
#     proc_c = compute_type_cooccurrence_prior(
#         proc_sets, proc_vocab,
#         topk=topk, complexity_mode=complexity_mode
#     )
    
#     print("Computing DRUG prior...")
#     drug_c = compute_type_cooccurrence_prior(
#         drug_sets, drug_vocab,
#         topk=topk, complexity_mode=complexity_mode
#     )
    
#     save_path = os.path.join(save_dir, f"cooc_centrality_{dataset_name}.pt")
#     torch.save({"cond": cond_c, "proc": proc_c, "drug": drug_c}, save_path)
#     print(f"Saved co-occurrence centrality to {save_path}")
    
#     return cond_c, proc_c, drug_c