# offline_hypergraph.py
import os
import torch
from core.utils import *
from tqdm import tqdm
from pyhealth.data import  Patient
from torch.utils.data import  Dataset



class Hypergraph:
    """
    离线构建 hypergraph 结构并保存到 ./graph/cache/{patient_id}.pt

    输出格式（torch.save 的 dict）:
    {
        "patient_id": ...,
        "visit_count": N,
        "cond_tokens": List[List[int]],
        "proc_tokens": List[List[int]],
        "drug_tokens": List[List[int]],
        "adm_time": List[str] or tensor (we save both: raw list + rel_time_tensor),
        "rel_adm_time": torch.tensor([...], dtype=torch.float32),  # relative time
        "structure": {
            "node_num": { 'cond': int, 'proc': int, 'drug': int,
                          'cond_hyperedge':1, 'proc_hyperedge':1, 'drug_hyperedge':1,
                          'time_hyperedge': num_visit },
            "edges": { (src_type, rel, dst_type): torch.LongTensor([[src...],[dst...]]), ... }
        }
    }
    其中 edges 的 tensor 全部保存在 CPU 上的 int64 tensor（便于序列化/共享）。
    """

    def __init__(self, sequence_dataset, tokenizers, note_emb_dict, cache_dir: str = "./graph/cache", device="cpu", feature_key_map: dict | None = None):
        """
        sequence_dataset: list-like of samples (如 task_dataset.samples 或 MMDataset.sequence_dataset)
        tokenizers: dict-like 包含 keys: 'cond_hist'/'procedures'/'drugs'，并提供 batch_encode_2d 方法
        """
        self.sequence_dataset = sequence_dataset
        self.note_emb_dict = note_emb_dict
        self.tokenizers = tokenizers
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if feature_key_map is None:
            feature_key_map = {"cond": "cond_hist", "proc": "procedures", "drug": "drugs"}
        self.feature_key_map = feature_key_map

    def build_all_in_one(self):
        """
        返回一个 list，每个元素是 _build_single(sample) 的结果
        不保存到磁盘
        """
        out = []
        for i in tqdm(range(len(self.sequence_dataset))):
            sample = self.sequence_dataset[i]
            graph = self._build_single(sample)
            out.append(graph)
        return out

    def _build_single(self, sample):
        """
        按照你之前 _build_structure/_fill_node_features 的要求构建结构并返回保存 dict。
        sample 必须包含 keys:
         - 'patient_id','cond_hist','procedures','drugs','adm_time'
         (格式与 diag_prediction_mimic3_fn 生成的 samples 保持一致)
        """
        # ---- 基本字段 ----
        patient_id = sample["patient_id"]
        cond_hist = sample[self.feature_key_map["cond"]]      # List[List[str]]
        procedures = sample[self.feature_key_map["proc"]]    # List[List[str]]
        drugs = sample[self.feature_key_map["drug"]]              # List[List[str]]
        adm_time_raw = sample["adm_time"]    # List[str] (原始时间字符串)
        visit_ids = sample["visit_history"]       # 按时间顺序

        note_emb_list = []

        for v_idx, proc_tokens in enumerate(procedures):
            if len(proc_tokens) == 0:
                continue  # skip empty visit
            # print(type(visit_ids), visit_ids)
            vid = visit_ids[v_idx]  # 对应 procedure visit
            vid_int = int(vid)
            if vid_int in self.note_emb_dict:
                note_emb_list.append(self.note_emb_dict[vid_int])
            else:
                note_emb_list.append(np.zeros(768, dtype=np.float32))

        note_emb = torch.tensor(note_emb_list, dtype=torch.float32)


        # ---- tokenization（严格使用 batch_encode_2d）----
        # 使用你给的 tokenizer 接口名：batch_encode_2d(..., padding=False)
        # 如果 tokenizer 的 key 不同，请在调用方传入合适的 dict
        cond_tokens = self.tokenizers[self.feature_key_map['cond']].batch_encode_2d(cond_hist, padding=False)
        proc_tokens = self.tokenizers[self.feature_key_map['proc']].batch_encode_2d(procedures, padding=False)
        drug_tokens = self.tokenizers[self.feature_key_map['drug']].batch_encode_2d(drugs, padding=False)

        # visit count
        num_visit = len(procedures)

        # ---- 使用你的 convert_to_relative_time 计算相对时间并保存为 CPU tensor ----
        try:
            rel_time = convert_to_relative_time(adm_time_raw)
        except Exception as e:
            # 如果 convert_to_relative_time 不在当前作用域，raise 明确错误
            raise RuntimeError("convert_to_relative_time 未找到或执行失败，请在当前作用域定义该函数") from e

        rel_time_tensor = torch.tensor(rel_time, dtype=torch.float32, device='cpu')

        # ---- 构建拓扑（严格照你之前的 _build_structure 实现） ----
        edges = {}
        node_num = {}

        node_num['time_hyperedge'] = num_visit

        # helper: make_token_struct -> produce total count and edge_index CPU tensor
        def make_token_struct(lens):
            total = sum(lens)
            src_list, dst_list = [], []
            node_offset = 0
            for v_idx, ln in enumerate(lens):
                if ln == 0:
                    continue
                local_nodes = list(range(node_offset, node_offset + ln))
                src_list.extend(local_nodes)
                dst_list.extend([v_idx] * ln)
                node_offset += ln
            if len(src_list) == 0:
                return total, torch.empty((2, 0), dtype=torch.long)
            return total, torch.tensor([src_list, dst_list], dtype=torch.long)

        cond_lens = tuple(len(toks) for toks in cond_tokens)
        proc_lens = tuple(len(toks) for toks in proc_tokens)
        drug_lens = tuple(len(toks) for toks in drug_tokens)

        cond_total, cond_in_time = make_token_struct(cond_lens)
        node_num['cond'] = cond_total
        edges[('cond', 'in_time', 'time_hyperedge')] = cond_in_time

        proc_total, proc_in_time = make_token_struct(proc_lens)
        node_num['proc'] = proc_total
        edges[('proc', 'in_time', 'time_hyperedge')] = proc_in_time

        drug_total, drug_in_time = make_token_struct(drug_lens)
        node_num['drug'] = drug_total
        edges[('drug', 'in_time', 'time_hyperedge')] = drug_in_time

        # type hyperedges (1 each per patient)
        node_num['cond_hyperedge'] = 1
        node_num['proc_hyperedge'] = 1
        node_num['drug_hyperedge'] = 1

        # token -> type_hyperedge within-type connections (src: 0..total-1, dst: zeros)
        if cond_total > 0:
            edges[('cond', 'in_type', 'cond_hyperedge')] = torch.stack([
                torch.arange(cond_total, dtype=torch.long),
                torch.zeros(cond_total, dtype=torch.long)
            ], dim=0)
        else:
            edges[('cond', 'in_type', 'cond_hyperedge')] = torch.empty((2, 0), dtype=torch.long)

        if proc_total > 0:
            edges[('proc', 'in_type', 'proc_hyperedge')] = torch.stack([
                torch.arange(proc_total, dtype=torch.long),
                torch.zeros(proc_total, dtype=torch.long)
            ], dim=0)
        else:
            edges[('proc', 'in_type', 'proc_hyperedge')] = torch.empty((2, 0), dtype=torch.long)

        if drug_total > 0:
            edges[('drug', 'in_type', 'drug_hyperedge')] = torch.stack([
                torch.arange(drug_total, dtype=torch.long),
                torch.zeros(drug_total, dtype=torch.long)
            ], dim=0)
        else:
            edges[('drug', 'in_type', 'drug_hyperedge')] = torch.empty((2, 0), dtype=torch.long)

        # type_hyperedge -> time_hyperedge (type -> all visits)
        if num_visit > 0:
            edges[('cond_hyperedge', 'attend_time', 'time_hyperedge')] = torch.tensor(
                [[0] * num_visit, list(range(num_visit))], dtype=torch.long
            )
            edges[('proc_hyperedge', 'attend_time', 'time_hyperedge')] = torch.tensor(
                [[0] * num_visit, list(range(num_visit))], dtype=torch.long
            )
            edges[('drug_hyperedge', 'attend_time', 'time_hyperedge')] = torch.tensor(
                [[0] * num_visit, list(range(num_visit))], dtype=torch.long
            )

            # reverse: time_hyperedge -> type_hyperedge
            edges[('time_hyperedge', 'attend_type', 'cond_hyperedge')] = torch.tensor(
                [list(range(num_visit)), [0] * num_visit], dtype=torch.long
            )
            edges[('time_hyperedge', 'attend_type', 'proc_hyperedge')] = torch.tensor(
                [list(range(num_visit)), [0] * num_visit], dtype=torch.long
            )
            edges[('time_hyperedge', 'attend_type', 'drug_hyperedge')] = torch.tensor(
                [list(range(num_visit)), [0] * num_visit], dtype=torch.long
            )
        else:
            edges[('cond_hyperedge', 'attend_time', 'time_hyperedge')] = torch.empty((2,0), dtype=torch.long)
            edges[('proc_hyperedge', 'attend_time', 'time_hyperedge')] = torch.empty((2,0), dtype=torch.long)
            edges[('drug_hyperedge', 'attend_time', 'time_hyperedge')] = torch.empty((2,0), dtype=torch.long)
            edges[('time_hyperedge', 'attend_type', 'cond_hyperedge')] = torch.empty((2,0), dtype=torch.long)
            edges[('time_hyperedge', 'attend_type', 'proc_hyperedge')] = torch.empty((2,0), dtype=torch.long)
            edges[('time_hyperedge', 'attend_type', 'drug_hyperedge')] = torch.empty((2,0), dtype=torch.long)

        # time_hyperedge connect (visit->visit) same rule as your code
        if num_visit > 1:
            src_list, dst_list = [], []
            if num_visit > 4:
                window_size = 4
                for i in range(num_visit):
                    j_start = max(0, i - window_size)
                    js = list(range(j_start, i + 1))
                    src_list.extend(js)
                    dst_list.extend([i] * len(js))
            else:
                src_list = [i for i in range(num_visit) for _ in range(num_visit)]
                dst_list = [j for _ in range(num_visit) for j in range(num_visit)]
            edges[('time_hyperedge', 'connect', 'time_hyperedge')] = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
        else:
            edges[('time_hyperedge', 'connect', 'time_hyperedge')] = torch.empty((2,0), dtype=torch.long)

        struct = {'node_num': node_num, 'edges': edges}

        # ---- visit_node_offset: 为了兼容你希望的格式，记录每个 visit 的起始 token node index（以 type-wise flatten，保持与你的 _build_structure 对齐）
        # 注意：在你原始实现中，每 type 的节点是单独编号（0..cond_total-1, 0..proc_total-1, ...）
        # 这里我们提供每个 visit 在每个 type 内的 offset（便于 later 查表）
        def make_visit_offsets(lens):
            offsets = []
            off = 0
            for ln in lens:
                offsets.append(off)
                off += ln
            return offsets

        visit_offsets = {
            'cond': make_visit_offsets(cond_lens),
            'proc': make_visit_offsets(proc_lens),
            'drug': make_visit_offsets(drug_lens)
        }

        # ---- 最终保存结构 ----
        out = {
            "patient_id": patient_id,
            "visit_count": num_visit,

            "cond_tokens": cond_tokens,
            "proc_tokens": proc_tokens,
            "drug_tokens": drug_tokens,

            "adm_time": adm_time_raw,                 # 原始字符串 list（可选）
            "rel_adm_time": rel_time_tensor,         # 相对时间 tensor (CPU)
            "note_emb": note_emb,

            "structure": struct,
            "visit_offsets": visit_offsets,          # per-type visit start offsets (CPU ints)
        }

        return out

class MMDataset(Dataset):
    def __init__(self, dataset, tokenizer, note_emb_dict, device, feature_key_map: dict | None = None):
        self.sequence_dataset = dataset.samples
        self.tokenizer = tokenizer
        self.note_emb_dict = note_emb_dict
        self.device = device
        # 离线构图：一次性构建
        hg_builder = Hypergraph(self.sequence_dataset, self.tokenizer, self.note_emb_dict, device=self.device, feature_key_map=feature_key_map)
        self.graph_data = hg_builder.build_all_in_one()  # list of graph dicts

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        sequence_data = self.sequence_dataset[idx]
        graph_data = self.graph_data[idx]
        return sequence_data, graph_data