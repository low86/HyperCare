# offline_hypergraph.py
import os
import torch
from core.utils import *
from tqdm import tqdm
from pyhealth.data import  Patient
from torch.utils.data import  Dataset



class Hypergraph:
    """
    Build offline hypergraph structures and return a dict that can be saved
    (e.g., to ./graph/cache/{patient_id}.pt).

    Output format (torch.save dict):
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
    All edge tensors are stored on CPU as int64 for easy serialization/sharing.
    """

    def __init__(self, sequence_dataset, tokenizers, note_emb_dict, cache_dir: str = "./graph/cache", device="cpu", feature_key_map: dict | None = None):
        """
        sequence_dataset: list-like of samples (e.g., task_dataset.samples or MMDataset.sequence_dataset)
        tokenizers: dict-like containing keys such as 'cond_hist'/'procedures'/'drugs',
                    each exposing batch_encode_2d(...)
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
        Return a list where each element is _build_single(sample) output.
        This method does not save to disk.
        """
        out = []
        for i in tqdm(range(len(self.sequence_dataset))):
            sample = self.sequence_dataset[i]
            graph = self._build_single(sample)
            out.append(graph)
        return out

    def _build_single(self, sample):
        """
        Build the hypergraph structure and return a dict.
        sample must contain keys:
         - 'patient_id','cond_hist','procedures','drugs','adm_time'
        The format should match samples produced by your data pipeline.
        """
        # ---- basic fields ----
        patient_id = sample["patient_id"]
        cond_hist = sample[self.feature_key_map["cond"]]      # List[List[str]]
        procedures = sample[self.feature_key_map["proc"]]    # List[List[str]]
        drugs = sample[self.feature_key_map["drug"]]              # List[List[str]]
        adm_time_raw = sample["adm_time"]    # List[str] (raw time strings)
        visit_ids = sample["visit_history"]       # ordered by time

        note_emb_list = []

        for v_idx, proc_tokens in enumerate(procedures):
            if len(proc_tokens) == 0:
                continue  # skip empty visit
            # print(type(visit_ids), visit_ids)
            vid = visit_ids[v_idx]  # visit id aligned with procedures
            vid_int = int(vid)
            if vid_int in self.note_emb_dict:
                note_emb_list.append(self.note_emb_dict[vid_int])
            else:
                note_emb_list.append(np.zeros(768, dtype=np.float32))

        note_emb = torch.tensor(note_emb_list, dtype=torch.float32)


        # ---- tokenization (use batch_encode_2d with padding=False) ----
        # If tokenizer keys differ, pass a suitable feature_key_map.
        cond_tokens = self.tokenizers[self.feature_key_map['cond']].batch_encode_2d(cond_hist, padding=False)
        proc_tokens = self.tokenizers[self.feature_key_map['proc']].batch_encode_2d(procedures, padding=False)
        drug_tokens = self.tokenizers[self.feature_key_map['drug']].batch_encode_2d(drugs, padding=False)

        # visit count
        num_visit = len(procedures)

        # ---- compute relative time and store as CPU tensor ----
        try:
            rel_time = convert_to_relative_time(adm_time_raw)
        except Exception as e:
            # Fail fast if convert_to_relative_time is missing or failed.
            raise RuntimeError("convert_to_relative_time not found or failed to run") from e

        rel_time_tensor = torch.tensor(rel_time, dtype=torch.float32, device='cpu')

        # ---- build structure (aligned with your previous _build_structure logic) ----
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

        # visit_offsets: record per-visit start token index (type-wise flatten)
        # Note: in the original implementation, each type has its own index space
        # (0..cond_total-1, 0..proc_total-1, ...). These offsets help lookup later.
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

        # ---- final output ----
        out = {
            "patient_id": patient_id,
            "visit_count": num_visit,

            "cond_tokens": cond_tokens,
            "proc_tokens": proc_tokens,
            "drug_tokens": drug_tokens,

            "adm_time": adm_time_raw,                 # raw time string list (optional)
            "rel_adm_time": rel_time_tensor,         # relative time tensor (CPU)
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
        # Offline hypergraph: build once at initialization
        hg_builder = Hypergraph(self.sequence_dataset, self.tokenizer, self.note_emb_dict, device=self.device, feature_key_map=feature_key_map)
        self.graph_data = hg_builder.build_all_in_one()  # list of graph dicts

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        sequence_data = self.sequence_dataset[idx]
        graph_data = self.graph_data[idx]
        return sequence_data, graph_data
