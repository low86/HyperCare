######################################
# 2025-2-28 #
# Online Hypergraph Generator Version #
######################################

import torch
import numpy as np
from core.utils import *
import torch.nn as nn
from torch_geometric.data import HeteroData
from benchmarks.modules.TSEncoder import Time2Vec


class HypergraphEncoder(nn.Module):
    def __init__(self, tokenizer, hidden_dim, time_dim, device, feature_key_map: dict | None = None, node_emb_path: str | None = None):
        super().__init__()
        if feature_key_map is None:
            feature_key_map = {"cond": "cond_hist", "proc": "procedures", "drug": "drugs"}
        self.feature_key_map = feature_key_map
        self.c_tokenizer = tokenizer[self.feature_key_map['cond']]
        self.d_tokenizer = tokenizer[self.feature_key_map['drug']]
        self.p_tokenizer = tokenizer[self.feature_key_map['proc']]
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.device = torch.device(device)
        node_emb_dict = None
        if node_emb_path:
            node_emb_dict = np.load(node_emb_path, allow_pickle=True).item()

        self.node_emb_dim = hidden_dim
        if node_emb_dict:
            sample_val = next(iter(node_emb_dict.values()))
            if isinstance(sample_val, torch.Tensor):
                self.node_emb_dim = int(sample_val.numel())
            else:
                self.node_emb_dim = int(len(sample_val))

        self.cond_emb = self._init_semantic_embedding(
            self.c_tokenizer, "CCSCM:", self.node_emb_dim, node_emb_dict
        )
        self.proc_emb = self._init_semantic_embedding(
            self.p_tokenizer, "CCSPROC:", self.node_emb_dim, node_emb_dict
        )
        self.drug_emb = self._init_semantic_embedding(
            self.d_tokenizer, "ATC:", self.node_emb_dim, node_emb_dict
        )
        self.time2vec = Time2Vec(time_dim).to(self.device)
        self.global_type_emb = nn.Embedding(3, self.node_emb_dim).to(self.device)
        self.note_projection = nn.Linear(self.time_dim + 768, 256).to(self.device)

    def _unique_token_ids(self, tokens_2d):
        # tokens_2d: list of lists (may include empty lists)
        if len(tokens_2d) == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        # flatten
        flat = []
        for toks in tokens_2d:
            if len(toks) > 0:
                flat.extend(toks)
        if len(flat) == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        uniq = torch.tensor(list(flat), dtype=torch.long, device=self.device)
        return uniq

    def _get_vocab_dict(self, tokenizer):
        v = tokenizer.vocabulary
        if isinstance(v, dict):
            return v
        for attr in ["token2idx", "_token2idx"]:
            if hasattr(v, attr):
                return getattr(v, attr)
        if hasattr(v, "items"):
            try:
                return dict(v)
            except Exception:
                pass
        if hasattr(tokenizer, "convert_indices_to_tokens"):
            size = tokenizer.get_vocabulary_size()
            return {
                tokenizer.convert_indices_to_tokens([i])[0]: i
                for i in range(size)
            }
        raise AttributeError("Tokenizer vocabulary is not accessible as a dict.")

    def _init_semantic_embedding(self, tokenizer, prefix, emb_dim, node_emb_dict):
        if not node_emb_dict:
            return nn.Embedding(tokenizer.get_vocabulary_size(), emb_dim)

        vocab = self._get_vocab_dict(tokenizer)  # token -> index
        vocab_size = tokenizer.get_vocabulary_size()
        weight = torch.zeros((vocab_size, emb_dim), dtype=torch.float32)
        filled = torch.zeros(vocab_size, dtype=torch.bool)
        collected = []

        hit = 0
        for token, idx in vocab.items():
            key = f"{prefix}{token}"
            vec = node_emb_dict.get(key)
            if vec is None:
                continue
            if isinstance(vec, torch.Tensor):
                v = vec.detach().cpu().float()
            else:
                v = torch.tensor(vec, dtype=torch.float32)
            if v.numel() != emb_dim:
                continue
            weight[idx] = v
            filled[idx] = True
            collected.append(v)
            hit += 1

        if collected:
            mean_vec = torch.stack(collected, dim=0).mean(dim=0)
            missing = ~filled
            weight[missing] = mean_vec

        pad_idx = tokenizer.get_padding_index()
        if pad_idx is not None and pad_idx < vocab_size:
            weight[pad_idx] = torch.zeros(emb_dim)

        print(
            f"[semantic_init] {prefix} hits {hit}/{vocab_size}"
        )
        return nn.Embedding.from_pretrained(weight, freeze=False)

    def _aggregate_code_emb(self, emb_table, uniq_ids, agg='mean'):
        if uniq_ids.numel() == 0:
            # return zero vector
            return torch.zeros((self.node_emb_dim,), device=self.device)
        code_embs = emb_table(uniq_ids)  # [U, node_emb_dim]
        if agg == 'mean':
            return code_embs.mean(dim=0)
        elif agg == 'sum':
            return code_embs.sum(dim=0)
        elif agg == 'max':
            return code_embs.max(dim=0)[0]
        else:
            return code_embs.mean(dim=0)

    def _rebuild_heterodata_from_structure(self, struct):
        data = HeteroData()
        node_num = struct['node_num']
        edges = struct['edges']

        # set node counts
        data['cond'].num_nodes = int(node_num.get('cond', 0))
        data['proc'].num_nodes = int(node_num.get('proc', 0))
        data['drug'].num_nodes = int(node_num.get('drug', 0))

        data['cond_hyperedge'].num_nodes = int(node_num.get('cond_hyperedge', 0))
        data['proc_hyperedge'].num_nodes = int(node_num.get('proc_hyperedge', 0))
        data['drug_hyperedge'].num_nodes = int(node_num.get('drug_hyperedge', 0))

        data['time_hyperedge'].num_nodes = int(node_num.get('time_hyperedge', 0))

        # assign edges (move to device)
        for (src_type, rel, dst_type), eidx_cpu in edges.items():
            if eidx_cpu.numel() == 0:
                eidx = torch.empty_like(eidx_cpu, device=self.device)
            else:
                eidx = eidx_cpu.to(self.device)
            data[src_type, rel, dst_type].edge_index = eidx

        return data

    def _fill_node_features(self, data, patient, cond_tokens, proc_tokens, drug_tokens, note_emb):
        # --- time embeddings & time node features ---
        if isinstance(patient.get("rel_adm_time", None), torch.Tensor):
            visit_times = patient["rel_adm_time"].to(self.device)
        else:
            # fallback if not provided
            visit_times = torch.tensor(convert_to_relative_time(patient["adm_time"]), dtype=torch.float32, device=self.device)

        num_visit = data['time_hyperedge'].num_nodes

        if num_visit > 0:
            # use precomputed time_emb if available; otherwise compute
            time_emb = patient.get("time_emb", None)
            if time_emb is None:
                time_emb = self.time2vec(visit_times)  # [num_visit, time_dim]
            else:
                if isinstance(time_emb, torch.Tensor):
                    time_emb = time_emb.to(self.device, dtype=torch.float32)
                else:
                    time_emb = torch.tensor(time_emb, dtype=torch.float32, device=self.device)
                assert time_emb.shape[0] == num_visit, \
                    f"time_emb visits {time_emb.shape[0]} != num_visit {num_visit}"

            data['time_hyperedge'].adm_time = visit_times

            if isinstance(note_emb, torch.Tensor):
                note_emb = note_emb.to(device=self.device, dtype=torch.float32)
            else:
                note_emb = torch.tensor(note_emb, dtype=torch.float32, device=self.device)
            assert note_emb.shape[0] == num_visit, \
                f"note_emb visits {note_emb.shape[0]} != num_visit {num_visit}"
            note_emb = self.note_projection(note_emb)
            data['time_hyperedge'].x = torch.cat([time_emb, note_emb], dim=-1)

        else:
            data['time_hyperedge'].adm_time = torch.empty((0,), dtype=visit_times.dtype, device=self.device)
            data['time_hyperedge'].x = torch.zeros((0, self.node_emb_dim + self.time_dim), device=self.device)

        # --- token nodes: cond/proc/drug ---
        node_types = [
            (self.feature_key_map['cond'], 'cond', cond_tokens, self.cond_emb),
            (self.feature_key_map['proc'], 'proc', proc_tokens, self.proc_emb),
            (self.feature_key_map['drug'], 'drug', drug_tokens, self.drug_emb)
        ]

        for raw_key, node_key, tokens_2d, emb_table in node_types:
            total_nodes = data[node_key].num_nodes
            if total_nodes == 0:
                data[node_key].x = torch.zeros((0, self.node_emb_dim + self.time_dim), device=self.device)
                continue

            emb_list = []
            # tokens_2d is list of visits (including possibly empty lists)
            for v_idx, toks in enumerate(tokens_2d):
                if len(toks) == 0:
                    continue
                tok_tensor = torch.tensor(toks, dtype=torch.long, device=self.device)
                visit_emb = emb_table(tok_tensor)  # [len(toks), node_emb_dim]

                # time embedding for that visit
                t_emb = time_emb[v_idx:v_idx+1].repeat(len(toks), 1)
                visit_emb = torch.cat([visit_emb, t_emb], dim=-1)  # [len(toks), node_emb_dim+time_dim]
                emb_list.append(visit_emb)

            if len(emb_list) == 0:
                data[node_key].x = torch.zeros((0, self.node_emb_dim + self.time_dim), device=self.device)
            else:
                data[node_key].x = torch.cat(emb_list, dim=0)

        # --- type_hyperedge nodes: cond_hyperedge/proc_hyperedge/drug_hyperedge ---
        cond_uniq = self._unique_token_ids(cond_tokens)  # cond_tokens is tokens_2d passed to this function
        cond_agg = self._aggregate_code_emb(self.cond_emb, cond_uniq, agg='sum')  # [hidden_dim]
        # ------- compute for proc -------
        proc_uniq = self._unique_token_ids(proc_tokens)
        proc_agg = self._aggregate_code_emb(self.proc_emb, proc_uniq, agg='sum')
        # ------- compute for drug -------
        drug_uniq = self._unique_token_ids(drug_tokens)
        drug_agg = self._aggregate_code_emb(self.drug_emb, drug_uniq, agg='sum')
        # ------- combine with global_type_emb -------
        # global vectors (shape [1, hidden_dim])
        g_cond = self.global_type_emb(torch.tensor([0], device=self.device)).squeeze(0)  # [hidden_dim]
        g_proc = self.global_type_emb(torch.tensor([1], device=self.device)).squeeze(0)
        g_drug = self.global_type_emb(torch.tensor([2], device=self.device)).squeeze(0)

        # assign to data[type_hyperedge].x
        data['cond_hyperedge'].x = (g_cond + cond_agg).unsqueeze(0)
        data['proc_hyperedge'].x = (g_proc + proc_agg).unsqueeze(0)
        data['drug_hyperedge'].x = (g_drug + drug_agg).unsqueeze(0)

        # --- type ids for token nodes ---
        num_cond = data['cond'].num_nodes
        num_proc = data['proc'].num_nodes
        num_drug = data['drug'].num_nodes

        data['cond'].type_id = torch.zeros(num_cond, dtype=torch.long, device=self.device) if num_cond > 0 else torch.empty((0,), dtype=torch.long, device=self.device)
        data['proc'].type_id = torch.ones(num_proc, dtype=torch.long, device=self.device) if num_proc > 0 else torch.empty((0,), dtype=torch.long, device=self.device)
        data['drug'].type_id = torch.full((num_drug,), 2, dtype=torch.long, device=self.device) if num_drug > 0 else torch.empty((0,), dtype=torch.long, device=self.device)

        # sanity checks
        # assert data['time_hyperedge'].x.shape[-1] == (self.hidden_dim + self.time_dim)
        if data['cond'].num_nodes > 0:
            assert data['cond'].x.shape[-1] == (self.node_emb_dim + self.time_dim)
        if data['proc'].num_nodes > 0:
            assert data['proc'].x.shape[-1] == (self.node_emb_dim + self.time_dim)
        if data['drug'].num_nodes > 0:
            assert data['drug'].x.shape[-1] == (self.node_emb_dim + self.time_dim)

    def forward(self, patient):
        struct = patient['structure']
        data = self._rebuild_heterodata_from_structure(struct)

        cond_tokens = patient.get('cond_tokens', [])
        proc_tokens = patient.get('proc_tokens', [])
        drug_tokens = patient.get('drug_tokens', [])

        note_emb = patient.get('note_emb', None)

        self._fill_node_features(data, patient, cond_tokens, proc_tokens, drug_tokens, note_emb)
        return data
