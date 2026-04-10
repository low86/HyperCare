######################################
# 2025-10-28 #
# Online Hypergraph Generator Version #
######################################

from random import random
import sys
import math
import torch
from utils import *
import torch.nn as nn
from torchdiffeq import odeint
from models.bottleneck import *
from torch_geometric.nn import Linear

from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from benchmarks.modules.TSEncoder import Time2Vec
# from models.HyperEncoder import HypergraphEncoder
from benchmarks.modules.Seqmodels import TransformerLayer
from torch_scatter import scatter_add, scatter_mean, scatter_softmax, scatter_sum, scatter_max


class HypergraphEncoder(nn.Module):
    def __init__(self, tokenizer, hidden_dim, time_dim, device):
        super().__init__()
        self.c_tokenizer = tokenizer['cond_hist']
        self.d_tokenizer = tokenizer['drugs']
        self.p_tokenizer = tokenizer['procedures']
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.device = torch.device(device)
        self.cond_emb = nn.Embedding(self.c_tokenizer.get_vocabulary_size(), hidden_dim)
        self.proc_emb = nn.Embedding(self.p_tokenizer.get_vocabulary_size(), hidden_dim)
        self.drug_emb = nn.Embedding(self.d_tokenizer.get_vocabulary_size(), hidden_dim)
        self.time2vec = Time2Vec(time_dim).to(self.device)
        self.global_type_emb = nn.Embedding(3, hidden_dim).to(self.device)
        self.note_projection = nn.Linear(256*3, 64).to(self.device)

    def _unique_token_ids(self, tokens_2d):
        # tokens_2d: list of lists (可能有空列表)
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

    def _aggregate_code_emb(self, emb_table, uniq_ids, agg='mean'):
        if uniq_ids.numel() == 0:
            # return zero vector
            return torch.zeros((self.hidden_dim,), device=self.device)
        code_embs = emb_table(uniq_ids)  # [U, hidden_dim]
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

    def _fill_node_features(self, data, patient, patient_emb, cond_tokens, proc_tokens, drug_tokens, note_emb):
        # --- time embeddings & time node features ---
        if isinstance(patient.get("rel_adm_time", None), torch.Tensor):
            visit_times = patient["rel_adm_time"].to(self.device)
        else:
            # fallback if not provided
            visit_times = torch.tensor(convert_to_relative_time(patient["adm_time"]), dtype=torch.float32, device=self.device)

        num_visit = data['time_hyperedge'].num_nodes

        if num_visit > 0:
            # compute time embedding once
            time_emb = self.time2vec(visit_times)  # [num_visit, time_dim]

            data['time_hyperedge'].adm_time = visit_times
            # ensure patient_emb on device and correct shape [hidden_dim]
            if patient_emb.device != self.device:
                patient_emb = patient_emb.to(self.device)
            # =====================================================
            # >>>>> 修改开始：加入 note_emb（来自离线构图） <<<<<
            # =====================================================
            if note_emb is not None:
                note_emb = torch.tensor(note_emb, dtype=torch.float32, device=self.device)
                assert note_emb.shape[0] == num_visit, \
                    f"note_emb visits {note_emb.shape[0]} != num_visit {num_visit}"
                note_emb = self.note_projection(note_emb)
            else:
                note_emb = torch.zeros((num_visit, 256), device=self.device)

            patient_part = patient_emb.unsqueeze(0).repeat(num_visit, 1)

            data['time_hyperedge'].x = torch.cat(
                [time_emb, note_emb], dim=-1
            )

        else:
            data['time_hyperedge'].adm_time = torch.empty((0,), dtype=visit_times.dtype, device=self.device)
            data['time_hyperedge'].x = torch.zeros((0, self.hidden_dim + self.time_dim), device=self.device)

        # --- token nodes: cond/proc/drug ---
        node_types = [
            ('cond_hist', 'cond', cond_tokens, self.cond_emb),
            ('procedures', 'proc', proc_tokens, self.proc_emb),
            ('drugs', 'drug', drug_tokens, self.drug_emb)
        ]

        for raw_key, node_key, tokens_2d, emb_table in node_types:
            total_nodes = data[node_key].num_nodes
            if total_nodes == 0:
                data[node_key].x = torch.zeros((0, self.hidden_dim + self.time_dim), device=self.device)
                continue

            emb_list = []
            # tokens_2d is list of visits (including possibly empty lists)
            for v_idx, toks in enumerate(tokens_2d):
                if len(toks) == 0:
                    continue
                tok_tensor = torch.tensor(toks, dtype=torch.long, device=self.device)
                visit_emb = emb_table(tok_tensor)  # [len(toks), hidden_dim]

                # time embedding for that visit
                t_emb = time_emb[v_idx:v_idx+1].repeat(len(toks), 1)
                visit_emb = torch.cat([visit_emb, t_emb], dim=-1)  # [len(toks), hidden+time_dim]
                emb_list.append(visit_emb)

            if len(emb_list) == 0:
                data[node_key].x = torch.zeros((0, self.hidden_dim + self.time_dim), device=self.device)
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
            assert data['cond'].x.shape[-1] == (self.hidden_dim + self.time_dim)
        if data['proc'].num_nodes > 0:
            assert data['proc'].x.shape[-1] == (self.hidden_dim + self.time_dim)
        if data['drug'].num_nodes > 0:
            assert data['drug'].x.shape[-1] == (self.hidden_dim + self.time_dim)

    def forward(self, patient, patient_emb):
        struct = patient['structure']
        data = self._rebuild_heterodata_from_structure(struct)

        cond_tokens = patient.get('cond_tokens', [])
        proc_tokens = patient.get('proc_tokens', [])
        drug_tokens = patient.get('drug_tokens', [])

        note_emb = patient.get('note_emb', None)

        if patient_emb.device != self.device:
            patient_emb = patient_emb.to(self.device)

        self._fill_node_features(data, patient, patient_emb, cond_tokens, proc_tokens, drug_tokens, note_emb)
        return data


class Time2NodeAggregatorImproved(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, step_size=0.5, lambda2=0.2, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.step_size = nn.Parameter(torch.tensor(step_size))
        self.lambda2 = nn.Parameter(torch.tensor(lambda2))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sigmoid()  # new adaptive gate

    def forward(self, data, node_types=['cond','proc','drug']):
        for node_type in node_types:
            if node_type not in data.node_types: continue
            edge_type = (node_type, 'in_time', 'time_hyperedge')
            if edge_type not in data.edge_types: continue
            src, dst = data[edge_type].edge_index
            node_x = data[node_type].x
            time_x = data['time_hyperedge'].x

            q = self.q_proj(node_x[src]).view(-1, self.num_heads, self.head_dim)
            k = self.k_proj(time_x[dst]).view(-1, self.num_heads, self.head_dim)
            v = self.v_proj(time_x[dst]).view(-1, self.num_heads, self.head_dim)

            att_score = (q * k).sum(-1) / (self.head_dim ** 0.5)
            att = scatter_softmax(self.leaky(att_score), src, dim=0).unsqueeze(-1)
            att = self.dropout(att)
            agg_heads = scatter_add(att * v, src, dim=0, dim_size=node_x.size(0))
            agg = self.out_proj(agg_heads.view(node_x.size(0), self.hidden_dim))

            # ODE-inspired update + residual + gate
            dN = (agg - node_x) - self.lambda2 * node_x
            gate = self.gate(self.step_size)  # learnable between (0,1)
            N_new = node_x + gate * dN
            data[node_type].x = self.layernorm(N_new + node_x)  # residual for gradient
        return data

# -------------------------------------------#
# PatientSequenceEncoder #
# -------------------------------------------#
class PatientSequenceEncoder(nn.Module):

    def __init__(
            self,
            tokenizers: dict,
            hidden_dim: int,
            output_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.5,
            device: str = 'cuda'
    ):
        super(PatientSequenceEncoder, self).__init__()
        self.device = torch.device(device)
        self.num_heads = num_heads
        self.num_layers = num_layers    
        self.hidden_dim = hidden_dim
        self.tokenizers = tokenizers
        self.feature_keys = list(tokenizers.keys())

        # ===== Embedding 层 =====
        self.embeddings = nn.ModuleDict()
        for key, tok in tokenizers.items():
            self.embeddings[key] = nn.Embedding(
                num_embeddings=tok.get_vocabulary_size(),
                embedding_dim=hidden_dim,
                padding_idx=tok.get_padding_index()
            )

        # ===== Transformer Encoder 层 =====
        self.transformers = nn.ModuleDict({
            key: TransformerLayer(feature_size=hidden_dim, heads=self.num_heads, dropout=dropout, num_layers=self.num_layers)
            for key in self.feature_keys
        })

        self.bottleneck_fusion = BottleneckFusion(
            dim=hidden_dim,
            num_tokens=3,
            num_heads=2,
            num_layers=1
        )

        # ===== 时间向量编码器 + 输出层 =====
        self.fc = nn.Linear(hidden_dim * len(self.feature_keys), output_dim)
        self.gammas = nn.Parameter(torch.zeros(len(self.feature_keys)))
        self.to(self.device)

    # ============== Forward ==============
    def forward(self, batchdata):
        patient_emb = []
        patient_cls = []

        for key in self.feature_keys:
            tokenizer = self.tokenizers[key]
            x = tokenizer.batch_encode_3d(batchdata[key])  # [B, T, L]
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[key](x)  # [B, T, L, H]
            x = torch.sum(x, dim=2)  # [B, T, H]
            mask = torch.any(x != 0, dim=2)
            x_encoded, x_cls = self.transformers[key](x, mask)  # x: [B, H]
            patient_emb.append(x_encoded)
            patient_cls.append(x_cls)

        # fused = self.bottleneck_fusion(patient_emb)  # [B, K=M, D]
        fused = self.bottleneck_fusion(patient_emb)  # [B, K=M, D]
        modality_reprs = [
            cls + torch.sigmoid(self.gammas[i]) * (fused[:, i, :] - cls)
            for i, cls in enumerate(patient_cls)
        ]
        patient_embs = torch.cat(modality_reprs, dim=1)  # [B, M*D]
        seq_logits = self.fc(patient_embs)
        return seq_logits, patient_embs


# -------------------------------------------#
# Node to time hyperedge   #
# -------------------------------------------#
class Node2TimeAggregator(nn.Module):
    def __init__(self, hidden_dim, time_dim, mode='attention', num_heads=4, att_dropout=0.5, gamma_time=0.8,
                 tau_time=4.0):
        super(Node2TimeAggregator, self).__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.time_dim = time_dim
        self.gamma_time = nn.Parameter(torch.tensor(gamma_time))
        self.tau_time = nn.Parameter(torch.tensor(tau_time))
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        if mode == 'attention':
            assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
            self.head_dim = hidden_dim // num_heads
            # 投影矩阵
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.leakyrelu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(att_dropout)
        elif mode in ['sum', 'mean']:
            pass
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _compute_time_factor(self, visit_times, dst, batch_idx):
        t_dst = visit_times[dst]  # [E]
        dst_batch = batch_idx[dst]  # [E]

        # 每个 batch 的最大时间
        t_last_per_batch, _ = scatter_max(visit_times, batch_idx, dim=0)  # [B]
        t_last_edges = t_last_per_batch[dst_batch]  # [E]

        delta = (t_last_edges - t_dst).clamp(min=0.0)
        delta_log = torch.log1p(delta)

        gamma_pos = F.softplus(self.gamma_time)
        tau_pos = F.softplus(self.tau_time)
        factor = 1.0 + gamma_pos * torch.exp(- delta_log / (tau_pos + 1e-6))
        return factor.unsqueeze(-1)  # [E,1]

    def forward(self, data, node_types=['cond', 'proc', 'drug']):
        device = data['time_hyperedge'].x.device
        num_visits = data['time_hyperedge'].num_nodes
        agg_all = torch.zeros(num_visits, self.hidden_dim, device=device)

        for node_type in node_types:
            if node_type not in data.node_types or data[node_type].num_nodes == 0:
                continue

            edge_type = (node_type, 'in_time', 'time_hyperedge')
            if edge_type not in data.edge_types:
                continue
            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0:
                continue

            src, dst = edge_index
            node_x = data[node_type].x
            time_x = data['time_hyperedge'].x

            if self.mode == 'sum':
                agg = scatter_add(node_x[src], dst, dim=0, dim_size=num_visits)
            elif self.mode == 'mean':
                agg = scatter_mean(node_x[src], dst, dim=0, dim_size=num_visits)
            elif self.mode == 'attention':
                def reshape_heads(x):
                    return x.view(-1, self.num_heads, self.head_dim)

                # projections
                q_all = self.q_proj(time_x)
                k_all = self.k_proj(node_x)
                v_all = self.v_proj(node_x)

                # index per-edge
                q = reshape_heads(q_all[dst])  # [E, H, D_h]
                k = reshape_heads(k_all[src])  # [E, H, D_h]
                v = reshape_heads(v_all[src])  # [E, H, D_h]

                # base attention
                att_score = (q * k).sum(-1) / (self.head_dim ** 0.5)
                att_score = self.leakyrelu(att_score)
                batch_idx = data['time_hyperedge'].batch
                visit_times = data['time_hyperedge'].adm_time  # [num_visits]
                time_factor_h = self._compute_time_factor(visit_times, dst, batch_idx)
                att_score = att_score * time_factor_h
                # normalize
                att = scatter_softmax(att_score, dst, dim=0)
                att = self.dropout(att).unsqueeze(-1)
                # aggregate
                agg_heads = scatter_add(att * v, dst, dim=0, dim_size=num_visits)
                agg = agg_heads.reshape(num_visits, self.hidden_dim)
            else:
                raise ValueError("Invalid mode")
            agg_all += agg
        data['time_hyperedge'].x = data['time_hyperedge'].x + agg_all
        return data
    


class Time2NodeAggregator(nn.Module):
    """
    Time->Node feedback: 把 time_hyperedge 的信息聚合回各类节点，返回更新后的节点张量 N_new
    我们做残差式更新： N_new = N + step * (agg - N) + lambda2 * (-N)
    """
    def __init__(self, hidden_dim, time_dim, mode='attention', num_heads=4, att_dropout=0.5, gamma_time=1.5, tau_time=15.0):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.time_dim = time_dim

        if mode == 'attention':
            assert hidden_dim % num_heads == 0
            self.head_dim = hidden_dim // num_heads
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.leakyrelu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(att_dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, data, node_types=['cond','proc','drug']):
        for node_type in node_types:
            if node_type not in data.node_types or data[node_type].num_nodes == 0:
                continue
            edge_type = (node_type, 'in_time', 'time_hyperedge')
            if edge_type not in data.edge_types:
                continue
            edge_index = data[edge_type].edge_index
            src, dst = edge_index  # src=node, dst=time
            node_x = data[node_type].x
            time_x = data['time_hyperedge'].x

            if self.mode in ['sum','mean']:
                agg = scatter_add(time_x[dst], src, dim=0, dim_size=node_x.size(0)) \
                       if self.mode=='sum' else scatter_mean(time_x[dst], src, dim=0, dim_size=node_x.size(0))
            elif self.mode == 'attention':
                # multi-head attention
                def reshape_heads(x):
                    return x.view(-1, self.num_heads, self.head_dim)
                q = reshape_heads(node_x[src])
                k = reshape_heads(time_x[dst])
                v = reshape_heads(time_x[dst])
                att_score = (q*k).sum(-1)/ (self.head_dim**0.5)
                att_score = self.leakyrelu(att_score)
                att = scatter_softmax(att_score, src, dim=0).unsqueeze(-1)
                att = self.dropout(att)
                agg_heads = scatter_add(att*v, src, dim=0, dim_size=node_x.size(0))
                agg = agg_heads.view(node_x.size(0), self.hidden_dim)
                agg = self.out_proj(agg)
            else:
                raise ValueError("Invalid mode")
            # residual + layernorm
            # data[node_type].x = self.layernorm(node_x + agg)
            data[node_type].x = node_x + agg
        return data


class Time2TimeAggregator(nn.Module):
    """
    基于 edge_index 的局部时间注意力（past-only window）。
    使用边 (src -> dst) 表示 src (past visit) 对 dst (current visit) 的影响。
    复杂度 O(E).
    """
    def __init__(self, hidden_dim, num_heads=4, time_decay_init=365.0, dropout=0.5):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # query 用 dst 的 time node
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)  # key 用 src 的 time node
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.time_decay = nn.Parameter(torch.tensor(time_decay_init))
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        # 可选的 residual scale
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        # 若没有连接边直接返回
        edge_key = ('time_hyperedge', 'connect', 'time_hyperedge')
        if edge_key not in data.edge_types:
            return data

        H = data['time_hyperedge'].x                     # [N_time, D]
        times = data['time_hyperedge'].adm_time.float()   # [N_time]
        edge_index = data[edge_key].edge_index.to(H.device)  # [2, E]
        src, dst = edge_index[0], edge_index[1]           # src->dst (past->current)

        # 投影并分头
        Q_all = self.q_proj(H).view(-1, self.num_heads, self.head_dim)  # [N_time, H, dh]
        K_all = self.k_proj(H).view(-1, self.num_heads, self.head_dim)
        V_all = self.v_proj(H).view(-1, self.num_heads, self.head_dim)

        q_edges = Q_all[dst]
        k_edges = K_all[src]
        v_edges = V_all[src]

        att_score = (q_edges * k_edges).sum(-1) / (self.head_dim ** 0.5)
        # 时间衰减 weight per-edge: exp(-|Δt| / tau)
        tau = F.softplus(self.time_decay) + 1e-8
        delta_t = (times[dst] - times[src]).abs()  # should be >=0 if past-only, shape [E]
        decay = torch.exp(- delta_t / tau).unsqueeze(-1)  # [E, 1]
        att_score = att_score * decay  # scale attention by decay

        att_norm = scatter_softmax(att_score, dst, dim=0)  # [E, H]
        att_norm = self.dropout(att_norm).unsqueeze(-1)   # [E, H, 1]

        weighted_v = att_norm * v_edges  # [E, H, dh]
        out_heads = scatter_add(weighted_v, dst, dim=0, dim_size=H.size(0))  # [N_time, H, dh]

        out = out_heads.permute(0, 1, 2).contiguous().view(H.size(0), -1)
        out = self.out_proj(out)

        data['time_hyperedge'].x = H + out

        return data
    

class Type2TimeAggregator(nn.Module):
    """
    将类型超边 (cond/proc/drug) 聚合到时间超边 (visit) 上
    支持 multi-head attention 和时间衰减
    """
    def __init__(self, dim, num_heads=4, dropout=0.5, gamma_time=1.5, tau_time=15.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        # 类型融合 (可选 transformer fusion)
        self.fusion_attn = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim*4, dropout=dropout
        )
        self.gammas = nn.Parameter(torch.zeros(3))  # cond/proc/drug residual scale

        # Multi-Head Attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # 时间衰减
        self.gamma_time = nn.Parameter(torch.tensor(gamma_time))
        self.tau_time = nn.Parameter(torch.tensor(tau_time))

    def _compute_time_factor(self, visit_times, dst):
        """
        visit_times: [num_time_nodes] tensor
        dst: edge dst index
        """
        t_dst = visit_times[dst]
        t_last = visit_times.max()
        delta = (t_last - t_dst).clamp(min=0.0)
        delta_log = torch.log1p(delta)
        gamma_pos = F.softplus(self.gamma_time)
        tau_pos = F.softplus(self.tau_time)
        factor = 1.0 + gamma_pos * torch.exp(- delta_log / (tau_pos + 1e-6))
        return factor.unsqueeze(-1)  # [E,1]

    def forward(self, g):

        # 1️⃣ 类型融合
        Hc = g['cond_hyperedge'].x
        Hd = g['drug_hyperedge'].x
        Hp = g['proc_hyperedge'].x
        H_types = torch.stack([Hc, Hd, Hp], dim=1)  # [B,3,D]
        fused = self.fusion_attn(H_types)
        g['cond_hyperedge'].x = Hc + torch.sigmoid(self.gammas[0]) * (fused[:,0]-Hc)
        g['drug_hyperedge'].x = Hd + torch.sigmoid(self.gammas[1]) * (fused[:,1]-Hd)
        g['proc_hyperedge'].x = Hp + torch.sigmoid(self.gammas[2]) * (fused[:,2]-Hp)

        # 2️⃣ Type -> Time 多头注意力
        time_x = g['time_hyperedge'].x
        num_time = g['time_hyperedge'].num_nodes
        visit_times = getattr(g['time_hyperedge'], 'rel_adm_time', torch.zeros(num_time, device=time_x.device))

        agg_all = torch.zeros_like(time_x)
        type_map = {
            'cond_hyperedge': g['cond_hyperedge'].x,
            'drug_hyperedge': g['drug_hyperedge'].x,
            'proc_hyperedge': g['proc_hyperedge'].x
        }

        for node_type, node_x in type_map.items():
            edge_key = (node_type, 'attend_time', 'time_hyperedge')
            if edge_key not in g.edge_types:
                continue
            edge_index = g[edge_key].edge_index  # [2,E]
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index

            # multi-head projections
            q = self.q_proj(time_x[dst]).view(-1, self.num_heads, self.head_dim)
            k = self.k_proj(node_x[src]).view(-1, self.num_heads, self.head_dim)
            v = self.v_proj(node_x[src]).view(-1, self.num_heads, self.head_dim)

            # attention
            score = (q * k).sum(-1) / (self.head_dim ** 0.5)
            score = self.leakyrelu(score)

            attn = scatter_softmax(score, dst, dim=0).unsqueeze(-1)
            attn = self.dropout(attn)

            agg_heads = scatter_add(attn * v, dst, dim=0, dim_size=num_time)
            agg_all += agg_heads.reshape(num_time, self.dim)

        # residual + output projection
        g['time_hyperedge'].x = time_x + agg_all
        return g


# -------------------------------------------#
# Hypergraph Model #
# -------------------------------------------#
class HypergraphLayer(nn.Module):

    def __init__(self, input_dim, out_features, time_dim, num_heads, gamma_time, tau_time):
        super().__init__()
        self.event2time = Node2TimeAggregator(hidden_dim=input_dim, time_dim=time_dim, mode='attention', num_heads=num_heads, 
                                                         gamma_time=gamma_time, tau_time=tau_time, att_dropout=0.5)
        self.time2event = Time2NodeAggregator(hidden_dim=input_dim, time_dim=time_dim, mode='attention', num_heads=num_heads, 
                                                         gamma_time=gamma_time, tau_time=tau_time, att_dropout=0.5)
        self.time2time = Time2TimeAggregator(hidden_dim=input_dim, num_heads=num_heads, time_decay_init=365.0, dropout=0.5)
        self.time2event_updated = Time2NodeAggregatorImproved(hidden_dim=input_dim, num_heads=num_heads, step_size=0.5, lambda2=0.1, dropout=0.5)
        self.type2time = Type2TimeAggregator(input_dim, num_heads=num_heads, dropout=0.5, gamma_time=gamma_time, tau_time=tau_time)



    def forward(self, g):
        # 再进行时间超边消息传递（更动态）
        g = self.event2time(g)
        g = self.time2time(g)
        g = self.time2event(g)
        return g


class HypergraphModel(nn.Module):
    def __init__(self, tokenizer, hidden_dim, time_dim, output_dim, num_heads, num_layers, gamma_time, tau_time, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = HypergraphEncoder(tokenizer, hidden_dim, time_dim, device=self.device)
        self.patient_encoder = PatientSequenceEncoder(tokenizer, hidden_dim, output_dim, num_heads, num_layers, device=self.device)
        self.layer = HypergraphLayer(hidden_dim, output_dim, time_dim, num_heads, gamma_time, tau_time)
        self.to(self.device)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.fc_graph = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.node_projs = nn.ModuleDict({
            'cond': Linear(-1, hidden_dim),
            'proc': Linear(-1, hidden_dim),
            'drug': Linear(-1, hidden_dim),
            'time_hyperedge': Linear(-1, hidden_dim),
            'cond_hyperedge': Linear(-1, hidden_dim),
            'proc_hyperedge': Linear(-1, hidden_dim),
            'drug_hyperedge': Linear(-1, hidden_dim),
        })
        self.out_lin = nn.Linear(hidden_dim*3, hidden_dim)
        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        self.writer = None
        self.global_step = 0
        
    def set_writer(self, writer):
        self.writer = writer

    def build_hypergraph_batch(self, graph_list, patient_emb):
        """
        graph_list: batch_size 个 offline graph 字典
        """
        h_list = []
        for i, gdict in enumerate(graph_list):
            g = self.encoder(gdict, patient_emb[i])  # 只做 embedding lookup
            h_list.append(g)
        # 合并 batch
        return Batch.from_data_list(h_list)


    def forward(self, batchdata):
  
        seqdata, graph_list = batchdata
        seq_logits, patient_emb = self.patient_encoder(seqdata)  # [B, D]

        patient_emb = self.out_lin(patient_emb)  # [B, hidden_dim]


        g_batch = self.build_hypergraph_batch(graph_list, patient_emb)

        for node_type in g_batch.node_types:
            g_batch[node_type].x = self.node_projs[node_type](g_batch[node_type].x)

        for layer in range(self.num_layers):
            g_batch = self.layer(g_batch)
  
        time_embeds = g_batch['time_hyperedge'].x  # [num_time_hyperedges, hidden_dim]
        batch_index = g_batch['time_hyperedge'].batch  # [num_time_hyperedges]
        time_adm = g_batch['time_hyperedge'].adm_time
  
        _, last_idx = scatter_max(time_adm, batch_index, dim=0)

        last_time_embeds = time_embeds[last_idx]  # [num_patients, hidden_dim]
        # --- 8. 预测 ---
        graph_out = self.fc_graph(last_time_embeds)  # [B, output_dim]
        # --- 9. 融合 ---
        fused_out = self.alpha * graph_out + (1 - self.alpha) * seq_logits
        # fused_out = graph_out

        if self.writer is not None and self.training:
            self.writer.add_scalar('Parameters/alpha', self.alpha.item(), self.global_step)
            self.global_step += 1
        # fused_out = seq_logits
        return fused_out, last_time_embeds