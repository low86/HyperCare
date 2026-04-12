######################################
# 2025-10-28 #
# Online Hypergraph Generator Version #
######################################

import torch
from core.utils import *
import torch.nn as nn
from torch_geometric.nn import Linear

from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter_mean, scatter_softmax, scatter_sum, scatter_max
from models.online_encoder import HypergraphEncoder



# -------------------------------------------#
# Node to time hyperedge
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
            # projection matrices
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

        # last visit time per batch
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
    Time -> node feedback using attention or simple aggregation.
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
            # residual update
            data[node_type].x = node_x + agg
        return data


class Time2TimeAggregator(nn.Module):
    """
    Local time attention based on edge_index (past-only window).
    Uses edges (src -> dst) to represent influence from past visits to current visit.
    Complexity: O(E).
    """
    def __init__(self, hidden_dim, num_heads=4, time_decay_init=365.0, dropout=0.5):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # query from dst time node
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)  # key from src time node
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.time_decay = nn.Parameter(torch.tensor(time_decay_init))
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        # optional residual scale
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        # return early if no time-to-time edges
        edge_key = ('time_hyperedge', 'connect', 'time_hyperedge')
        if edge_key not in data.edge_types:
            return data

        H = data['time_hyperedge'].x                     # [N_time, D]
        times = data['time_hyperedge'].adm_time.float()   # [N_time]
        edge_index = data[edge_key].edge_index.to(H.device)  # [2, E]
        src, dst = edge_index[0], edge_index[1]           # src->dst (past->current)

        # project and split heads
        Q_all = self.q_proj(H).view(-1, self.num_heads, self.head_dim)  # [N_time, H, dh]
        K_all = self.k_proj(H).view(-1, self.num_heads, self.head_dim)
        V_all = self.v_proj(H).view(-1, self.num_heads, self.head_dim)

        q_edges = Q_all[dst]
        k_edges = K_all[src]
        v_edges = V_all[src]

        att_score = (q_edges * k_edges).sum(-1) / (self.head_dim ** 0.5)
        # time decay weight per-edge: exp(-|delta_t| / tau)
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
    


# -------------------------------------------#
# Hypergraph Model
# -------------------------------------------#
class HypergraphLayer(nn.Module):

    def __init__(self, input_dim, out_features, time_dim, num_heads, gamma_time, tau_time):
        super().__init__()
        self.event2time = Node2TimeAggregator(hidden_dim=input_dim, time_dim=time_dim, mode='attention', num_heads=num_heads, \
                                                         gamma_time=gamma_time, tau_time=tau_time, att_dropout=0.5)
        self.time2event = Time2NodeAggregator(hidden_dim=input_dim, time_dim=time_dim, mode='attention', num_heads=num_heads, \
                                                         gamma_time=gamma_time, tau_time=tau_time, att_dropout=0.5)
        self.time2time = Time2TimeAggregator(hidden_dim=input_dim, num_heads=num_heads, time_decay_init=365.0, dropout=0.5)

    def forward(self, g):
        g = self.event2time(g)
        g = self.time2time(g)
        g = self.time2event(g)
        return g


class HypergraphModel(nn.Module):
    def __init__(self, tokenizer, hidden_dim, time_dim, output_dim, num_heads, num_layers, gamma_time, tau_time, device='cuda', feature_key_map: dict | None = None, node_emb_path: str | None = None):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = HypergraphEncoder(
            tokenizer,
            hidden_dim,
            time_dim,
            device=self.device,
            feature_key_map=feature_key_map,
            node_emb_path=node_emb_path,
        )
        self.layer = HypergraphLayer(hidden_dim, output_dim, time_dim, num_heads, gamma_time, tau_time)
        self.to(self.device)
        self.fc = nn.Linear(hidden_dim, output_dim)
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

    def build_hypergraph_batch(self, graph_list):
        """
        graph_list: list of offline graphs for the batch
        """
        h_list = []
        for i, gdict in enumerate(graph_list):
            g = self.encoder(gdict)  # embedding lookup
            h_list.append(g)
       
        return Batch.from_data_list(h_list)


    def forward(self, batchdata):
  
        _, graph_list = batchdata
        g_batch = self.build_hypergraph_batch(graph_list)

        for node_type in g_batch.node_types:
            g_batch[node_type].x = self.node_projs[node_type](g_batch[node_type].x)

        for layer in range(self.num_layers):
            g_batch = self.layer(g_batch)
  
        time_embeds = g_batch['time_hyperedge'].x  # [num_time_hyperedges, hidden_dim]
        batch_index = g_batch['time_hyperedge'].batch  # [num_time_hyperedges]
        time_adm = g_batch['time_hyperedge'].adm_time
  
        _, last_idx = scatter_max(time_adm, batch_index, dim=0)

        last_time_embeds = time_embeds[last_idx]  # [num_patients, hidden_dim]
        # --- Prediction ---
        graph_out = self.fc(last_time_embeds)  # [B, output_dim]

        return graph_out
