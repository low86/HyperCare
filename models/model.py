######################################
# 2025-10-28 #
# Online Hypergraph Generator Version #
######################################

import os
import torch
from core.utils import *
import torch.nn as nn
from torch_geometric.nn import Linear

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
        # Optional guidance bias callable: fn(q, u, data, edge_index) -> bias per-edge
        self.guidance_fn = None

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

    def set_guidance_fn(self, fn):
        """
        Set optional guidance bias function.
        Expected signature: fn(q, u, data, edge_index) -> bias with shape [E, H] or [E, 1] or [E]
        """
        self.guidance_fn = fn

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
                if self.guidance_fn is not None:
                    bias = self.guidance_fn(q, k, data, edge_index)
                    if bias is not None:
                        if bias.dim() == 1:
                            bias = bias.unsqueeze(-1)
                        att_score = att_score + bias
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
    


class TypeHyperedgeInitializer(nn.Module):
    """Initialize one patient-level hyperedge for each clinical event type."""

    def __init__(
        self,
        hidden_dim,
        node_types=("cond", "proc", "drug"),
        use_type_centrality=False,
        population_priors_path=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = tuple(node_types)
        self.use_type_centrality = use_type_centrality

        if use_type_centrality:
            priors = self._load_population_priors(population_priors_path)
            for node_type in self.node_types:
                centrality = priors.get(node_type, {}).get("centrality")
                if not isinstance(centrality, torch.Tensor) or centrality.ndim != 1:
                    raise ValueError(
                        f"Population prior for {node_type} must provide a 1-D "
                        "'centrality' tensor."
                    )
                self.register_buffer(
                    f"{node_type}_centrality", centrality.float().clone()
                )

    @staticmethod
    def _load_population_priors(population_priors_path):
        if not population_priors_path:
            raise ValueError(
                "population_priors_path is required when use_type_centrality=True."
            )
        if not os.path.isfile(population_priors_path):
            raise FileNotFoundError(
                f"Population priors not found: {population_priors_path}"
            )
        try:
            return torch.load(
                population_priors_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            return torch.load(population_priors_path, map_location="cpu")

    def forward(self, data):
        if "type_hyperedge" not in data.node_types:
            return data

        type_x = torch.zeros(
            data["type_hyperedge"].num_nodes,
            self.hidden_dim,
            dtype=data["time_hyperedge"].x.dtype,
            device=data["time_hyperedge"].x.device,
        )

        for node_type in self.node_types:
            edge_type = (node_type, "in_type", "type_hyperedge")
            if node_type not in data.node_types or edge_type not in data.edge_types:
                continue

            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            node_x = data[node_type].x

            if self.use_type_centrality:
                centrality = getattr(self, f"{node_type}_centrality")
                token_ids = data[node_type].token_ids[src]
                if token_ids.numel() and token_ids.max() >= centrality.numel():
                    raise ValueError(
                        f"Centrality vocabulary for {node_type} does not match "
                        "the graph tokenizer."
                    )
                weights = centrality[token_ids].to(node_x.dtype)
                weighted_sum = scatter_add(
                    node_x[src] * weights.unsqueeze(-1),
                    dst,
                    dim=0,
                    dim_size=type_x.size(0),
                )
                normalizer = scatter_add(
                    weights, dst, dim=0, dim_size=type_x.size(0)
                ).clamp_min(torch.finfo(node_x.dtype).eps)
                type_x += weighted_sum / normalizer.unsqueeze(-1)
            else:
                type_x += scatter_mean(
                    node_x[src], dst, dim=0, dim_size=type_x.size(0)
                )

        data["type_hyperedge"].x = type_x
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
    

class Type2TimeAggregator(nn.Module):
    """Aggregate connected type hyperedges into their visit hyperedges."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.5):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        edge_key = ("type_hyperedge", "to_time", "time_hyperedge")
        if edge_key not in data.edge_types:
            return data

        edge_index = data[edge_key].edge_index
        if edge_index.numel() == 0:
            return data

        time_x = data["time_hyperedge"].x
        type_x = data["type_hyperedge"].x
        src, dst = edge_index

        q_all = self.q_proj(time_x).view(-1, self.num_heads, self.head_dim)
        k_all = self.k_proj(type_x).view(-1, self.num_heads, self.head_dim)
        v_all = self.v_proj(type_x).view(-1, self.num_heads, self.head_dim)

        q_edges = q_all[dst]
        k_edges = k_all[src]
        v_edges = v_all[src]
        att_score = (q_edges * k_edges).sum(-1) / (self.head_dim ** 0.5)
        att = scatter_softmax(att_score, dst, dim=0)
        att = self.dropout(att).unsqueeze(-1)

        out_heads = scatter_add(
            att * v_edges, dst, dim=0, dim_size=time_x.size(0)
        )
        out = out_heads.reshape(time_x.size(0), self.hidden_dim)
        out = self.out_proj(out)
        has_type = scatter_add(
            torch.ones_like(dst, dtype=out.dtype),
            dst,
            dim=0,
            dim_size=time_x.size(0),
        ).gt(0).unsqueeze(-1)
        data["time_hyperedge"].x = time_x + out * has_type
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
        self.type2time = Type2TimeAggregator(
            hidden_dim=input_dim, num_heads=num_heads, dropout=0.5
        )

    def forward(self, g):
        g = self.event2time(g)
        g = self.time2time(g)
        g = self.type2time(g)  # Uncomment to enable type-to-time aggregation.
        g = self.time2event(g)
        return g


class HypergraphModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        hidden_dim,
        time_dim,
        output_dim,
        num_heads,
        num_layers,
        gamma_time,
        tau_time,
        device='cuda',
        feature_key_map: dict | None = None,
        node_emb_path: str | None = None,
        note_emb_dim: int = 768,
        use_type_centrality: bool = False,
        population_priors_path: str | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = HypergraphEncoder(
            tokenizer,
            hidden_dim,
            time_dim,
            device=self.device,
            feature_key_map=feature_key_map,
            node_emb_path=node_emb_path,
            note_emb_dim=note_emb_dim,
        )
        self.type_initializer = TypeHyperedgeInitializer(
            hidden_dim,
            node_types=tuple(self.encoder.feature_key_map),
            use_type_centrality=use_type_centrality,
            population_priors_path=population_priors_path,
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
        })

    def build_hypergraph_batch(self, graph_batch):
        """Materialize features once for an already-collated offline batch."""
        if not hasattr(graph_batch, 'node_types'):
            raise TypeError(
                "HyperCare expects a prebatched PyG graph. Delete the old "
                "HyperCare cache so the v2 offline graph can be rebuilt."
            )
        return self.encoder(graph_batch)


    def forward(self, batchdata):
  
        _, graph_batch = batchdata
        g_batch = self.build_hypergraph_batch(graph_batch)

        for node_type, projection in self.node_projs.items():
            if node_type in g_batch.node_types:
                g_batch[node_type].x = projection(g_batch[node_type].x)

        # Type hyperedges are initialized once from their incident occurrences.
        g_batch = self.type_initializer(g_batch)

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
