"""Vectorized online feature materialization for prebatched patient graphs."""

import os

import numpy as np
import torch
import torch.nn as nn

from benchmarks.modules.TSEncoder import Time2Vec


class HypergraphEncoder(nn.Module):
    """Attach trainable features to one already-collated PyG heterogeneous batch."""

    def __init__(
        self,
        tokenizer,
        hidden_dim,
        time_dim,
        device,
        feature_key_map: dict | None = None,
        node_emb_path: str | None = None,
        note_emb_dim: int = 768,
    ):
        super().__init__()
        self.feature_key_map = feature_key_map or {
            "cond": "cond_hist",
            "proc": "procedures",
            "drug": "drugs",
        }
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.device = torch.device(device)

        node_emb_dict = None
        if node_emb_path and os.path.exists(node_emb_path):
            node_emb_dict = np.load(node_emb_path, allow_pickle=True).item()
        elif node_emb_path:
            print(
                f"[HyperCare] node_emb_path not found: {node_emb_path}; "
                "using random code embeddings.",
                flush=True,
            )

        self.node_emb_dim = hidden_dim
        if node_emb_dict:
            sample_value = next(iter(node_emb_dict.values()))
            self.node_emb_dim = int(
                sample_value.numel()
                if isinstance(sample_value, torch.Tensor)
                else np.asarray(sample_value).size
            )

        tokenizers = {
            node_type: tokenizer[feature_key]
            for node_type, feature_key in self.feature_key_map.items()
        }
        self.cond_emb = self._init_semantic_embedding(
            tokenizers["cond"], "CCSCM:", self.node_emb_dim, node_emb_dict
        )
        self.proc_emb = self._init_semantic_embedding(
            tokenizers["proc"], "CCSPROC:", self.node_emb_dim, node_emb_dict
        )
        self.drug_emb = self._init_semantic_embedding(
            tokenizers["drug"], "ATC:", self.node_emb_dim, node_emb_dict
        )
        self.time2vec = Time2Vec(time_dim)
        self.note_emb_dim = note_emb_dim
        self.note_projection = nn.Linear(note_emb_dim, hidden_dim)

    @staticmethod
    def _get_vocab_dict(tokenizer):
        vocabulary = tokenizer.vocabulary
        if isinstance(vocabulary, dict):
            return vocabulary
        for attr in ("token2idx", "_token2idx"):
            if hasattr(vocabulary, attr):
                return getattr(vocabulary, attr)
        if hasattr(vocabulary, "items"):
            try:
                return dict(vocabulary)
            except Exception:
                pass
        if hasattr(tokenizer, "convert_indices_to_tokens"):
            return {
                tokenizer.convert_indices_to_tokens([index])[0]: index
                for index in range(tokenizer.get_vocabulary_size())
            }
        raise AttributeError("Tokenizer vocabulary is not accessible as a dict.")

    def _init_semantic_embedding(self, tokenizer, prefix, emb_dim, node_emb_dict):
        if not node_emb_dict:
            return nn.Embedding(
                tokenizer.get_vocabulary_size(),
                emb_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        vocab = self._get_vocab_dict(tokenizer)
        vocab_size = tokenizer.get_vocabulary_size()
        weight = torch.zeros((vocab_size, emb_dim), dtype=torch.float32)
        filled = torch.zeros(vocab_size, dtype=torch.bool)
        collected = []
        hit = 0

        for token, index in vocab.items():
            vector = node_emb_dict.get(f"{prefix}{token}")
            if vector is None:
                continue
            vector = torch.as_tensor(vector, dtype=torch.float32).reshape(-1)
            if vector.numel() != emb_dim:
                continue
            weight[index] = vector
            filled[index] = True
            collected.append(vector)
            hit += 1

        if collected:
            weight[~filled] = torch.stack(collected, dim=0).mean(dim=0)
        else:
            nn.init.normal_(weight)

        padding_index = tokenizer.get_padding_index()
        if padding_index is not None and padding_index < vocab_size:
            weight[padding_index].zero_()

        print(f"[semantic_init] {prefix} hits {hit}/{vocab_size}")
        return nn.Embedding.from_pretrained(
            weight,
            freeze=False,
            padding_idx=padding_index,
        )

    @staticmethod
    def _visit_index_for_occurrences(data, node_type, device):
        edge_index = data[node_type, "in_time", "time_hyperedge"].edge_index
        source, destination = edge_index
        node_count = data[node_type].num_nodes
        if source.numel() != node_count:
            raise ValueError(
                f"Every {node_type} occurrence must belong to exactly one visit."
            )
        visit_index = torch.empty(node_count, dtype=torch.long, device=device)
        if node_count:
            visit_index[source] = destination
        return visit_index

    def forward(self, data):
        if not hasattr(data, "node_types"):
            raise TypeError(
                "HypergraphEncoder expects a prebatched HeteroData/Batch. "
                "Delete the old HyperCare cache so it can be rebuilt."
            )

        data = data.to(self.device, non_blocking=True)
        time_store = data["time_hyperedge"]
        visit_times = time_store.rel_adm_time.float()
        time_store.adm_time = visit_times

        note_emb = time_store.note_emb.float()
        if note_emb.ndim != 2 or note_emb.size(1) != self.note_emb_dim:
            raise ValueError(
                f"Expected note embeddings [N, {self.note_emb_dim}], "
                f"got {tuple(note_emb.shape)}."
            )

        # Time2Vec stays in the autograd graph; only raw relative times are cached.
        time_emb = self.time2vec(visit_times)
        time_store.x = torch.cat(
            [time_emb, self.note_projection(note_emb)], dim=-1
        )

        embeddings = {
            "cond": self.cond_emb,
            "proc": self.proc_emb,
            "drug": self.drug_emb,
        }
        for node_type, embedding in embeddings.items():
            node_store = data[node_type]
            if node_store.num_nodes == 0:
                node_store.x = torch.empty(
                    (0, self.node_emb_dim + self.time_dim),
                    dtype=time_emb.dtype,
                    device=self.device,
                )
                continue
            visit_index = self._visit_index_for_occurrences(
                data, node_type, self.device
            )
            token_features = embedding(node_store.token_ids.long())
            node_store.x = torch.cat(
                [token_features, time_emb[visit_index]], dim=-1
            )

        return data
