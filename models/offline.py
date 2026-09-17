"""Offline construction of immutable, CPU-resident patient hypergraphs.

The expensive Python work (tokenization, incidence construction and note
alignment) happens once when the dataset cache is built. Training batches
these ``HeteroData`` objects directly instead of rebuilding one graph per
patient on every forward pass.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from core.utils import convert_to_relative_time


class Hypergraph:
    """Build the static part of every patient graph on CPU."""

    def __init__(
        self,
        sequence_dataset,
        tokenizers,
        note_emb_dict,
        cache_dir: str = "./graph/cache",
        device="cpu",
        feature_key_map: dict | None = None,
        time_dim: int | None = None,
        note_emb_dim: int | None = None,
    ):
        del cache_dir, device, time_dim  # Kept for the public constructor API.
        self.sequence_dataset = sequence_dataset
        self.tokenizers = tokenizers
        self.note_emb_dict = note_emb_dict
        self.feature_key_map = feature_key_map or {
            "cond": "cond_hist",
            "proc": "procedures",
            "drug": "drugs",
        }
        self.note_emb_dim = note_emb_dim or self._infer_note_emb_dim(note_emb_dict) or 768

    @staticmethod
    def _infer_note_emb_dim(note_emb_dict):
        for value in note_emb_dict.values():
            if isinstance(value, torch.Tensor):
                return int(value.numel())
            array = np.asarray(value)
            if array.size:
                return int(array.size)
        return None

    @staticmethod
    def _occurrence_time_incidence(tokens_by_visit):
        """Flatten occurrence token ids and map every occurrence to one visit."""
        token_ids = []
        sources = []
        destinations = []
        offset = 0
        for visit_index, visit_tokens in enumerate(tokens_by_visit):
            count = len(visit_tokens)
            token_ids.extend(visit_tokens)
            sources.extend(range(offset, offset + count))
            destinations.extend([visit_index] * count)
            offset += count

        edge_index = (
            torch.tensor([sources, destinations], dtype=torch.long)
            if sources
            else torch.empty((2, 0), dtype=torch.long)
        )
        return torch.tensor(token_ids, dtype=torch.long), edge_index

    @staticmethod
    def _time_connectivity(num_visits, window_size=4):
        """Return causal visit edges: self plus at most ``window_size`` history."""
        if num_visits <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        sources = []
        destinations = []
        for current in range(num_visits):
            history = range(max(0, current - window_size), current + 1)
            sources.extend(history)
            destinations.extend([current] * len(history))
        return torch.tensor([sources, destinations], dtype=torch.long)

    def build_all_in_one(self):
        return [
            self._build_single(self.sequence_dataset[index])
            for index in tqdm(
                range(len(self.sequence_dataset)),
                desc="Prebuilding HyperCare graphs",
            )
        ]

    def _build_single(self, sample):
        encoded = {}
        for node_type, feature_key in self.feature_key_map.items():
            encoded[node_type] = self.tokenizers[feature_key].batch_encode_2d(
                sample[feature_key], padding=False
            )

        num_visits = len(encoded["proc"])
        if any(len(visits) != num_visits for visits in encoded.values()):
            raise ValueError("Every modality must contain the same number of visits.")

        visit_ids = sample["visit_history"]
        if len(visit_ids) != num_visits:
            raise ValueError("visit_history must align with the visit sequence.")
        if len(sample["adm_time"]) != num_visits:
            raise ValueError("adm_time must align with the visit sequence.")

        note_rows = []
        for visit_id in visit_ids:
            value = self.note_emb_dict.get(int(visit_id))
            if value is None:
                value = np.zeros(self.note_emb_dim, dtype=np.float32)
            row = np.asarray(value, dtype=np.float32).reshape(-1)
            if row.size != self.note_emb_dim:
                raise ValueError(
                    f"Note embedding for visit {visit_id} has dimension {row.size}; "
                    f"expected {self.note_emb_dim}."
                )
            note_rows.append(row)

        note_emb = torch.from_numpy(np.stack(note_rows, axis=0))
        rel_time = torch.tensor(
            convert_to_relative_time(sample["adm_time"]), dtype=torch.float32
        )

        data = HeteroData()
        data["time_hyperedge"].num_nodes = num_visits
        data["time_hyperedge"].rel_adm_time = rel_time
        data["time_hyperedge"].note_emb = note_emb

        for node_type, visits in encoded.items():
            token_ids, edge_index = self._occurrence_time_incidence(visits)
            data[node_type].num_nodes = int(token_ids.numel())
            data[node_type].token_ids = token_ids
            data[node_type, "in_time", "time_hyperedge"].edge_index = edge_index

        data[
            "time_hyperedge", "connect", "time_hyperedge"
        ].edge_index = self._time_connectivity(num_visits)

        return {"patient_id": sample["patient_id"], "hetero_data": data}


class MMDataset(Dataset):
    """Pair sequence labels with prebuilt static patient graphs."""

    CACHE_SCHEMA_VERSION = 2

    def __init__(
        self,
        dataset,
        tokenizer,
        note_emb_dict,
        device="cpu",
        feature_key_map: dict | None = None,
        time_dim: int | None = None,
        note_emb_dim: int | None = None,
    ):
        self.sequence_dataset = dataset.samples
        self.cache_schema_version = self.CACHE_SCHEMA_VERSION
        builder = Hypergraph(
            self.sequence_dataset,
            tokenizer,
            note_emb_dict,
            device=device,
            feature_key_map=feature_key_map,
            time_dim=time_dim,
            note_emb_dim=note_emb_dim,
        )
        self.graph_data = builder.build_all_in_one()

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return self.sequence_dataset[idx], self.graph_data[idx]
