import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from benchmarks.modules.TSEncoder import AddGlobalLaplacianPE, AddMetaPathRandomWalkSE
from core.utils import convert_to_relative_time


class PatientGraph(torch.utils.data.Dataset):
    def __init__(self, tokenizer, subset, dim, device, trans_dim=0, di=False, feature_key_map=None):
        if feature_key_map is None:
            feature_key_map = {"co": "cond_hist", "pr": "procedures", "dh": "drugs"}
        self.feature_key_map = feature_key_map
        self.c_tokenizer = tokenizer[self.feature_key_map["co"]]
        self.d_tokenizer = tokenizer[self.feature_key_map["dh"]]
        self.p_tokenizer = tokenizer[self.feature_key_map["pr"]]
        self.dataset = subset
        self.di_edge = di
        self.dim = dim
        self.se = False
        if trans_dim != 0:
            self.se = True
            self.global_pe_transform = AddGlobalLaplacianPE(k=trans_dim, device=device)
            self.local_se_transform = AddMetaPathRandomWalkSE(trans_dim, device=device)

        self.all_data = self._process_()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def _build_edge_index(self, token_lists):
        src = []
        dst = []
        for visit_idx, tokens in enumerate(token_lists):
            for t in tokens:
                src.append(t)
                dst.append(visit_idx)
        if len(src) == 0:
            return torch.empty((2, 0), dtype=torch.int64)
        return torch.tensor([src, dst], dtype=torch.int64)

    def _build_edge_time(self, token_lists):
        times = []
        for visit_idx, tokens in enumerate(token_lists):
            times.extend([visit_idx] * len(tokens))
        if len(times) == 0:
            return torch.empty((0,), dtype=torch.float32)
        return torch.tensor(times, dtype=torch.float32)

    def _process_(self):
        all_data = []
        print("Constructing graph...")
        for dp in tqdm(range(len(self.dataset))):
            data = HeteroData()
            num_visit = len(self.dataset[dp]["procedures"])

            data["visit"].x = torch.zeros(num_visit, self.dim)
            cond_key = self.feature_key_map["co"]
            proc_key = self.feature_key_map["pr"]
            drug_key = self.feature_key_map["dh"]

            dpc = self.c_tokenizer.batch_encode_2d(
                self.dataset[dp][cond_key], padding=False
            )
            dpp = self.p_tokenizer.batch_encode_2d(
                self.dataset[dp][proc_key], padding=False
            )
            dpd = self.d_tokenizer.batch_encode_2d(
                self.dataset[dp][drug_key], padding=False
            )

            data["visit"].time = convert_to_relative_time(self.dataset[dp]["adm_time"])

            data["co"].x = torch.zeros(self.c_tokenizer.get_vocabulary_size(), self.dim)
            data["pr"].x = torch.zeros(self.p_tokenizer.get_vocabulary_size(), self.dim)
            data["dh"].x = torch.zeros(self.d_tokenizer.get_vocabulary_size(), self.dim)

            civ = self._build_edge_index(dpc)
            piv = self._build_edge_index(dpp)
            div = self._build_edge_index(dpd)
            data["co", "in", "visit"].edge_index = civ
            data["pr", "in", "visit"].edge_index = piv
            data["dh", "in", "visit"].edge_index = div

            if num_visit > 1:
                viv = torch.tensor(
                    [[i for i in range(num_visit - 1)], [i + 1 for i in range(num_visit - 1)]],
                    dtype=torch.int64,
                )
            else:
                viv = torch.empty((2, 0), dtype=torch.int64)

            data["co", "in", "visit"].edge_time = self._build_edge_time(dpc)
            data["pr", "in", "visit"].edge_time = self._build_edge_time(dpp)
            data["dh", "in", "visit"].edge_time = self._build_edge_time(dpd)

            data["visit", "has", "co"].edge_index = torch.flip(civ, [0])
            data["visit", "has", "pr"].edge_index = torch.flip(piv, [0])
            data["visit", "has", "dh"].edge_index = torch.flip(div, [0])

            if self.di_edge:
                data["visit", "connect", "visit"].edge_index = viv
            else:
                if viv.numel() == 0:
                    data["visit", "connect", "visit"].edge_index = viv
                else:
                    data["visit", "connect", "visit"].edge_index = torch.cat(
                        [viv, torch.flip(viv, [0])], dim=1
                    )

            if self.se:
                data = self.global_pe_transform.apply_laplacian_pe(data)
                f_metapaths = [
                    [
                        ("co", "in", "visit"),
                        ("visit", "has", "pr"),
                        ("pr", "in", "visit"),
                        ("visit", "has", "co"),
                    ],
                    [
                        ("pr", "in", "visit"),
                        ("visit", "has", "co"),
                        ("co", "in", "visit"),
                        ("visit", "has", "pr"),
                    ],
                    [("dh", "in", "visit"), ("visit", "has", "dh")],
                ]
                data = self.local_se_transform.forward(data=data, metapaths=f_metapaths)
                for node_type in data.node_types:
                    if node_type not in ["co", "pr", "dh"]:
                        if "laplacian_pe" in data[node_type]:
                            del data[node_type].laplacian_pe
                        if "random_walk_se" in data[node_type]:
                            del data[node_type].random_walk_se

            all_data.append(data)
        return all_data


class MMDataset(Dataset):
    def __init__(self, dataset, tokenizer, dim, device, trans_dim=0, di=False, feature_key_map=None):
        self.sequence_dataset = dataset.samples
        self.tokenizer = tokenizer
        self.trans_dim = trans_dim
        self.di = di
        self.dim = dim
        self.device = device
        self.graph_data = PatientGraph(
            self.tokenizer,
            self.sequence_dataset,
            dim=self.dim,
            device=self.device,
            trans_dim=self.trans_dim,
            di=self.di,
            feature_key_map=feature_key_map,
        ).all_data

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        sequence_data = self.sequence_dataset[idx]
        graph_data = self.graph_data[idx]
        return sequence_data, graph_data
