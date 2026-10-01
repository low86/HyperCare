######################################
# 2026-2-2 #
# Diagnosis baselines #
######################################
# NOTE:
# We do not use the official pyhealth baseline implementations here because some samples
# contain empty feature lists (e.g., empty codes per visit), which can cause failures or
# misaligned batching in the default pipeline. These local baselines are designed to
# handle empty inputs robustly and keep the data flow consistent for our diagnosis tasks.
import math
import torch
import torch.nn as nn
from benchmarks.modules.Seqmodels import *
from benchmarks.modules.TSEncoder import *
from benchmarks.modules.TRANS import TRANS, graph_meta
from models.model import HypergraphModel

# -------------------------------------------#
# Transformer #
# -------------------------------------------#
class Transformer(nn.Module):
    def __init__(
        self,
        Tokenizers,
        output_size,
        device,
        embedding_dim = 128,
        dropout = 0.5
    ):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(heads=2,
                feature_size=embedding_dim, dropout = dropout,num_layers=2
            )
        
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata) :
        patient_emb = []
        for feature_key in self.feature_keys:

            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
                # (patient, visit, event)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x !=0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        return logits

# -------------------------------------------#
# RETAIN #
# -------------------------------------------#    
class RETAIN(nn.Module):
    def __init__(self, Tokenizers, output_size, device,
        embedding_dim: int = 128, dropout = 0.5
        ):
        super(RETAIN, self).__init__()
        self.embedding_dim = embedding_dim
        Tokenizers =  {k: Tokenizers[k] for k in list(Tokenizers)[1:]}
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()

        # add feature RETAIN layers
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
        self.retain = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, dropout = dropout)

        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.device = device


    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata):

        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.sum(x, dim=2) != 0
            x = self.retain[feature_key](x, mask)
            patient_emb.append(x)
        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits

# -------------------------------------------#
# StageNet #
# -------------------------------------------#
class StageNet(nn.Module):
    def __init__(self, Tokenizers, output_size, device, embedding_dim: int = 128,
        chunk_size: int = 128,
        levels: int = 3,
    ):
        super(StageNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.levels = levels
        Tokenizers =  {k: Tokenizers[k] for k in list(Tokenizers)[1:]}
        self.feature_keys = Tokenizers.keys()

        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()

        self.stagenet = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
            self.stagenet[feature_key] = StageNetLayer(
                input_dim=embedding_dim,
                chunk_size=self.chunk_size,
                levels=self.levels,
            )
        self.fc = nn.Linear(
            len(self.feature_keys) * self.chunk_size * self.levels, output_size
        )
        self.device = device

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata):
        patient_emb = []
        distance = []
        mask_dict = {}
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
 
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x !=0, dim=2)
            mask_dict[feature_key] = mask
            time = None
            x, _, cur_dis = self.stagenet[feature_key](x, time=time, mask=mask)
            patient_emb.append(x)
            distance.append(cur_dis)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits

# =========================
# model factory
# =========================
def build_model(model_name, task_dataset, tokenizers, device, args):
    label_size = len(task_dataset.get_all_tokens("conditions"))

    if model_name == "HyperCare":
        return HypergraphModel(
            tokenizers,
            hidden_dim=args.hidden_dim,
            time_dim=args.time_dim,
            output_dim=label_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            gamma_time=args.gamma_time,
            tau_time=args.tau_time,
            device=device,
            feature_key_map={"cond": "cond_hist", "proc": "procedures", "drug": "drugs"},
            node_emb_path=getattr(args, "node_emb_path", ""),
            note_emb_dim=getattr(args, "note_emb_dim", 768),
            use_type_centrality=getattr(args, "use_type_centrality", False),
            population_priors_path=getattr(args, "population_priors_path", ""),
        )

    if model_name == "Transformer":
        return Transformer(tokenizers, label_size, device)
    if model_name == "RETAIN":
        return RETAIN(tokenizers, label_size, device)
    if model_name == "StageNet":
        return StageNet(tokenizers, label_size, device)
    if model_name == "TRANS":
        return TRANS(
            tokenizers,
            128,
            label_size,
            device,
            graph_meta=graph_meta,
            pe=args.pe_dim,
            feature_key_map={"co": "cond_hist", "pr": "procedures", "dh": "drugs"},
        )

    raise ValueError(f"Unsupported model: {model_name}")
