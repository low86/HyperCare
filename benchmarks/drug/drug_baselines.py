######################################
# 2026-2-2 #
# Medication baselines #
######################################
import os
from joblib import dump, load
from models.hypercare.offline import MMDataset
from core.utils import *
from benchmarks.Task import *
from benchmarks.modules.TRANS import *
from pyhealth.models import (
    SafeDrug,
    GAMENet,
    Deepr,
    AdaCare,
    ConCare,
    GRASP,
    MICRON,
    MoleRec,
    Agent,
    RETAIN,
    StageNet,
    Transformer
)
from models.hypercare.model import HypergraphModel

# =========================
# model factory
# =========================
def build_model(model_name, task_dataset, tokenizers, device, args):
    label_size = len(task_dataset.get_all_tokens("drugs"))

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
            feature_key_map={"cond": "conditions", "proc": "procedures", "drug": "drugs_hist"},
        )

    if model_name == "Transformer":
        return Transformer(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures", "drugs_hist"],
            label_key="drugs",
            mode="multilabel",
        )

    elif model_name == "RETAIN":
        return RETAIN(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )

    elif model_name == "StageNet":
        return StageNet(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )

    elif model_name == "Deepr":
        return Deepr(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures", "drugs_hist"],
            label_key="drugs",
            mode="multilabel",
        )

    elif model_name == "AdaCare":
        return AdaCare(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            use_embedding=[True, True],
            mode="multilabel",
        )

    elif model_name == "ConCare":
        return ConCare(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
            use_embedding=[True, True],
        )

    elif model_name == "GRASP":
        return GRASP(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            use_embedding=[True, True],
            mode="multilabel",
        )
    
    elif model_name == "DrAgent":
        return Agent(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
    
    elif model_name == "SafeDrug":
        return SafeDrug(
            dataset=task_dataset,
        )

    elif model_name == "GAMENet":
        return GAMENet(
            dataset=task_dataset,
        )

    elif model_name == "MICRON":
        return MICRON(
            dataset=task_dataset,
        )

    elif model_name == "MoleRec":
        return MoleRec(
            dataset=task_dataset,
        )


    elif model_name == "TRANS":
        model = TRANS(
            tokenizers,
            128,
            label_size,
            device,
            graph_meta=graph_meta,
            pe=args.pe_dim,
            feature_key_map={"co": "conditions", "pr": "procedures", "dh": "drugs_hist"},
        )
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")
