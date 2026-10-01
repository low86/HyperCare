######################################
# 2026-2-2 #
# Model Task Spec #
######################################
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from joblib import dump, load

from pyhealth.tokenizer import Tokenizer

from core.registry import TASK_SPECS
from benchmarks.modules.trans_dataset import MMDataset as TransMMDataset
from models.offline import MMDataset as HyperCareDataset
from core.utils import (
    get_init_tokenizers,
    get_label_tokenizer,
    load_dataset,
    mm_dataloader,
    seq_dataloader,
    split_dataset,
)


@dataclass
class TaskSpec:
    task: str
    dataset: str
    label_key: str
    feature_keys: List[str]
    task_dataset: object
    tokenizers: Dict[str, Tokenizer]
    label_tokenizer: Tokenizer
    dataloaders: Tuple[object, object, object]

    @classmethod
    def build(cls, args):
        if args.task not in TASK_SPECS:
            raise ValueError(f"Unknown task: {args.task}")
        spec = TASK_SPECS[args.task]

        if args.dataset not in spec["task_fn"]:
            raise ValueError(f"Task {args.task} does not support dataset {args.dataset}")

        task_dataset = load_dataset(
            args.dataset,
            root=args.data_root,
            task_fn=spec["task_fn"][args.dataset],
            dev=args.devm,
        )

        tokenizers = get_init_tokenizers(task_dataset, keys=spec["feature_keys"])
        label_tokenizer = get_label_tokenizer(
            task_dataset.get_all_tokens(spec["label_key"])
        )

        if args.model == "HyperCare":
            if args.task == "diag":
                feature_key_map = {"cond": "cond_hist", "proc": "procedures", "drug": "drugs"}
            elif args.task == "drug":
                feature_key_map = {"cond": "conditions", "proc": "procedures", "drug": "drugs_hist"}
            else:
                raise ValueError(f"HyperCare does not support task: {args.task}")

            os.makedirs(args.output_dir, exist_ok=True)
            note_cache_tag = "note"
            if not args.note_emb_path or not os.path.exists(args.note_emb_path):
                note_cache_tag = "nonote"
            cache_path = os.path.join(
                args.output_dir,
                f"{args.task}_{args.dataset}_hypercare_t{args.time_dim}_n{args.note_emb_dim}_{note_cache_tag}.pt",
            )
            if os.path.exists(cache_path):
                mdataset = load(cache_path)
            else:
                if args.note_emb_path and os.path.exists(args.note_emb_path):
                    note_emb_dict = np.load(args.note_emb_path, allow_pickle=True).item()
                else:
                    note_emb_dict = {}
                    print(
                        "[HyperCare] note_emb_path not found or not provided; "
                        "using zero note embeddings.",
                        flush=True,
                    )
                mdataset = HyperCareDataset(
                    task_dataset,
                    tokenizers,
                    note_emb_dict,
                    device="cpu",
                    feature_key_map=feature_key_map,
                    time_dim=args.time_dim,
                    note_emb_dim=args.note_emb_dim,
                )
                dump(mdataset, cache_path)
            trainset, validset, testset = split_dataset(mdataset)
            dataloaders = mm_dataloader(
                trainset, validset, testset, batch_size=args.batch_size
            )
        elif args.model == "TRANS":
            if "procedures" not in spec["feature_keys"]:
                raise ValueError("TRANS requires procedures feature in task spec.")
            if "cond_hist" in spec["feature_keys"]:
                cond_key = "cond_hist"
            elif "conditions" in spec["feature_keys"]:
                cond_key = "conditions"
            else:
                raise ValueError("TRANS requires cond_hist or conditions feature.")

            if "drugs" in spec["feature_keys"]:
                drug_key = "drugs"
            elif "drugs_hist" in spec["feature_keys"]:
                drug_key = "drugs_hist"
            else:
                raise ValueError("TRANS requires drugs or drugs_hist feature.")

            feature_key_map = {"co": cond_key, "pr": "procedures", "dh": drug_key}
            os.makedirs(args.output_dir, exist_ok=True)
            cache_path = os.path.join(
                args.output_dir, f"{args.task}_{args.dataset}_trans_{args.pe_dim}.pkl"
            )
            if os.path.exists(cache_path):
                mdataset = load(cache_path)
            else:
                mdataset = TransMMDataset(
                    task_dataset,
                    tokenizers,
                    dim=128,
                    device="cpu",
                    trans_dim=args.pe_dim,
                    feature_key_map=feature_key_map,
                )
                dump(mdataset, cache_path)
            trainset, validset, testset = split_dataset(mdataset)
            dataloaders = mm_dataloader(
                trainset, validset, testset, batch_size=args.batch_size
            )
        else:
            dataloaders = seq_dataloader(task_dataset, batch_size=args.batch_size)

        return cls(
            task=args.task,
            dataset=args.dataset,
            label_key=spec["label_key"],
            feature_keys=spec["feature_keys"],
            task_dataset=task_dataset,
            tokenizers=tokenizers,
            label_tokenizer=label_tokenizer,
            dataloaders=dataloaders,
        )
