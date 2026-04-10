######################################
# 2026-2-2 #
# Main #
######################################
import argparse
import os
import random

import numpy as np
import torch

from core.model_factory import build_model
from core.task_spec import TaskSpec
from core.trainer_adapter import pick_trainer

# Supported baselines:
# - task=drug: Transformer, RETAIN, StageNet, Deepr, AdaCare, ConCare, GRASP,
#              DrAgent, SafeDrug, GAMENet, MICRON, MoleRec, TRANS
# - task=diag: Transformer, RETAIN, StageNet, KAME, TRANS

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="diag", choices=["diag", "drug"])
    parser.add_argument("--model", type=str, default="Transformer")
    parser.add_argument("--dataset", type=str, default="mimic3", choices=["mimic3", "mimic4", "ccae"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dev", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devm", type=bool, default=False)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./logs")
    parser.add_argument("--show_progress", type=bool, default=True)

    #TRANS specific args
    parser.add_argument("--pe_dim", type=int, default=0)

    #Hypergraph specific args
    parser.add_argument("--time_dim", type=int, default=32, help="time dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers in graph model")
    parser.add_argument("--gamma_time", type=float, default=0.8,help="initial gamma_time")
    parser.add_argument("--tau_time", type=float, default=4.0, help="initial tau_time in days")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def resolve_data_root(args):
    if args.data_root:
        return args.data_root
    fileroot = {
        "mimic3": r"C:\Users\pc\Desktop\code\baseline\mimic3\hosp",
        "mimic4": r"C:\Users\pc\Desktop\code\baseline\mimic4\hosp",
        "ccae": r"./data/processed_dip.pkl",
    }
    return fileroot[args.dataset]


def main():
    args = parse_args()
    args.data_root = resolve_data_root(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.dev}" if torch.cuda.is_available() else "cpu")

    print(
        f"[main] task={args.task} model={args.model} dataset={args.dataset}",
        flush=True,
    )

    # Step 1: build TaskSpec (dataset/tokenizers/label_key/dataloaders)
    spec = TaskSpec.build(args)

    # Step 2: build model
    model = build_model(args, spec, device)

    # Step 3: select trainer/adapter
    trainer = pick_trainer(args, spec, device)

    # Step 4: trainer.fit(model, spec) / trainer.evaluate(model, spec)
    ckpt_path = trainer.fit(model)
    test_loss, test_metrics = trainer.evaluate(model, ckpt_path)

    print(
        f"[main] label_key={spec.label_key} feature_keys={spec.feature_keys}",
        flush=True,
    )
    print(f"[main] dataloaders ready, device={device}", flush=True)
    print(f"[main] test_loss={test_loss:.4f}", flush=True)
    if test_metrics:
        print(f"[main] test_metrics={test_metrics}", flush=True)


if __name__ == "__main__":
    main()
