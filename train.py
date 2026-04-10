from html import parser
import os
from tqdm import *
import random
import argparse
import numpy as np
from joblib import dump, load
import torch
import torch.optim as optim
from utils import *
from benchmarks.Task import *
from graph.GraphBuilder import *
from benchmarks.diag.Model import *
from benchmarks.diag.diag_baselines import *
from multiprocessing import freeze_support
from models.Hypergraph import HypergraphModel
from benchmarks.metrics import compute_metrics
from pyhealth.metrics import multilabel_metrics_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

NOTE_EMB_PATH = "./logs/note_emb_dict.npy"
note_emb_dict = np.load(NOTE_EMB_PATH, allow_pickle=True).item()

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--model', type=str, default="Hypergraph", help='Transformer, RETAIN, StageNet, KAME, TRANS, Hypergraph')
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4', 'ccae'])
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--pe_dim', type=int, default=0, help='dimensions of spatial encoding')
    parser.add_argument('--devm', type=bool, default=False, help='develop mode')
    parser.add_argument('--test_epoch', type=int, default=10, help='test epoch')
    parser.add_argument('--time_dim', type=int, default=32, help='time dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in graph model')
    parser.add_argument('--gamma_time', type=float, default=0.8, help='initial gamma_time (will be learnable)')
    parser.add_argument('--tau_time', type=float, default=4.0, help='initial tau_time in days (will be learnable)')

    fileroot = {
        'mimic3': 'C:\\Users\pc\Desktop\code\\baseline\mimic3\hosp',
        'mimic4': 'C:\\Users\pc\Desktop\code\\baseline\mimic4\hosp',
        'ccae': './data/processed_dip.pkl'
    }
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(f'{args.dataset}--{args.model}')

    cudaid = f"cuda:{args.dev}"
    device = torch.device(cudaid if torch.cuda.is_available() else "cpu")

    try:
        if args.dataset == 'mimic4':
            task_dataset = load_dataset(args.dataset,
                                        root=fileroot[args.dataset],
                                        task_fn=diag_prediction_mimic4_fn,
                                        dev=args.devm)
        elif args.dataset == 'mimic3':
            task_dataset = load_dataset(args.dataset,
                                        root=fileroot[args.dataset],
                                        task_fn=diag_prediction_mimic3_fn,                                    
                                        dev=args.devm)
        else:
            task_dataset = load_dataset(args.dataset,
                                        root=fileroot[args.dataset])
    except FileNotFoundError as e:
        print(f"文件路径错误: {str(e)}")
        print(f"请检查以下路径是否存在：{fileroot[args.dataset]}")
        return


    Tokenizers = get_init_tokenizers(task_dataset)
    label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))


    try:
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=16, progress_bar=True, use_memory_fs=False)
    except ImportError:
        print("Pandarallel 未安装，禁用并行处理")


    if args.model == 'Transformer':
        train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)
        model = Transformer(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

    elif args.model == 'RETAIN':
        train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)
        model = RETAIN(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

    elif args.model == 'KAME':
        train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)
        Tokenizers.update(get_parent_tokenizers(task_dataset))
        model = KAME(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

    elif args.model == 'StageNet':
        train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)
        model = StageNet(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)
    
    elif args.model == 'Hypergraph':

        data_path = f'./logs/{args.dataset}.pt'
        if os.path.exists(data_path):
            mdataset = load(data_path)
        else:
            mdataset = MMDataset(task_dataset, Tokenizers, note_emb_dict, device=device)
            dump(mdataset, data_path)
        trainset, validset, testset = split_dataset(mdataset)
        train_loader, val_loader, test_loader = mm_dataloader(trainset, validset, testset, batch_size=args.batch_size)
        model = HypergraphModel(Tokenizers, hidden_dim=args.hidden_dim, time_dim=args.time_dim, output_dim=len(task_dataset.get_all_tokens('conditions')),
                                num_heads=args.num_heads, num_layers=args.num_layers, gamma_time=args.gamma_time, tau_time=args.tau_time, device=device)

    elif args.model == 'TRANS':
        data_path = f'./logs/{args.dataset}_{args.pe_dim}.pkl'
        if os.path.exists(data_path):
            mdataset = load(data_path)
        else:
            mdataset = MMDataset(task_dataset, Tokenizers, dim=128, device=device, trans_dim=args.pe_dim)
            dump(mdataset, data_path)
        trainset, validset, testset = split_dataset(mdataset)
        train_loader, val_loader, test_loader = mm_dataloader(trainset, validset, testset,batch_size=args.batch_size)
        model = TRANS(Tokenizers, 128, len(task_dataset.get_all_tokens('conditions')), device, graph_meta=graph_meta, pe=args.pe_dim)

    ckptpath = f'./logs/trained_{args.model}_{args.dataset}.ckpt'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

    best = float('inf')
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        model = model.to(device)
        train_loss = train(train_loader, model, label_tokenizer, optimizer, device)
        val_loss = valid(val_loader, model, label_tokenizer, device)
        scheduler.step()

        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} | lr: {optimizer.param_groups[0]['lr']:.2e} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        if val_loss < best:
            torch.save(model.state_dict(), ckptpath)
            best = val_loss
        if epoch % 10 == 0:
            best_model = torch.load(ckptpath)
            model.load_state_dict(best_model)
            model = model.to(device)
            y_true, y_prob = test(test_loader, model, label_tokenizer)
            selected_metrics = compute_metrics(
            y_true=y_true,
            y_prob=y_prob,
            metrics=["f1_samples", "jaccard_samples", "pr_auc_samples", "precision_samples","roc_auc_samples"])
            print(
                f"[multilabel_metrics] "
                f"F1: {selected_metrics['f1_samples']:.4f} | "
                f"Jaccard: {selected_metrics['jaccard_samples']:.4f} | "
                f"PR-AUC: {selected_metrics['pr_auc_samples']:.4f} | "
                f"Precision: {selected_metrics['precision_samples']:.4f} | "
                f"ROC-AUC: {selected_metrics['roc_auc_samples']:.4f}\n"
                f"Visit-level: {selected_metrics['visit_level']:.4f}\n"
                f"Code-level: {selected_metrics['code_level']:.4f}"
            )

            
    if args.model == 'TRANS':
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        device = torch.device('cpu')
        model = TRANS(Tokenizers, 128, len(task_dataset.get_all_tokens('conditions')), device, graph_meta=graph_meta, pe=args.pe_dim)


    best_model = torch.load(ckptpath)
    model.load_state_dict(best_model)
    model = model.to(device)
    y_t_all, y_p_all = [], []
    y_true, y_prob = test(test_loader, model, label_tokenizer)
    selected_metrics = compute_metrics(
            y_true=y_true,
            y_prob=y_prob,
            metrics=["f1_samples", "jaccard_samples", "pr_auc_samples", "precision_samples","roc_auc_samples"])
    print(
        f"[multilabel_metrics] "
        f"F1: {selected_metrics['f1_samples']:.4f} | "
        f"Jaccard: {selected_metrics['jaccard_samples']:.4f} | "
        f"PR-AUC: {selected_metrics['pr_auc_samples']:.4f} | "
        f"Precision: {selected_metrics['precision_samples']:.4f} | "
        f"ROC-AUC: {selected_metrics['roc_auc_samples']:.4f}\n"
        f"Visit-level: {selected_metrics['visit_level']:.4f}\n"
        f"Code-level: {selected_metrics['code_level']:.4f}"
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()