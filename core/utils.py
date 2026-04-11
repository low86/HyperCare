import torch
import numpy as np
from tqdm import *
from joblib import load
from datetime import datetime
from pyhealth.tokenizer import Tokenizer
from pyhealth.datasets import  MIMIC4Dataset, MIMIC3Dataset
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.medcode import InnerMap

def get_label_tokenizer(label_tokens):
    special_tokens = []
    label_tokenizer = Tokenizer(
        label_tokens,
        special_tokens=special_tokens,
    )
    return label_tokenizer

def batch_to_multihot(label, num_labels: int) -> torch.tensor:

    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot

def prepare_labels(
        labels,
        label_tokenizer: Tokenizer,
    ) -> torch.Tensor:
    labels_index = label_tokenizer.batch_encode_2d(
        labels, padding=False, truncation=False
    )
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = batch_to_multihot(labels_index, num_labels)
    return labels

def parse_datetimes(datetime_strings):
    return [datetime.strptime(dt_str, "%Y-%m-%d %H:%M") for dt_str in datetime_strings]

def timedelta_to_days(tdelta):
    return tdelta.days + tdelta.seconds / 86400.0  

def convert_to_relative_time(datetime_strings):
    datetimes = parse_datetimes(datetime_strings)
    base_time = min(datetimes)
    return [timedelta_to_days(dt - base_time) for dt in datetimes]

def load_dataset(dataset, root , tables=["diagnoses_icd", "procedures_icd", "prescriptions"], task_fn = None, dev = False, use_icd_raw: bool = False):
    if dataset=='mimic3':
        code_mapping = {
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM",
            "ICD9PROC": "CCSPROC",
        }
        if use_icd_raw:
            code_mapping = {
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            }
        dataset = MIMIC3Dataset(
            root = root,
            dev = dev,
            tables = ['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS'], 
            code_mapping=code_mapping,
            refresh_cache=False,
        )
    elif dataset == 'mimic4':
        code_mapping = {
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM",
            "ICD9PROC": "CCSPROC",
            "ICD10CM": "CCSCM",
            "ICD10PROC": "CCSPROC",
        }
        if use_icd_raw:
            code_mapping = {
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            }
        dataset = MIMIC4Dataset(
            root=root,
            dev=dev,
            tables=tables, 
            code_mapping=code_mapping,
            refresh_cache=False,
        )
    else:
        return load(root)
    return dataset.set_task(task_fn=task_fn)


def get_init_tokenizers(task_dataset, keys = ['cond_hist', 'procedures', 'drugs']):
    Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
    return Tokenizers

def get_parent_tokenizers(task_dataset, keys = ['cond_hist', 'procedures']):
    parent_tokenizers = {}
    dictionary = {'cond_hist':InnerMap.load("ICD9CM"), 'procedures':InnerMap.load("ICD9PROC")}
    for feature_key in keys:
        assert feature_key in dictionary.keys()
        tokens = task_dataset.get_all_tokens(feature_key)
        parent_tokens = set()
        for token in tokens:
            try:
                parent_tokens.update(dictionary[feature_key].get_ancestors(token))
            except:
                continue
        parent_tokenizers[feature_key + '_parent'] = Tokenizer(tokens=list(parent_tokens), special_tokens=["<pad>"])
    return parent_tokenizers


def split_dataset(dataset, train_ratio=0.75, valid_ratio=0.1, test_ratio=0.15):
    # Ensure the ratios sum to 1
    total = train_ratio + valid_ratio + test_ratio
    if total != 1.0:
        raise ValueError("Ratios must sum to 1.")
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size
    # Randomly splitting the dataset
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])
    return train_set, valid_set, test_set


def custom_collate_fn(batch):
    sequence_data_list = [item[0] for item in batch]
    graph_data_list = [item[1] for item in batch]

    sequence_data_batch = {key: [d[key] for d in sequence_data_list if d[key]!=[]] for key in sequence_data_list[0]}

    graph_data_batch = graph_data_list

    return sequence_data_batch, graph_data_batch


def mm_dataloader(trainset, validset, testset, batch_size = 64):
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
 
    return train_loader, val_loader, test_loader 

def custom_collate_fn_seq(batch):
  
    sequence_data_batch = {
        key: [d[key] for d in batch if d[key] != []]
        for key in batch[0]
    }
    return sequence_data_batch


def seq_dataloader(dataset, split_ratio = [0.75, 0.1, 0.15], batch_size = 64):
    train_dataset, val_dataset, test_dataset = split_by_patient(dataset, split_ratio)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
