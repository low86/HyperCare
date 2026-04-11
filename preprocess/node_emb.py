import os
import torch
import random
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pyhealth.medcode import InnerMap

# -----------------------------
# Set a global seed (optional, recommended at the top of the script)
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# If you want reproducible runs, call set_seed(42) before running.
# set_seed(42)

icd9cm = InnerMap.load("ICD9CM")
icd9proc = InnerMap.load("ICD9PROC")
atc = InnerMap.load("ATC")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# -----------------------------
# 1. Load ClinicalBERT
# -----------------------------
def load_med_text_encoder(model_name="emilyalsentzer/Bio_ClinicalBERT"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


# -----------------------------
# 2. Encode text into a CLS vector
# -----------------------------
def encode_text(model, tokenizer, text: str, device="cpu"):
    token = tokenizer(text, return_tensors="pt", truncation=True, max_length=32).to(device)
    with torch.no_grad():
        out = model(**token).last_hidden_state[:, 0]  # CLS embedding
    return out.squeeze(0)

def code_to_text(code: str):
    if not isinstance(code, str):
        return None

    code = code.strip()

    # ---- ICD-9 diagnosis ----
    try:
        text = icd9cm.lookup(code)
        if text is not None:
            return text
    except Exception:
        pass

    # ---- ICD-9 procedure ----
    try:
        text = icd9proc.lookup(code)
        if text is not None:
            return text
    except Exception:
        pass

    try:
        text = atc.lookup(code)
        if text:
            return text
    except Exception:
        pass

    return None




def add_codes(code_set, entry):
    """entry may be list[str] or list[list[str]]"""
    if entry is None:
        return
    if isinstance(entry, list):
        for x in entry:
            if isinstance(x, list):      # nested list
                for c in x:
                    code_set.add(c)
            else:                        # flat list
                code_set.add(x)


# -----------------------------
# 4. Adapt for your task_dataset version
# -----------------------------
# def build_semantic_emb(task_dataset, device="cpu"):
#     print(">>> Loading ClinicalBERT...")
#     tokenizer, model = load_med_text_encoder()
#     model = model.to(device)
# def build_semantic_emb(task_dataset, device="cpu", out_dim=32, seed=42):
#     print(">>> Loading ClinicalBERT...")
#     tokenizer, model = load_med_text_encoder()
#     model = model.to(device)
#     # model.eval()

#     # New: offline semantic projection layer
#     torch.manual_seed(seed)
#     # projector = nn.Linear(768, out_dim).to(device)
#     # projector.eval()

#     code_set = set()

#     for sample in task_dataset:
#         add_codes(code_set, sample["cond_hist"])
#         add_codes(code_set, sample["procedures"])
#         add_codes(code_set, sample["drugs"])


#     print(f">>> Total unique codes: {len(code_set)}")
#     # print(list(code_set)[:20])

#     semantic_emb = {}
#     missing = 0

#     # 2. Build semantic embedding for each code
#     for code in code_set:
#         text = code_to_text(code)
#         if text is None:
#             missing += 1
#             semantic_emb[code] = torch.zeros(768)  # BERT dim
#         else:
#             emb = encode_text(model, tokenizer, text, device)
#             semantic_emb[code] = emb.cpu()

#     print(f">>> Missing text: {missing}/{len(code_set)}")
#     print(">>> Semantic embedding build done.")
#     return semantic_emb
from sklearn.decomposition import PCA

def build_semantic_emb(task_dataset, device="cpu", out_dim=64, seed=42):
    print(">>> Loading ClinicalBERT...")
    tokenizer, model = load_med_text_encoder()
    model = model.to(device)
    model.eval()

    code_set = set()
    for sample in task_dataset:
        add_codes(code_set, sample.get("cond_hist", []))
        add_codes(code_set, sample.get("procedures", []))
        add_codes(code_set, sample.get("drugs", []))

    print(f">>> Total unique codes: {len(code_set)}")

    # Step 1: collect valid 768-d embeddings
    valid_embeddings = []
    valid_codes = []

    missing = 0
    for code in code_set:
        text = code_to_text(code)
        if text is None:
            missing += 1
            continue
        with torch.no_grad():
            emb = encode_text(model, tokenizer, text, device)
            valid_embeddings.append(emb.cpu().numpy())
            valid_codes.append(code)

    print(f">>> Missing text: {missing}/{len(code_set)}")
    print(f">>> Valid codes for PCA: {len(valid_embeddings)}")

    # Step 2: fit PCA (fixed random seed for reproducibility)
    pca = PCA(n_components=out_dim, random_state=seed)
    pca.fit(valid_embeddings)

    # Step 3: build final embedding dict
    semantic_emb = {}
    reduced = pca.transform(valid_embeddings)

    for code, emb_low in zip(valid_codes, reduced):
        semantic_emb[code] = torch.tensor(emb_low)

    # Missing codes use the mean embedding (better than all zeros)
    mean_emb = torch.tensor(pca.mean_)
    for code in code_set:
        if code not in semantic_emb:
            semantic_emb[code] = mean_emb

    print(">>> Semantic embedding build done with PCA.")
    return semantic_emb
