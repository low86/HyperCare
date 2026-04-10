import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

NOTEEVENTS_PATH = r"C:\Users\pc\Desktop\code\data\NOTEEVENTS.csv"

print("Loading NOTEEVENTS.csv...")
notes_df = pd.read_csv(
    NOTEEVENTS_PATH,
    usecols=["HADM_ID", "CHARTTIME", "TEXT"]
)
notes_df = notes_df.dropna(subset=["HADM_ID"])
notes_df = notes_df.sort_values(["HADM_ID", "CHARTTIME"])


# 缓存：避免重复查找
_NOTE_CACHE = {}

def clean_note(text: str):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\[\*\*.*?\*\*\]", "", text)       # 去除匿名标签
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def get_notes_for_visit(hadm_id):
    """返回单个 HADM_ID 对应的所有清洗后的 note（按时间排序）"""
    if hadm_id in _NOTE_CACHE:
        return _NOTE_CACHE[hadm_id]

    df = notes_df[notes_df["HADM_ID"] == hadm_id]
    if df.empty:
        _NOTE_CACHE[hadm_id] = []
        return []

    notes = [clean_note(t) for t in df["TEXT"].tolist()]
    _NOTE_CACHE[hadm_id] = notes
    return notes


# class NoteEncoder:
#     def __init__(self, device="cuda"):
#         self.device = device

#         # ClinicalBERT
#         model_name = "emilyalsentzer/Bio_ClinicalBERT"

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(device)
#         self.model.eval()

#     @torch.no_grad()
#     def encode(self, notes: list[str]):

#         if len(notes) == 0:
#             # 空 note 返回全零向量
#             return torch.zeros(1, self.model.config.hidden_size)

#         text = " ".join(notes)[:4000]  # 防止特别长（ClinicalBERT 只吃 512 tokens）

#         inputs = self.tokenizer(
#             text,
#             max_length=512,
#             truncation=True,
#             return_tensors="pt"
#         ).to(self.device)

#         outputs = self.model(**inputs)       # last hidden state
#         cls_emb = outputs[0][:, 0, :]         # 取 CLS (1, hidden_dim)

#         return cls_emb.cpu()                 # 返回 CPU 上的 embedding

class NoteEncoder:
    def __init__(self, device="cuda"):
        self.device = device

        # Clinical-Longformer
        model_name = "yikuan8/Clinical-Longformer"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Longformer 的最大长度是 4096
        self.max_length = 4096

    @torch.no_grad()
    def encode(self, notes: list[str]):

        if len(notes) == 0:
            return torch.zeros(1, self.model.config.hidden_size)

        # 更长的输入不需要截断到 4000
        text = " ".join(notes)
        text = text[:16000]  # 长一些也没事，tokenizer 会按 max_length 截断

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        # Longformer CLS token（[0, 0, :]）
        cls_emb = outputs.last_hidden_state[:, 0, :]

        return cls_emb.cpu()



note_encoder = NoteEncoder(device="cuda")

note_emb_dict = {}

all_hadm_ids = notes_df["HADM_ID"].unique()
print("HADM_ID 类型检查:")
print(f"all_hadm_ids 类型: {type(all_hadm_ids)}")
print(f"all_hadm_ids 长度: {len(all_hadm_ids)}")
print(f"前5个HADM_ID: {all_hadm_ids[:5]}")
print(f"第一个HADM_ID的类型: {type(all_hadm_ids[0])}")
print(f"第一个HADM_ID的值: '{int(all_hadm_ids[0])}'")

# 检查是否有特殊值
print(f"\n检查特殊值:")
print(f"是否有NaN: {pd.isna(all_hadm_ids).any()}")
print(f"是否有空字符串: {(all_hadm_ids == '').any()}")

for hid in tqdm(all_hadm_ids):
    notes = get_notes_for_visit(hid)
    emb = note_encoder.encode(notes).squeeze(0).cpu().numpy()  # [768]
    note_emb_dict[int(hid)] = emb

np.save("note_emb_dict.npy", note_emb_dict)


# import re
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel


# NOTEEVENTS_PATH = r"C:\\Users\\pc\\Desktop\\code\\data\\NOTEEVENTS.csv"

# print("Loading NOTEEVENTS.csv...")
# notes_df = pd.read_csv(
#     NOTEEVENTS_PATH,
#     usecols=["HADM_ID", "CHARTTIME", "TEXT"]
# )
# notes_df = notes_df.dropna(subset=["HADM_ID"])
# notes_df = notes_df.sort_values(["HADM_ID", "CHARTTIME"])


# # 缓存：避免重复查找
# _NOTE_CACHE = {}

# def clean_note(text: str):
#     if not isinstance(text, str):
#         return ""
#     t = re.sub(r"\[\*\*.*?\*\*\]", "", text)       # 去除匿名标签
#     t = t.replace("\n", " ").replace("\r", " ")
#     t = re.sub(r"\s+", " ", t)
#     return t.strip()

# def get_notes_for_visit(hadm_id):
#     """返回单个 HADM_ID 对应的所有清洗后的 note（按时间排序）"""
#     if hadm_id in _NOTE_CACHE:
#         return _NOTE_CACHE[hadm_id]

#     df = notes_df[notes_df["HADM_ID"] == hadm_id]
#     if df.empty:
#         _NOTE_CACHE[hadm_id] = []
#         return []

#     notes = [clean_note(t) for t in df["TEXT"].tolist()]
#     _NOTE_CACHE[hadm_id] = notes
#     return notes



# class NoteEncoder:
#     def __init__(self, device="cuda", batch_size=4):
#         self.device = device
#         self.batch_size = batch_size

#         model_name = "yikuan8/Clinical-Longformer"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(device)
#         self.model.eval()

#         self.max_length = 4096

#     @torch.no_grad()
#     def batch_encode(self, batch_notes: list[list[str]]):
#         """ batch_notes: list of list[str], 每个元素对应一个 HADM 的所有 note """

#         texts = []
#         for notes in batch_notes:
#             if len(notes) == 0:
#                 texts.append("")  # 空则 later 会输出零向量
#             else:
#                 t = " ".join(notes)
#                 t = t[:16000]
#                 texts.append(t)

#         # tokenizer 可以直接处理 batch
#         inputs = self.tokenizer(
#             texts,
#             max_length=self.max_length,
#             truncation=True,
#             padding=True,  # ⭐ 启用 batch padding！
#             return_tensors="pt"
#         ).to(self.device)

#         outputs = self.model(**inputs)
#         cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, 768]

#         return cls_emb.cpu()


# note_encoder = NoteEncoder(device="cuda")

# all_hadm_ids = notes_df["HADM_ID"].unique()
# print("HADM_ID 类型检查:")
# print(f"all_hadm_ids 类型: {type(all_hadm_ids)}")
# print(f"all_hadm_ids 长度: {len(all_hadm_ids)}")
# print(f"前5个HADM_ID: {all_hadm_ids[:5]}")
# print(f"第一个HADM_ID的类型: {type(all_hadm_ids[0])}")
# print(f"第一个HADM_ID的值: '{int(all_hadm_ids[0])}'")

# # 检查是否有特殊值
# print(f"\n检查特殊值:")
# print(f"是否有NaN: {pd.isna(all_hadm_ids).any()}")
# print(f"是否有空字符串: {(all_hadm_ids == '').any()}")
# note_emb_dict = {}
# batch_size = 8
# batch_hadm = []
# batch_notes = []

# for hid in tqdm(all_hadm_ids):

#     notes = get_notes_for_visit(hid)

#     batch_hadm.append(hid)
#     batch_notes.append(notes)

#     # 当 batch 满了 → 统一推理
#     if len(batch_hadm) == batch_size:
#         embeddings = note_encoder.batch_encode(batch_notes)  # [B, 768]

#         for h, emb in zip(batch_hadm, embeddings):
#             note_emb_dict[int(h)] = emb.numpy()

#         batch_hadm = []
#         batch_notes = []

# # 收尾：处理剩余不足 batch 的部分
# if len(batch_hadm) > 0:
#     embeddings = note_encoder.batch_encode(batch_notes)
#     for h, emb in zip(batch_hadm, embeddings):
#         note_emb_dict[int(h)] = emb.numpy()

# np.save("note_emb_dict.npy", note_emb_dict)

