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


# Cache to avoid repeated lookups
_NOTE_CACHE = {}

def clean_note(text: str):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\[\*\*.*?\*\*\]", "", text)       # remove de-identified tags
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def get_notes_for_visit(hadm_id):
    """Return all cleaned notes for a single HADM_ID, ordered by time."""
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
#             # empty note -> zero vector
#             return torch.zeros(1, self.model.config.hidden_size)

#         text = " ".join(notes)[:4000]  # guard against very long text (ClinicalBERT ~512 tokens)

#         inputs = self.tokenizer(
#             text,
#             max_length=512,
#             truncation=True,
#             return_tensors="pt"
#         ).to(self.device)

#         outputs = self.model(**inputs)       # last hidden state
#         cls_emb = outputs[0][:, 0, :]         # CLS token (1, hidden_dim)

#         return cls_emb.cpu()                 # return CPU embedding

class NoteEncoder:
    def __init__(self, device="cuda"):
        self.device = device

        # Clinical-Longformer
        model_name = "yikuan8/Clinical-Longformer"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Longformer max length is 4096
        self.max_length = 4096

    @torch.no_grad()
    def encode(self, notes: list[str]):

        if len(notes) == 0:
            return torch.zeros(1, self.model.config.hidden_size)

        # Longer input does not need to be truncated to 4000
        text = " ".join(notes)
        text = text[:16000]  # tokenizer will truncate to max_length

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        # Longformer CLS token ([0, 0, :])
        cls_emb = outputs.last_hidden_state[:, 0, :]

        return cls_emb.cpu()



note_encoder = NoteEncoder(device="cuda")

note_emb_dict = {}

all_hadm_ids = notes_df["HADM_ID"].unique()
print("HADM_ID type check:")
print(f"all_hadm_ids type: {type(all_hadm_ids)}")
print(f"all_hadm_ids length: {len(all_hadm_ids)}")
print(f"first 5 HADM_ID: {all_hadm_ids[:5]}")
print(f"first HADM_ID type: {type(all_hadm_ids[0])}")
print(f"first HADM_ID value: '{int(all_hadm_ids[0])}'")

# Check for special values
print("\nCheck special values:")
print(f"has NaN: {pd.isna(all_hadm_ids).any()}")
print(f"has empty string: {(all_hadm_ids == '').any()}")

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


# # Cache to avoid repeated lookups
# _NOTE_CACHE = {}

# def clean_note(text: str):
#     if not isinstance(text, str):
#         return ""
#     t = re.sub(r"\[\*\*.*?\*\*\]", "", text)       # remove de-identified tags
#     t = t.replace("\n", " ").replace("\r", " ")
#     t = re.sub(r"\s+", " ", t)
#     return t.strip()

# def get_notes_for_visit(hadm_id):
#     """Return all cleaned notes for a single HADM_ID, ordered by time."""
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
#         """batch_notes: list of list[str], each element is all notes for one HADM."""

#         texts = []
#         for notes in batch_notes:
#             if len(notes) == 0:
#                 texts.append("")  # empty -> zero vector later
#             else:
#                 t = " ".join(notes)
#                 t = t[:16000]
#                 texts.append(t)

#         # tokenizer can handle batch directly
#         inputs = self.tokenizer(
#             texts,
#             max_length=self.max_length,
#             truncation=True,
#             padding=True,  # enable batch padding
#             return_tensors="pt"
#         ).to(self.device)

#         outputs = self.model(**inputs)
#         cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, 768]

#         return cls_emb.cpu()


# note_encoder = NoteEncoder(device="cuda")

# all_hadm_ids = notes_df["HADM_ID"].unique()
# print("HADM_ID type check:")
# print(f"all_hadm_ids type: {type(all_hadm_ids)}")
# print(f"all_hadm_ids length: {len(all_hadm_ids)}")
# print(f"first 5 HADM_ID: {all_hadm_ids[:5]}")
# print(f"first HADM_ID type: {type(all_hadm_ids[0])}")
# print(f"first HADM_ID value: '{int(all_hadm_ids[0])}'")

# # Check for special values
# print("\nCheck special values:")
# print(f"has NaN: {pd.isna(all_hadm_ids).any()}")
# print(f"has empty string: {(all_hadm_ids == '').any()}")
# note_emb_dict = {}
# batch_size = 8
# batch_hadm = []
# batch_notes = []

# for hid in tqdm(all_hadm_ids):

#     notes = get_notes_for_visit(hid)

#     batch_hadm.append(hid)
#     batch_notes.append(notes)

#     # When batch is full -> run inference
#     if len(batch_hadm) == batch_size:
#         embeddings = note_encoder.batch_encode(batch_notes)  # [B, 768]

#         for h, emb in zip(batch_hadm, embeddings):
#             note_emb_dict[int(h)] = emb.numpy()

#         batch_hadm = []
#         batch_notes = []

# # Finalize: process remaining items smaller than batch
# if len(batch_hadm) > 0:
#     embeddings = note_encoder.batch_encode(batch_notes)
#     for h, emb in zip(batch_hadm, embeddings):
#         note_emb_dict[int(h)] = emb.numpy()

# np.save("note_emb_dict.npy", note_emb_dict)
