import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_PRESETS = {
    "clinical-longformer": "yikuan8/Clinical-Longformer",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "bge-m3": "BAAI/bge-m3",
    "jina-v3": "jinaai/jina-embeddings-v3",
}


def iter_summary_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            summary = obj.get("summary", "")
            if summary:
                yield int(obj["hadm_id"]), str(summary)


def load_resume(path):
    if os.path.exists(path):
        try:
            obj = np.load(path, allow_pickle=True).item()
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def atomic_save(path, obj):
    tmp = path + ".tmp.npy"
    np.save(tmp, obj, allow_pickle=True)
    os.replace(tmp, path)


class SummaryEncoder:
    def __init__(self, model_name, device, trust_remote_code=False, local_files_only=False):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_batch(self, texts, max_length):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]
        return cls.detach().cpu().float().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Encode clinical note summaries into visit embeddings.")
    parser.add_argument("--summary_jsonl", required=True)
    parser.add_argument("--output_npy", default="./logs/note_emb_dict.npy")
    parser.add_argument(
        "--encoder",
        default="bioclinicalbert",
        help="Preset name or Hugging Face model/path. Presets: "
        + ", ".join(sorted(MODEL_PRESETS)),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_npy) or ".", exist_ok=True)

    model_name = MODEL_PRESETS.get(args.encoder, args.encoder)
    encoder = SummaryEncoder(
        model_name=model_name,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    emb_dict = load_resume(args.output_npy) if args.resume else {}
    rows = [(hid, txt) for hid, txt in iter_summary_jsonl(args.summary_jsonl)]
    print(f"[embed] summaries={len(rows)} resume={len(emb_dict)} model={model_name}")

    pending_ids = []
    pending_texts = []
    processed = 0

    def flush():
        nonlocal pending_ids, pending_texts, processed
        if not pending_ids:
            return
        embs = encoder.encode_batch(pending_texts, args.max_length)
        for hid, emb in zip(pending_ids, embs):
            emb_dict[int(hid)] = emb.astype("float32")
            processed += 1
        pending_ids = []
        pending_texts = []
        if args.save_every and processed % args.save_every == 0:
            atomic_save(args.output_npy, emb_dict)

    for hadm_id, summary in tqdm(rows, desc="Encoding summaries"):
        if args.resume and hadm_id in emb_dict:
            continue
        pending_ids.append(hadm_id)
        pending_texts.append(summary)
        if len(pending_ids) >= args.batch_size:
            flush()

    flush()
    atomic_save(args.output_npy, emb_dict)
    dim = len(next(iter(emb_dict.values()))) if emb_dict else 0
    print(f"[done] embeddings saved to {args.output_npy}; dim={dim}")


if __name__ == "__main__":
    main()
