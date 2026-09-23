import argparse
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from tqdm import tqdm

from core.registry import TASK_SPECS
from core.utils import get_init_tokenizers, load_dataset
from preprocess.embed_note_summaries import MODEL_PRESETS, SummaryEncoder, atomic_save, load_resume


TYPE_CONFIG = {
    "diag": {
        "cond": ("cond_hist", "CCSCM", "condition or diagnosis"),
        "proc": ("procedures", "CCSPROC", "procedure"),
        "drug": ("drugs", "ATC", "medication"),
    },
    "drug": {
        "cond": ("conditions", "CCSCM", "condition or diagnosis"),
        "proc": ("procedures", "CCSPROC", "procedure"),
        "drug": ("drugs_hist", "ATC", "medication"),
    },
}


def tokenizer_vocab(tokenizer):
    vocab = tokenizer.vocabulary
    if isinstance(vocab, dict):
        return sorted(vocab, key=vocab.get)
    for attr in ["token2idx", "_token2idx"]:
        if hasattr(vocab, attr):
            token_to_idx = getattr(vocab, attr)
            return sorted(token_to_idx, key=token_to_idx.get)
    return tokenizer.convert_indices_to_tokens(list(range(tokenizer.get_vocabulary_size())))


def load_code_map(map_name):
    try:
        from pyhealth.medcode import InnerMap

        return InnerMap.load(map_name)
    except Exception:
        return None


def lookup_code_name(code_map, code):
    if code_map is None:
        return ""
    for method_name in ["lookup", "get_name", "get_description", "get_desc"]:
        method = getattr(code_map, method_name, None)
        if method is None:
            continue
        try:
            value = method(code)
        except Exception:
            continue
        if value:
            return str(value)
    for attr_name in ["graph", "code2name", "code_to_name", "vocabulary"]:
        obj = getattr(code_map, attr_name, None)
        if obj is None:
            continue
        try:
            value = obj[code]
        except Exception:
            continue
        if isinstance(value, dict):
            for key in ["name", "description", "desc"]:
                if value.get(key):
                    return str(value[key])
        elif value:
            return str(value)
    return ""


def build_prompt(code, code_system, code_name, code_kind):
    name_text = code_name if code_name else "unknown"
    return f"""You are a clinical terminology assistant.

Task:
Write a concise note-style clinical description for one structured EHR code.

Requirements:
- Use the code system, code, and official name when available.
- Describe what this {code_kind} usually represents in patient records.
- Use clinically precise wording.
- Do not invent patient-specific facts.
- Output strict JSON only, with this schema: {{"description": "..."}}

Code system: {code_system}
Code: {code}
Official name: {name_text}"""


def extract_description(text):
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        return " ".join(text.split())
    desc = obj.get("description", "")
    if not isinstance(desc, str):
        desc = str(desc)
    return " ".join(desc.split())


def call_openai_compatible(api_base, api_key, model, prompt, timeout=120, max_retries=2):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful clinical terminology assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM API request failed: {last_error}")


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_existing_descriptions(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("key")
            desc = obj.get("description")
            if key and desc:
                out[str(key)] = str(desc)
    return out


def iter_code_rows(task_dataset, tokenizers, task):
    rows = []
    for _, (feature_key, code_system, code_kind) in TYPE_CONFIG[task].items():
        code_map = load_code_map(code_system)
        for code in tokenizer_vocab(tokenizers[feature_key]):
            if code == "<pad>":
                continue
            key = f"{code_system}:{code}"
            rows.append(
                {
                    "key": key,
                    "code": code,
                    "code_system": code_system,
                    "code_kind": code_kind,
                    "name": lookup_code_name(code_map, code),
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build semantic embeddings for CCS/ATC code nodes.")
    parser.add_argument("--task", required=True, choices=["diag", "drug"])
    parser.add_argument("--dataset", required=True, choices=["mimic3", "mimic4"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--description_jsonl", default=None)
    parser.add_argument("--output_npy", default=None)
    parser.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--encoder", default="bge-m3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--devm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.description_jsonl is None:
        args.description_jsonl = f"./results/{args.task}_{args.dataset}_node_descriptions.jsonl"
    if args.output_npy is None:
        args.output_npy = f"./results/{args.task}_{args.dataset}_node_emb.npy"
    os.makedirs(os.path.dirname(args.description_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_npy) or ".", exist_ok=True)

    spec = TASK_SPECS[args.task]
    task_dataset = load_dataset(
        args.dataset,
        root=args.data_root,
        task_fn=spec["task_fn"][args.dataset],
        dev=args.devm,
    )
    tokenizers = get_init_tokenizers(task_dataset, keys=spec["feature_keys"])
    rows = iter_code_rows(task_dataset, tokenizers, args.task)
    print(f"[node] codes={len(rows)}")

    descriptions = load_existing_descriptions(args.description_jsonl) if args.resume else {}
    for row in tqdm(rows, desc="LLM code descriptions"):
        if row["key"] in descriptions:
            continue
        prompt = build_prompt(row["code"], row["code_system"], row["name"], row["code_kind"])
        raw = call_openai_compatible(args.api_base, args.api_key, args.llm_model, prompt)
        desc = extract_description(raw)
        descriptions[row["key"]] = desc
        append_jsonl(
            args.description_jsonl,
            {
                **row,
                "description": desc,
            },
        )

    encoder_name = MODEL_PRESETS.get(args.encoder, args.encoder)
    encoder = SummaryEncoder(
        model_name=encoder_name,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    emb_dict = load_resume(args.output_npy) if args.resume else {}

    pending_keys = []
    pending_texts = []
    processed = 0

    def flush():
        nonlocal pending_keys, pending_texts, processed
        if not pending_keys:
            return
        embs = encoder.encode_batch(pending_texts, args.max_length)
        for key, emb in zip(pending_keys, embs):
            emb_dict[str(key)] = emb.astype("float32")
            processed += 1
        pending_keys = []
        pending_texts = []
        if args.save_every and processed % args.save_every == 0:
            atomic_save(args.output_npy, emb_dict)

    for row in tqdm(rows, desc="Encoding code descriptions"):
        key = row["key"]
        if args.resume and key in emb_dict:
            continue
        desc = descriptions.get(key)
        if not desc:
            continue
        pending_keys.append(key)
        pending_texts.append(desc)
        if len(pending_keys) >= args.batch_size:
            flush()

    flush()
    atomic_save(args.output_npy, emb_dict)
    dim = len(next(iter(emb_dict.values()))) if emb_dict else 0
    print(f"[done] node embeddings saved to {args.output_npy}; dim={dim}")


if __name__ == "__main__":
    main()
