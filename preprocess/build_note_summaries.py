import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


TARGET_SECTIONS = [
    "chief_complaint",
    "past_medical_history",
]

SECTION_ALIASES = {
    "chief_complaint": [r"chief complaint", r"cc"],
    "past_medical_history": [r"past medical history", r"pmh"],
}

STOP_ALIASES = [
    r"admission date",
    r"discharge date",
    r"date of birth",
    r"sex",
    r"service",
    r"allergies",
    r"attending",
    r"physical exam",
    r"physical examination",
    r"history of present illness",
    r"hpi",
    r"hospital course",
    r"brief hospital course",
    r"pertinent results",
    r"labs",
    r"laboratory",
    r"microbiology",
    r"imaging",
    r"radiology",
    r"social history",
    r"family history",
    r"discharge disposition",
    r"discharge condition",
    r"condition on discharge",
    r"discharge diagnosis",
    r"discharge diagnoses",
    r"final diagnosis",
    r"final diagnoses",
    r"discharge medications",
    r"medications on discharge",
    r"discharge meds",
    r"follow[- ]?up",
    r"discharge instructions",
]

DEID_RE = re.compile(r"\[\*\*.*?\*\*\]", flags=re.IGNORECASE)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_header(line: str) -> str:
    line = DEID_RE.sub("[DEID]", line or "")
    line = re.sub(r"\s+", " ", line.strip().lower())
    return line.strip(" :")


def build_header_patterns():
    out = []
    for key, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            out.append((key, alias))
    return out


HEADER_PATTERNS = build_header_patterns()


def match_target_header(line: str):
    low = normalize_header(line)
    for key, alias in HEADER_PATTERNS:
        if re.fullmatch(alias, low):
            return key, None
        m = re.match(rf"({alias})\s*:\s*(.+)", low)
        if m and ":" in line:
            return key, line.split(":", 1)[1].strip()
    return None, None


def is_any_header(line: str) -> bool:
    low = normalize_header(line)
    all_aliases = STOP_ALIASES + [a for aliases in SECTION_ALIASES.values() for a in aliases]
    return any(re.match(rf"^{alias}\s*:?", low) for alias in all_aliases)


def clean_line(line: str) -> str:
    line = DEID_RE.sub("[DEID]", line or "")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def extract_sections(note_text: str) -> dict:
    sections = {key: "" for key in TARGET_SECTIONS}
    current = None
    buffer = []

    def flush():
        nonlocal buffer, current
        if current and buffer:
            text = "\n".join(buffer).strip()
            if text:
                if sections[current]:
                    sections[current] += "\n" + text
                else:
                    sections[current] = text
        buffer = []

    for raw_line in (note_text or "").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        key, inline = match_target_header(line)
        if key:
            flush()
            current = key
            if inline:
                inline = clean_line(inline)
                if inline:
                    buffer.append(inline)
            continue

        if current and is_any_header(line):
            flush()
            current = None
            continue

        if current:
            cleaned = clean_line(line)
            if cleaned:
                buffer.append(cleaned)

    flush()
    return sections


def sections_to_text(sections: dict) -> str:
    labels = {
        "chief_complaint": "Chief Complaint",
        "past_medical_history": "Past Medical History",
    }
    parts = []
    for key in TARGET_SECTIONS:
        text = normalize_space(sections.get(key, ""))
        if text:
            parts.append(f"[{labels[key]}]\n{text}")
    return "\n\n".join(parts)


def build_prompt(section_text: str) -> str:
    return f"""You are a clinical summarization assistant.

Task:
Write a concise clinical overview of this patient's presentation using only the selected note sections below.

Requirements:
- Summarize the reason for admission and relevant past medical history when available.
- Use only information explicitly present in the note. Do not infer missing facts.
- Do not include names, dates, phone numbers, addresses, IDs, or other personal identifiers.
- Prefer clinically precise wording over administrative wording.
- Output strict JSON only, with this schema: {{"summary": "..."}}

Selected sections:
\"\"\"{section_text}\"\"\""""


def extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        return {"summary": normalize_space(text)}
    summary = obj.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)
    return {"summary": normalize_space(summary)}


def call_openai_compatible(api_base, api_key, model, prompt, timeout=120, max_retries=2):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful clinical summarization assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 1024,
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
    raise RuntimeError(f"Qwen API request failed: {last_error}")


def load_existing_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ids.add(int(obj["hadm_id"]))
            except Exception:
                continue
    return ids


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_used_hadm_ids(dataset, task, data_root, devm=False):
    from core.registry import TASK_SPECS
    from core.utils import load_dataset

    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task}")
    spec = TASK_SPECS[task]
    task_dataset = load_dataset(
        dataset,
        root=data_root,
        task_fn=spec["task_fn"][dataset],
        dev=devm,
    )

    used_ids = set()
    for sample in task_dataset.samples:
        for visit_id in sample.get("visit_history", []):
            try:
                used_ids.add(int(visit_id))
            except (TypeError, ValueError):
                continue
    return used_ids


def read_notes(note_path, dataset, chunksize, max_notes=None, target_hadm_ids=None):
    notes = defaultdict(list)
    total_rows = 0
    for chunk in pd.read_csv(note_path, chunksize=chunksize):
        chunk.columns = [c.lower() for c in chunk.columns]
        if "hadm_id" not in chunk.columns or "text" not in chunk.columns:
            raise ValueError("Input note CSV must contain hadm_id and text columns.")

        if dataset == "mimic3" and "category" in chunk.columns:
            chunk = chunk[chunk["category"].astype(str).str.lower() == "discharge summary"]

        chunk = chunk.dropna(subset=["hadm_id", "text"])
        sort_cols = [c for c in ["hadm_id", "charttime", "chartdate"] if c in chunk.columns]
        if sort_cols:
            chunk = chunk.sort_values(sort_cols)

        for _, row in chunk.iterrows():
            hid = int(row["hadm_id"])
            if target_hadm_ids is not None and hid not in target_hadm_ids:
                continue
            notes[hid].append(str(row["text"]))
            total_rows += 1
            if max_notes is not None and len(notes) >= max_notes:
                return notes, total_rows
    return notes, total_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build clinical summaries from MIMIC discharge notes.")
    parser.add_argument("--dataset", required=True, choices=["mimic3", "mimic4"])
    parser.add_argument("--task", choices=["diag", "drug"], default=None)
    parser.add_argument("--data_root", default="", help="If provided with --task, only summarize visits used by the task dataset.")
    parser.add_argument("--note_csv", required=True, help="MIMIC-IV discharge.csv or MIMIC-III NOTEEVENTS.csv")
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chunksize", type=int, default=10000)
    parser.add_argument("--max_notes", type=int, default=None)
    parser.add_argument("--devm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_jsonl is None:
        args.output_jsonl = f"./results/{args.dataset}_note_summaries.jsonl"
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    target_hadm_ids = None
    if args.task or args.data_root:
        if not args.task or not args.data_root:
            raise ValueError("--task and --data_root must be provided together to filter notes.")
        target_hadm_ids = load_used_hadm_ids(
            args.dataset,
            args.task,
            args.data_root,
            devm=args.devm,
        )
        print(f"[notes] filtering to {len(target_hadm_ids)} task-used admissions")

    done_ids = load_existing_ids(args.output_jsonl) if args.resume else set()
    notes, total_rows = read_notes(
        args.note_csv,
        args.dataset,
        args.chunksize,
        args.max_notes,
        target_hadm_ids=target_hadm_ids,
    )
    print(f"[notes] loaded {len(notes)} admissions from {total_rows} rows")
    if done_ids:
        print(f"[notes] resume: {len(done_ids)} existing summaries")

    for hadm_id, texts in tqdm(sorted(notes.items()), desc="Qwen2.5 summaries"):
        if hadm_id in done_ids:
            continue
        sections = extract_sections("\n\n".join(texts))
        section_text = sections_to_text(sections)
        if not section_text:
            continue

        prompt = build_prompt(section_text)
        raw = call_openai_compatible(args.api_base, args.api_key, args.model, prompt)
        parsed = extract_json_object(raw)
        append_jsonl(
            args.output_jsonl,
            {
                "hadm_id": int(hadm_id),
                "summary": parsed["summary"],
                "sections_found": [k for k in TARGET_SECTIONS if sections.get(k)],
            },
        )

    print(f"[done] summaries saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
