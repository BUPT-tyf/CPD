"""Build DetectRL-X English general and rewriting test partitions."""
import hashlib
import json
import re
from pathlib import Path

import ijson
from .download import public_file
from tqdm.auto import tqdm

REVISION = "ccc9a0a74793dd3448e0da1650b430f5df1b0402"
REPO_ID = "WUJUNCHAO/DetectRL-X"
REFINEMENTS = ("polishing", "expanding", "condensing")


def text_or_empty(value):
    return value.strip() if isinstance(value, str) else ""


def pair_id_for(human_text, lang, model, domain):
    # The public schema does not list a stable source ID. This derived ID
    # matches rows across files only when their human control text is shared.
    normalized = re.sub(r"\s+", " ", human_text).strip()
    identity = "\x1f".join([
        lang, str(model).lower(), str(domain).lower(), normalized
    ])
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
    return f"drx_{lang}_{digest}"


def make_pair(row, index, split, strategy, source_file):
    lang = str(row.get("lang") or "").strip().lower()
    model = str(row.get("model") or "unknown").strip()
    domain = str(row.get("domain") or "unknown").strip()
    human = text_or_empty(row.get("human_written_text"))
    machine = text_or_empty(row.get("llm_generated_text"))

    if not human or not machine:
        return None

    pair_id = pair_id_for(human, lang, model, domain)
    positive_category = "A" if strategy == "general" else "AA"

    common = {
        "dataset": "DetectRL-X",
        "domain": domain,
        "lang": lang,
        "split": split,
        "pair_id": pair_id,
        "pair_model": model,
        "strategy": strategy,
        "source_file": source_file,
        "source_row_index": index,
        "pair_id_method": "lang_model_domain_human_text_hash",
    }

    human_record = {
        "id": f"{pair_id}_H_{strategy}_{index}",
        "prompt": "",
        "generation": human,
        "model": "human",
        "label": 0,
        "category": "H",
        "edit_type": "none",
        **common,
    }
    positive_record = {
        "id": f"{pair_id}_{positive_category}_{strategy}_{index}",
        "prompt": "",
        "generation": machine,
        "model": model,
        "label": 1,
        "category": positive_category,
        "edit_type": "none" if strategy == "general" else "ai_post_editing",
        **common,
    }
    return pair_id, human_record, positive_record


def write_jsonl(writer, record):
    writer.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def build(output_dir, raw_dir=None, revision=REVISION, group="general"):
    def pairs(strategy):
        filename = f"Binary/binary_{strategy}_open.json"
        if raw_dir is None:
            path = public_file(REPO_ID, revision, filename)
        else:
            path = raw_dir / filename
        with path.open("rb") as source:
            for index, row in enumerate(tqdm(ijson.items(source, "item"), desc=strategy)):
                if str(row.get("lang")).strip().lower() != "english" or str(row.get("split")).strip().lower() != "test":
                    continue
                pair = make_pair(row, index, "test", strategy, filename)
                if pair is not None:
                    yield pair

    if group == "general":
        with (output_dir / "DetectRLX_english_test_general.jsonl").open("w", encoding="utf-8") as handle:
            for _, human, machine in pairs("general"):
                write_jsonl(handle, human)
                write_jsonl(handle, machine)
        return

    directory = output_dir / "AA"
    directory.mkdir()
    seen_humans = set()
    with (directory / "DetectRLX_english_test_AA_all.jsonl").open("w", encoding="utf-8") as pooled:
        for strategy in REFINEMENTS:
            filename = f"DetectRLX_english_test_AA_{strategy}.jsonl"
            with (directory / filename).open("w", encoding="utf-8") as handle:
                for pair_id, human, edited in pairs(strategy):
                    write_jsonl(handle, human)
                    write_jsonl(handle, edited)
                    if pair_id not in seen_humans:
                        write_jsonl(pooled, human)
                        seen_humans.add(pair_id)
                    write_jsonl(pooled, edited)
