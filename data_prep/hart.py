"""Convert the official HART test partition."""
import json
from pathlib import Path
import re

import requests

REVISION = "3b9055a52a9c84196bc6acde160234c32fc3e3fe"
DOMAINS = ("essay", "arxiv", "writing", "news")
CATEGORY_LABEL = {"H": 0, "HA": 0, "AH": 1, "A": 1}


def as_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "positive"
    }


def source_kind(value):
    if value is None:
        return None
    value = str(value).strip().lower().split(":", 1)[0].strip()

    # Same source semantics as the official HART utility code.
    if value in {"human", "humanize"}:
        return "human"
    if value in {"machine", "rephrase"}:
        return "machine"
    return None


def category_from_levels(row):
    keys = ["task_level1", "task_level2", "task_level3"]
    if not all(k in row and row[k] is not None for k in keys):
        return None

    l1, l2, l3 = [as_bool(row[k]) for k in keys]

    if l3 and not l2:
        raise ValueError("Invalid labels: Level 3 is positive but Level 2 is negative")
    if l2 and not l1:
        raise ValueError("Invalid labels: Level 2 is positive but Level 1 is negative")

    if l3:
        return "A"       # Type 3: fully AI-generated
    if l2:
        return "AH"      # Type 2: AI-origin content edited by a human
    if l1:
        return "HA"      # Type 1: human-origin content edited by AI
    return "H"


def category_from_sources(row):
    if "content_source" not in row or "language_source" not in row:
        return None

    content = source_kind(row.get("content_source"))
    language = source_kind(row.get("language_source"))
    if content is None or language is None:
        return None

    mapping = {
        ("human", "human"): "H",
        ("human", "machine"): "HA",
        ("machine", "human"): "AH",
        ("machine", "machine"): "A",
    }
    return mapping[(content, language)]


def category_from_type(row):
    for key in ["type", "type_id", "text_type", "hart_type"]:
        if row.get(key) is None:
            continue
        value = row[key]

        if isinstance(value, (int, float)):
            type_id = int(value)
        else:
            match = re.search(r"([0-3])", str(value).lower())
            if not match:
                continue
            type_id = int(match.group(1))

        return {0: "H", 1: "HA", 2: "AH", 3: "A"}.get(type_id)

    return None


def get_category(row):
    # Prefer the official task-level annotations.
    candidates = [
        category_from_levels(row),
        category_from_sources(row),
        category_from_type(row),
    ]
    candidates = [x for x in candidates if x is not None]

    if not candidates:
        raise ValueError(
            "Cannot determine HART category. Row keys: "
            + ", ".join(sorted(row.keys()))
        )

    if len(set(candidates)) != 1:
        raise ValueError(f"Conflicting HART labels: {candidates}")

    return candidates[0]


def first_value(row, keys, default=""):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return default


def clean_text(value):
    return value.strip() if isinstance(value, str) else (
        str(value).strip() if value is not None else ""
    )


def get_generation(row):
    text = first_value(row, ["generation", "text", "response", "output"], None)
    if text is None:
        raise ValueError(
            "No generation/text field. Row keys: "
            + ", ".join(sorted(row.keys()))
        )
    text = clean_text(text)
    return text if text else None


def get_source_id(row):
    # Use explicit source IDs when available. Raw id is only a fallback;
    # it is not automatically proof that two versions are paired.
    for key in [
        "source_id", "origin_id", "original_id", "parent_id",
        "prompt_id", "base_id", "source_record_id"
    ]:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value), "explicit"

    value = row.get("id")
    if value is not None and str(value).strip():
        return str(value), "raw_id_fallback"

    return "", "missing"


def convert_record(row, domain, source_split, output_split, index):
    category = get_category(row)
    label = CATEGORY_LABEL[category]
    generation = get_generation(row)

    if generation is None:
        return None

    source_id, source_id_kind = get_source_id(row)

    # Validate against official Level-2 label when present.
    if row.get("task_level2") is not None:
        expected_label = int(as_bool(row["task_level2"]))
        if expected_label != label:
            raise ValueError(
                f"Level-2 label mismatch: category={category}, "
                f"mapped={label}, official={expected_label}"
            )

    raw_id = row.get("id", f"{domain}_{source_split}_{index}")

    return {
        "id": f"hart_{domain}_{source_split}_{index:06d}_{category}",
        "prompt": clean_text(first_value(
            row, ["prompt", "task_prompt", "title", "input"], ""
        )),
        "generation": generation,
        "model": clean_text(first_value(
            row,
            ["model", "generation_model", "generator_model", "model_name", "llm"],
            ""
        )),
        "label": label,
        "category": category,
        "dataset": "HART",
        "domain": clean_text(row.get("domain", domain)) or domain,
        "lang": "english",
        "split": output_split,
        "source_split": source_split,
        "raw_id": str(raw_id),
        "source_id": source_id,
        "source_id_kind": source_id_kind,

        # HART-specific fields retained for subgroup and paired analysis.
        "content_source": clean_text(row.get("content_source")),
        "language_source": clean_text(row.get("language_source")),
        "editor_model": clean_text(first_value(
            row,
            ["editor_model", "humanizer_model", "refinement_model", "editing_model"],
            ""
        )),
        "edit_method": clean_text(first_value(
            row,
            ["edit_method", "humanization_method", "humanizer",
             "refinement_method", "method", "operation"],
            ""
        )),
        "task_level1": row.get("task_level1"),
        "task_level2": row.get("task_level2"),
        "task_level3": row.get("task_level3"),
    }


def build(output_dir, raw_dir=None, revision=REVISION):
    split = "test"
    path = output_dir / f"HART_{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for domain in DOMAINS:
            name = f"{domain}.{split}.json"
            if raw_dir is None:
                url = f"https://raw.githubusercontent.com/baoguangsheng/truth-mirror/{revision}/benchmark/hart/{name}"
                response = requests.get(url, timeout=(15, 120))
                response.raise_for_status()
                rows = response.json()
            else:
                with (Path(raw_dir) / name).open(encoding="utf-8") as raw:
                    rows = json.load(raw)
            if not isinstance(rows, list):
                raise ValueError(f"Expected a JSON array in {name}")
            for index, row in enumerate(rows):
                record = convert_record(row, domain, split, split, index)
                if record is not None:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
