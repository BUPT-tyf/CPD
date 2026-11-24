"""Recover the paper's ordered GEN development and direct-test samples."""
import csv
from pathlib import Path

from .gen import source_id, stream_rows, to_record, write_jsonl_atomic


def build(output_dir, raw_dir=None):
    manifest = Path(__file__).with_name("gen_direct_rows.csv")
    with manifest.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    selected = {}
    for category, corpus in (("H", "human"), ("A", "ai_generated")):
        wanted = {int(e["raw_row"]) for e in entries if e["category"] == category}
        for index, row in enumerate(stream_rows(corpus, raw_dir)):
            if index in wanted:
                selected[category, index] = row
    for split, name in (("development", "GEN_dev.jsonl"), ("detection_test", "GEN_test.jsonl")):
        records = []
        for entry in entries:
            if entry["split"] == split:
                category = entry["category"]
                index = int(entry["raw_row"])
                row = selected[category, index]
                records.append(to_record(index, row, split, category, source_id(row)))
        write_jsonl_atomic(records, output_dir / name)
