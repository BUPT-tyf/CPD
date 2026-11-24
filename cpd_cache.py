"""Folder-managed, ID-based score storage for CPD.py (no cache hashes)."""

import csv
import json
import math
import os
from pathlib import Path
import warnings


MODEL_SETTING_KEYS = (
    "base_dir", "base_name", "ref_name", "max_length", "precision",
    "attn_impl", "full_logits_device", "disable_last_logits_opt",
)
CSV_METADATA = {
    "id", "source_index", "category", "label", "row_number_in_class",
    "tokens", "scored_tokens", "shuffle_seed", "random_token_seed",
}


def prepare_score_directory(directory, settings):
    """Record model settings only; migrate the old initial-run manifest once."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if "initial_run" not in previous:
            return
        # Keep the original model provenance when cleaning an existing folder.
        settings = previous["initial_run"]
    config = {
        "format_version": 1,
        "configuration_policy": "One folder per scoring configuration; "
                                "use a new folder if settings or sample texts change.",
        "model_settings": {key: settings[key] for key in MODEL_SETTING_KEYS if key in settings},
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ScoreCache:
    """Append complete ID records; the latest record wins after a restart.

    One file belongs to one dataset and one scoring configuration. Each update
    is flushed before returning, so a later interrupted run retains its scores.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        self._needs_compaction = False
        if not self.path.exists():
            return
        file_size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    if handle.tell() == file_size and not line.endswith(b"\n"):
                        warnings.warn(
                            f"Discarding incomplete final cache record: {self.path}",
                            stacklevel=2,
                        )
                        with self.path.open("r+b") as repair:
                            repair.truncate(offset)
                        break
                    raise ValueError(f"Invalid cache JSON at {self.path}:{offset}") from error
                if (not isinstance(record, dict)
                        or not isinstance(record.get("id"), str)
                        or not record["id"].strip()
                        or type(record.get("tokens")) is not int
                        or record["tokens"] < 0
                        or not isinstance(record.get("scores"), dict)):
                    raise ValueError(f"Invalid cache record at {self.path}:{offset}")
                original_record = record
                # Nonfinite or missing scores must be computed again.
                record = dict(record)
                record["scores"] = {
                    name: value for name, value in record["scores"].items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool) and math.isfinite(value)
                }
                # Evaluation labels come from the current dataset, not the cache.
                record = {key: record[key] for key in ("id", "source_index", "tokens", "scores")
                          if key in record}
                if record != original_record or record["id"] in self.records:
                    self._needs_compaction = True
                self.records[record["id"]] = record

    def get(self, sample_id):
        return self.records.get(sample_id)

    def put(self, sample, tokens, scores, overwrite=False):
        """Merge methods for this ID, leaving unrelated IDs and methods intact."""
        previous = self.get(sample["id"])
        if previous is not None and previous["tokens"] != tokens:
            raise ValueError(
                f"Token count changed for ID {sample['id']!r}. "
                "Use a new --score-dir for changed data or scoring settings."
            )
        merged = dict(previous["scores"]) if previous is not None else {}
        for name, score in scores.items():
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"Cannot cache nonfinite score: {sample['id']} / {name}")
            if overwrite or name not in merged:
                merged[name] = value
        record = {
            "id": sample["id"],
            "source_index": sample["source_index"],
            "tokens": int(tokens),
            "scores": merged,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A complete last record without a newline is also a valid input file.
        with self.path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write((json.dumps(record, ensure_ascii=False, allow_nan=False)
                          + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        self.records[sample["id"]] = record
        self._needs_compaction = True

    def compact(self):
        """Keep one current record per ID after a successful dataset run."""
        if not self.path.exists() or not self._needs_compaction:
            return
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for record in self.records.values():
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
            self._needs_compaction = False
        finally:
            temporary.unlink(missing_ok=True)


class ScoreDirectory:
    """Share ID/method lookups across files directly inside one score folder."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.records = {}
        self._files = {}
        self._locations = {}
        self._token_conflicts = set()
        self._changed = set()
        # JSONL records take precedence over imported CSV values. Within each
        # format, newer files override older values for the same ID and method.
        for pattern in ("*.csv", "*_scores.jsonl"):
            paths = [path for path in self.directory.glob(pattern)
                     if path.is_file() and path.resolve().parent == self.directory]
            for path in sorted(paths, key=lambda item: (item.stat().st_mtime_ns, item.name)):
                path = path.resolve()
                if pattern == "*.csv":
                    records = self._read_csv(path)
                else:
                    self._files[path] = ScoreCache(path)
                    records = self._files[path].records.values()
                for record in records:
                    self._merge(record, path)

    @staticmethod
    def _read_csv(path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"id", "tokens"}.issubset(reader.fieldnames or ()):
                return
            methods = [name for name in reader.fieldnames
                       if name not in CSV_METADATA and not name.startswith("scored_tokens_")]
            for row in reader:
                sample_id = row.get("id")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    continue
                try:
                    tokens = int(row["tokens"])
                except (TypeError, ValueError):
                    continue
                if tokens < 0:
                    continue
                scores = {}
                for name in methods:
                    try:
                        value = float(row[name])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        scores[name] = value
                if scores:
                    yield {"id": sample_id, "tokens": tokens, "scores": scores}

    def _merge(self, record, path):
        sample_id = record["id"]
        previous = self.records.get(sample_id)
        if previous is None:
            self.records[sample_id] = {
                "id": sample_id, "tokens": record["tokens"], "scores": dict(record["scores"]),
            }
        else:
            if previous["tokens"] != record["tokens"]:
                self._token_conflicts.add(sample_id)
            previous["scores"].update(record["scores"])
        self._locations.setdefault(sample_id, set()).add(path)

    def get(self, sample_id):
        if sample_id in self._token_conflicts:
            raise ValueError(
                f"Conflicting token counts for ID {sample_id!r} across files in {self.directory}. "
                "Shared IDs must represent the same text and scoring settings; "
                "use separate --score-dir folders otherwise."
            )
        return self.records.get(sample_id)

    def dataset(self, filename):
        path = (self.directory / filename).resolve()
        if path.parent != self.directory:
            raise ValueError("Score files must be directly inside --score-dir")
        if path not in self._files:
            self._files[path] = ScoreCache(path)
        return DatasetScoreCache(self, path)

    def put(self, path, sample, tokens, scores, overwrite=False):
        previous = self.get(sample["id"])
        if previous is not None and previous["tokens"] != tokens:
            raise ValueError(
                f"Token count changed for ID {sample['id']!r}. "
                "Use a new --score-dir for changed data or scoring settings."
            )
        if previous is not None and not overwrite:
            scores = {name: value for name, value in scores.items()
                      if name not in previous["scores"]}
        if not scores:
            return
        self._files[path].put(sample, tokens, scores, overwrite=overwrite)
        self._changed.add(path)
        if overwrite:
            # Keep other JSONL copies coherent after refreshing a shared ID.
            # CSV files are read-only imports; the updated JSONL takes priority.
            for other_path in sorted(self._locations.get(sample["id"], ())):
                if other_path == path or other_path not in self._files:
                    continue
                cache = self._files[other_path]
                record = cache.get(sample["id"])
                replacements = {name: value for name, value in scores.items()
                                if name in record["scores"]}
                if replacements:
                    original_sample = {
                        "id": sample["id"],
                        "source_index": record.get("source_index", sample["source_index"]),
                    }
                    cache.put(original_sample, tokens, replacements, overwrite=True)
                    self._changed.add(other_path)
        self._merge({"id": sample["id"], "tokens": tokens, "scores": scores}, path)

    def compact(self, path):
        for changed_path in sorted(self._changed | {path}):
            self._files[changed_path].compact()
        self._changed.clear()


class DatasetScoreCache:
    """Read from the shared directory and write new scores to this dataset."""

    def __init__(self, directory, path):
        self.directory = directory
        self.path = path

    def get(self, sample_id):
        return self.directory.get(sample_id)

    def put(self, sample, tokens, scores, overwrite=False):
        self.directory.put(self.path, sample, tokens, scores, overwrite=overwrite)

    def compact(self):
        self.directory.compact(self.path)
