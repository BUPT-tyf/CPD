"""Path conventions shared by the experiment entry points."""

from pathlib import Path
import re


def safe_stem(dataset):
    """Construct the historical CSV prefix without exposing an absolute path."""
    path = Path(dataset).with_suffix("")
    if "benchmark_2026" in path.parts:
        index = max(i for i, part in enumerate(path.parts) if part == "benchmark_2026")
        path = Path(*path.parts[index:])
    elif path.is_absolute():
        path = Path(path.parent.name) / path.name
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path.as_posix()).strip("_")


def validate_output_names(parser, datasets):
    seen = {}
    for dataset in datasets:
        stem = safe_stem(dataset)
        if stem in seen:
            parser.error(f"{dataset} and {seen[stem]} have the same output prefix {stem!r}; "
                         "run them with separate output directories")
        seen[stem] = dataset
