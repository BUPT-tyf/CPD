"""Reproduce the main CPD detection, threshold-transfer and editing experiments."""

import argparse
import csv
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN_DATASETS = ("DetectRLX_english_test_general.jsonl", "GEN_test.jsonl", "HART_test.jsonl")
DETECTORS = ("Lastde++", "Binoculars", "Fast_DetectGPT", "NTS", "TempTest", "LRR", "LogLikelihood", "CPD")
SINGLE_FAST = "Fast_DetectGPT_single"
HART_TASKS = (
    ("cpd_results_H_vs_AH", ["H"], ["AH"]),
    ("cpd_results_HA_vs_A", ["HA"], ["A"]),
    ("cpd_results_HA_vs_AH", ["HA"], ["AH"]),
    ("cpd_results_HA_H_vs_AH_A", ["HA", "H"], ["AH", "A"]),
    ("cpd_results_H_AH_vs_A_HA", ["H", "AH"], ["HA", "A"]),
)


def run(args, script, *options):
    cmd = list(map(str, (sys.executable, "-u", ROOT / script, *options)))
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def score(args, directory, datasets, *, negative=("H",), positive=("A",), min_tokens=0, attention="auto"):
    paths = [args.data_dir / name for name in datasets]
    if not args.dry_run:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared split: {path}")
    options = ["--base-dir", args.model_dir, "--ref-name", "falcon", "--base-name", "falcon-instruct",
               "--datasets", *paths, "--negative-categories", *negative, "--positive-categories", *positive,
               "--methods", *args.methods, "--contexts", 1, "--max-length", 1024, "--precision", "bf16",
               "--seed", 42, "--min-tokens-before-truncation", min_tokens, "--attn-impl", attention,
               "--short-batch-size", 1024, "--output-dir", directory]
    if args.sample_size:
        options.extend(("--sample-size", args.sample_size))
    run(args, "CPD.py", *options)
    if not args.dry_run:
        with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            pairs = {(row["dataset"], row["method"]) for row in csv.DictReader(stream)}
        expected = {(str(path), score_column(method)) for path in paths for method in args.methods}
        if expected != pairs:
            raise ValueError(f"Unexpected dataset/method coverage in {directory / 'summary.csv'}; "
                             f"missing={sorted(expected - pairs)}, extra={sorted(pairs - expected)}")


def score_column(method):
    return "CPD_k1" if method == "CPD" else method


def calibrate(args):
    dev = args.output_root / "cpd_results_GEN_dev"
    if args.dry_run:
        dev_file = dev / f"{args.data_dir.name}_GEN_dev_per_sample.csv"
    else:
        candidates = list(dev.glob("*GEN_dev_per_sample.csv"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one GEN_dev score file in {dev}. Run dev first.")
        dev_file = candidates[0]
    for method in args.methods:
        column = score_column(method)
        run(args, "evaluate_cpd_thresholds.py", "--dev-file", dev_file,
            "--test-dir", args.output_root / "cpd_results_main", "--method", column,
            "--selection", "fpr", "--target-fpr", 0.05,
            "--output-dir", args.output_root / "cpd_results_main_calibrated_GEN_dev" / column)


def compare_groups(args, parent, filename, dataset, tasks, min_tokens):
    rows = []
    for group, negative, positive in tasks:
        directory = parent / group
        score(args, directory, [dataset], negative=negative, positive=positive, min_tokens=min_tokens)
        if not args.dry_run:
            with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows.extend({"source_dir": group, "negative_categories": " ".join(negative),
                             "positive_categories": " ".join(positive), **row} for row in csv.DictReader(stream))
    print(f"Combined groups -> {parent / filename}")
    if not args.dry_run:
        with (parent / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=("main", "single-fast", "dev", "calibrate", "hart", "editing", "rewriting"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "benchmark_2026")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--methods", nargs="+", choices=(*DETECTORS, SINGLE_FAST),
                        help="Select methods within this experiment")
    parser.add_argument("--sample-size", type=int, help="Maximum number of samples per binary class")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands without reading data or running models")
    args = parser.parse_args()
    if args.sample_size is not None and (args.sample_size < 1 or args.recipe == "calibrate"):
        parser.error("--sample-size applies only to scoring runs and must be positive")
    allowed = (SINGLE_FAST,) if args.recipe == "single-fast" else DETECTORS
    if args.methods and set(args.methods) - set(allowed):
        parser.error(f"Methods available for {args.recipe}: {', '.join(allowed)}")
    args.methods = tuple(dict.fromkeys(args.methods or allowed))
    args.data_dir = args.data_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    out = args.output_root = args.output_root.resolve()

    if args.recipe == "main":
        score(args, out / "cpd_results_main", MAIN_DATASETS, min_tokens=9)
    elif args.recipe == "single-fast":
        score(args, out / "cpd_results_single_fast", MAIN_DATASETS, min_tokens=9, attention="eager")
    elif args.recipe == "dev":
        score(args, out / "cpd_results_GEN_dev", ["GEN_dev.jsonl"])
    elif args.recipe == "calibrate":
        calibrate(args)
    elif args.recipe == "hart":
        compare_groups(args, out / "cpd_results_HART", "summary.csv", "HART_test.jsonl", HART_TASKS, 15)
    elif args.recipe == "editing":
        tasks = [(c, [c], ["A"]) for c in ("H", "HA_20_40", "HA_40_60", "HA_60_80")]
        compare_groups(args, out / "cpd_results_GEN_edit", "summary_by_group.csv", "GEN_edit_test.jsonl", tasks, 9)
    elif args.recipe == "rewriting":
        files = [f"AA/DetectRLX_english_test_AA_{name}.jsonl" for name in ("all", "condensing", "expanding", "polishing")]
        score(args, out / "cpd_results_AA", files, positive=["AA"])


if __name__ == "__main__":
    main()
