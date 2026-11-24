"""Choose one CPD threshold on dev scores, then evaluate unchanged on test scores."""

import argparse
import csv
import hashlib
import json
import math
import shlex
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve


def read_scores(path, method):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not {"label", method}.issubset(fields):
            raise ValueError(f"{path}: requires label and {method} columns")
        rows = list(reader)
    labels, scores = [], []
    for line, row in enumerate(rows, start=2):
        if row["label"] not in {"negative", "positive"}:
            raise ValueError(f"{path}:{line}: unexpected label {row['label']!r}")
        try:
            score = float(row[method])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line}: invalid {method} score") from error
        if not math.isfinite(score):
            raise ValueError(f"{path}:{line}: nonfinite {method} score")
        labels.append(int(row["label"] == "positive"))
        scores.append(score)
    if set(labels) != {0, 1}:
        raise ValueError(f"{path}: requires both negative and positive samples")
    return fields, rows, np.array(labels), np.array(scores, dtype=np.float64)


def choose_threshold(labels, scores, selection, target_fpr):
    # Include every distinct score and the +inf all-negative operating point.
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    thresholds[0] = np.inf  # Older sklearn versions use max(score) + 1 instead.
    if selection == "fpr":
        candidates = np.flatnonzero(fpr <= target_fpr)
        # Highest TPR under the FPR cap; then lowest FPR, then highest threshold.
        index = min(candidates, key=lambda i: (-tpr[i], fpr[i], -thresholds[i]))
    else:
        # BA = (1 + TPR - FPR) / 2. All choices use dev labels only.
        index = min(
            range(len(thresholds)),
            key=lambda i: (-(tpr[i] - fpr[i]), fpr[i], -thresholds[i]),
        )
    return float(thresholds[index])


def compute_metrics(labels, scores, threshold):
    predictions = scores >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(predictions & positive))
    fp = int(np.count_nonzero(predictions & ~positive))
    tn = int(np.count_nonzero(~predictions & ~positive))
    fn = int(np.count_nonzero(~predictions & positive))
    fpr, tpr = fp / (fp + tn), tp / (tp + fn)
    return {
        "N_negative": tn + fp,
        "N_positive": tp + fn,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "FPR": fpr,
        "TPR": tpr,
        "Balanced_Accuracy": (tpr + 1 - fpr) / 2,
    }


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_name(path):
    return path.stem.removesuffix("_per_sample").removeprefix("benchmark_2026_")


def write_predictions(path, fields, rows, labels, scores, method, threshold):
    metadata = {"id", "category", "label", "row_number_in_class", "source_index", "tokens", "scored_tokens"}
    selected = [
        field for field in fields
        if field in metadata or field == method or field.startswith("scored_tokens_")
    ]
    output = []
    for row, label, score in zip(rows, labels, scores):
        predicted = int(score >= threshold)
        result = {field: row[field] for field in selected}
        result.update({
            "predicted_label": "positive" if predicted else "negative",
            "correct": int(predicted == label),
            "outcome": ("TP" if label else "FP") if predicted else ("FN" if label else "TN"),
        })
        output.append(result)
    write_csv(path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-file", type=Path,
        required=True, help="Development *_per_sample.csv from the scorer",
    )
    parser.add_argument("--test-dir", type=Path, required=True, help="Directory containing test *_per_sample.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/calibrated"))
    parser.add_argument("--method", default="CPD", help="Score column in test CSVs (default: CPD)")
    parser.add_argument("--dev-method", help="Corresponding dev column (default: same as --method)")
    parser.add_argument("--selection", choices=["fpr", "balanced_accuracy"], default="fpr")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()
    args.dev_method = args.dev_method or args.method
    if not 0 <= args.target_fpr <= 1:
        parser.error("--target-fpr must be between 0 and 1")
    if args.output_dir.resolve() in {args.dev_file.parent.resolve(), args.test_dir.resolve()}:
        parser.error("--output-dir must differ from the input directories")

    _, _, dev_labels, dev_scores = read_scores(args.dev_file, args.dev_method)
    threshold = choose_threshold(dev_labels, dev_scores, args.selection, args.target_fpr)
    dev_metrics = compute_metrics(dev_labels, dev_scores, threshold)
    threshold_value = threshold if math.isfinite(threshold) else "inf"
    calibration = {
        "method": args.method,
        "dev_method": args.dev_method,
        "selection": args.selection,
        "target_dev_FPR": args.target_fpr if args.selection == "fpr" else None,
        "threshold": threshold_value,
        "decision_rule": "score >= threshold => positive",
        "dev_dataset": dataset_name(args.dev_file),
        "dev_file": str(args.dev_file),
    }

    # The threshold is frozen before any test scores or labels are loaded.
    test_paths = sorted(args.test_dir.glob("*_per_sample.csv"))
    if not test_paths:
        parser.error(f"No *_per_sample.csv found in {args.test_dir}")
    loaded_tests, summaries = [], []
    for path in test_paths:
        if path.resolve() == args.dev_file.resolve():
            parser.error("The calibration file cannot also be a test file")
        fields, rows, labels, scores = read_scores(path, args.method)
        metrics = compute_metrics(labels, scores, threshold)
        summaries.append({
            "dataset": dataset_name(path),
            "test_file": str(path),
            **calibration,
            "dev_FPR": dev_metrics["FPR"],
            "dev_TPR": dev_metrics["TPR"],
            "dev_Balanced_Accuracy": dev_metrics["Balanced_Accuracy"],
            **metrics,
        })
        loaded_tests.append((path, fields, rows, labels, scores))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "dev_thresholds.csv", [{**calibration, **dev_metrics}])
    write_csv(args.output_dir / "summary.csv", summaries)
    for path, fields, rows, labels, scores in loaded_tests:
        write_predictions(
            args.output_dir / path.name, fields, rows, labels, scores, args.method, threshold,
        )

    source_files = [Path(__file__).resolve(), args.dev_file, *test_paths]
    manifest = {
        "calibration": {**calibration, **dev_metrics},
        "score_direction": "higher scores indicate positive (AI)",
        "selection_rule": (
            "Maximize dev TPR subject to dev FPR <= target; break ties by lower FPR, "
            "then higher threshold."
            if args.selection == "fpr" else
            "Maximize dev Balanced Accuracy; break ties by lower FPR, then higher threshold."
        ),
        "threshold_candidates": "All distinct dev scores, plus +inf; no interpolation.",
        "test_threshold_policy": "Apply the same frozen dev threshold to every test dataset.",
        "metric_units": "Fractions in [0, 1], not percentages.",
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
        },
    }
    (args.output_dir / "evaluation_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selection_text = (
        f"The threshold maximizes development-set TPR subject to FPR <= {args.target_fpr:.2%}."
        if args.selection == "fpr" else "The threshold maximizes development-set balanced accuracy."
    )
    table = "\n".join(
        f"| {row['dataset']} | {row['FPR']:.4%} | {row['TPR']:.4%} | "
        f"{row['Balanced_Accuracy']:.4%} | {row['TN']} | {row['FP']} | "
        f"{row['FN']} | {row['TP']} |"
        for row in summaries
    )
    command_args = [
        "python", "evaluate_cpd_thresholds.py",
        "--dev-file", str(args.dev_file), "--dev-method", args.dev_method,
        "--test-dir", str(args.test_dir), "--method", args.method,
    ]
    reproduce_command = shlex.join(command_args + [
        "--selection", args.selection, "--target-fpr", str(args.target_fpr),
        "--output-dir", str(args.output_dir),
    ])
    best_ba_command = shlex.join(command_args + [
        "--selection", "balanced_accuracy",
        "--output-dir", str(args.output_dir.with_name(args.output_dir.name + "_best_ba")),
    ])
    one_fpr_command = shlex.join(command_args + [
        "--selection", "fpr", "--target-fpr", "0.01",
        "--output-dir", str(args.output_dir.with_name(args.output_dir.name + "_1fpr")),
    ])
    readme = f"""# Test results at a threshold calibrated on the development set

Calibration uses the `{args.dev_method}` column in `{args.dev_file}`. Evaluation
uses the `{args.method}` column in `{args.test_dir}`. Both columns must contain
scores from the same method.

{selection_text} Candidates are all distinct development scores and positive
infinity. Ties are resolved by choosing the lower FPR, then the higher threshold.

The fixed threshold is **{threshold_value}**. A score at or above the threshold
is classified as positive (AI). A lower score is classified as negative (human).
On the development set, FPR was {dev_metrics['FPR']:.4%}, TPR was
{dev_metrics['TPR']:.4%}, and balanced accuracy (BA) was {dev_metrics['Balanced_Accuracy']:.4%}.

All test sets use the same threshold from `{dataset_name(args.dev_file)}`.
Test labels are used to compute the confusion matrix after classification.
FPR on another dataset can differ from the development target. Calibrating a
separate threshold for each dataset requires independent development scores for each one.

| Test set | Observed FPR | Observed TPR | BA | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## Metrics

- FPR = FP / (FP + TN)
- TPR = TP / (TP + FN)
- Balanced accuracy = (TPR + TNR) / 2 = (TPR + 1 - FPR) / 2

## Output files

- `dev_thresholds.csv` records the threshold, selection rule, and development metrics.
- `summary.csv` reports each test set's confusion counts and metrics. Rates are fractions from 0 to 1.
- `*_per_sample.csv` records each sample's score, predicted label, correctness, and outcome (TP, FP, TN, or FN).
- `evaluation_config.json` records the evaluation rules and SHA-256 hashes of the inputs and evaluation script.

In `{args.test_dir / 'summary.csv'}`, `TPR_at_5FPR` is the highest TPR on
the test ROC curve at FPR <= 5%. The TPR in this report uses the fixed development threshold.

## Reproduce this evaluation from the repository root

```bash
{reproduce_command}
```

### Select the threshold by development-set balanced accuracy

```bash
{best_ba_command}
```

### Use a 1% development FPR target

```bash
{one_fpr_command}
```
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"{args.method}: dev threshold = {threshold_value} ({args.selection})")
    print(
        f"Dev: FPR={dev_metrics['FPR']:.4%}, TPR={dev_metrics['TPR']:.4%}, "
        f"BA={dev_metrics['Balanced_Accuracy']:.4%}"
    )
    for row in summaries:
        print(
            f"{row['dataset']} / {row['method']}: "
            f"FPR={row['FPR']:.4%}, TPR={row['TPR']:.4%}, "
            f"BA={row['Balanced_Accuracy']:.4%}"
        )
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
