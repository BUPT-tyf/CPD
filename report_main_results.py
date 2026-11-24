"""Rebuild the direct-detection table from CPD and optional Single-Fast summaries."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
DATASETS = {"GEN_test": "GEN", "DetectRLX_english_test_general": "DetectRL-X", "HART_test": "HART"}
METHODS = ["LogLikelihood", "LRR", "TempTest", "NTS", "Binoculars", "Fast_DetectGPT", "Lastde++", "CPD_k1"]
METRICS = ["AUROC", "TPR_at_5FPR"]


def read_results(path, methods, auc_column=None, method_aliases=None):
    values = {method: {} for method in methods}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if auc_column is None:
            auc_column = "ROC_AUC" if "ROC_AUC" in fields else "AUROC"
        missing = {"method", "dataset", auc_column, "TPR_at_5FPR"} - fields
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            dataset = DATASETS.get(Path(row["dataset"]).stem)
            method = (method_aliases or {}).get(row["method"], row["method"])
            if dataset is None or method not in values:
                continue
            if dataset in values[method]:
                raise ValueError(f"{path}: duplicate {dataset}/{method}")
            scores = [float(row[auc_column]), float(row["TPR_at_5FPR"])]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
                raise ValueError(f"{path}: invalid {dataset}/{method} metrics")
            values[method][dataset] = scores
    for method, datasets in values.items():
        if set(datasets) != set(DATASETS.values()):
            raise ValueError(f"{path}: incomplete benchmark coverage for {method}")
        datasets["Macro"] = [mean(values) for values in zip(*datasets.values())]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=Path, default=ROOT / "reference_results/summary_main.csv")
    parser.add_argument("--single-fast-csv", type=Path, help="Summary from the single-fast experiment (outputs/cpd_results_single_fast/summary.csv)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/main_table.csv")
    args = parser.parse_args()
    values = read_results(args.main_csv, METHODS, "ROC_AUC")
    order = METHODS.copy()
    if args.single_fast_csv:
        values.update(read_results(args.single_fast_csv, ["Single-Fast"],
                                  method_aliases={"Fast_DetectGPT_single": "Single-Fast"}))
        order.insert(4, "Single-Fast")
    columns = [(dataset, index) for dataset in [*DATASETS.values(), "Macro"] for index in range(len(METRICS))]
    header = ["method", *[f"{dataset}_{METRICS[index]}_pct" for dataset, index in columns]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for method in order:
            writer.writerow([method, *[f"{100 * values[method][dataset][index]:.2f}" for dataset, index in columns]])
        if args.single_fast_csv:
            gains = [values["CPD_k1"][d][i] - max(values[m][d][i] for m in order if m != "CPD_k1")
                     for d, i in columns]
            writer.writerow(["Absolute gain (pp)", *[f"{100 * g:.2f}" for g in gains]])
    print(f"Saved {args.output} ({len(order)} detectors).")
    if not args.single_fast_csv:
        print("Supply --single-fast-csv after rerunning Single-Fast to include its row and all-baseline gains.")


if __name__ == "__main__":
    main()
