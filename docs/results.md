# Check reproduced results against the paper

The five CSV files in `reference_results/` contain aggregate results from the original experiments. They report AUROC, false-positive rate (FPR), and true-positive rate (TPR) on a scale of 0 to 1. Run the commands below from the repository root. To generate new scores, follow the [experiment guide](experiments.md).

## Main detection table

The following command uses only the Python standard library. It reads 24 archived rows, rebuilds the results for eight detectors in Table 1, and writes `outputs/main_table.csv`.

```bash
python report_main_results.py
```

| CPD (k=1) | GEN | DetectRL-X | HART | Macro average across benchmarks |
| --- | ---: | ---: | ---: | ---: |
| AUROC (%) | 94.18 | 91.90 | 98.31 | 94.80 |
| TPR@5% (%) | 80.25 | 65.60 | 92.95 | 79.60 |

Single-Fast, the ninth detector in Table 1, was evaluated in a separate run. Its historical full-precision summary and per-sample scores are unavailable. To reproduce the nine-detector table, run `main` and `single-fast` with the model checkpoints, then combine their summaries. The script computes the macro averages before rounding and reports CPD's gain over the strongest baseline in each column, in percentage points.

```bash
python report_main_results.py \
  --main-csv outputs/cpd_results_main/summary.csv \
  --single-fast-csv outputs/cpd_results_single_fast/summary.csv \
  --output outputs/main_table_all_methods.csv
```

The `single-fast` run names its per-sample score column `Fast_DetectGPT_single`. The report displays this method as Single-Fast. It also accepts older summaries that use `Single-Fast` as the method name and `AUROC` as the metric column.

## Fixed thresholds, HART, and rewriting

| Reference file | Output from a new run | How to compare the results |
| --- | --- | --- |
| `summary_main.csv` | `outputs/cpd_results_main/summary.csv` | Match the dataset filename and method for each comparison of human (H) and directly generated AI (A) texts across the three test sets. |
| `fixed_threshold.csv` | `outputs/cpd_results_main_calibrated_GEN_dev/<method>/summary.csv` | Table 13. Each of the eight methods selects a threshold on GEN development data and applies it to all three test sets. Match each reference row to the output directory for that method. |
| `summary_hart_content.csv` | `outputs/cpd_results_HART/summary.csv` | Table 2. Match the method within each of the five tasks. Remove the `cpd_results_` prefix from the new `source_dir` value to get the archived `task` value. |
| `summary_GEN_edit.csv` | `outputs/cpd_results_GEN_edit/summary_by_group.csv` | Table 17. Use `negative_categories` to select H or one of three levels of AI editing of human drafts (HA). Each condition compares 936 negative texts with the same 936 positive A texts. |
| `summary_AA.csv` | `outputs/cpd_results_AA/summary.csv` | Table 18. Use the dataset filename to identify condensing, expanding, polishing, or the pooled condition. The unedited English control is the DetectRL-X result in the main table. |

In the main table, TPR@5% is the highest TPR on each test set's ROC curve at an FPR of at most 5%. For threshold transfer, apply the full-precision threshold selected on development data and classify a text as positive when `score >= threshold`. For CPD, the GEN development threshold of **2.0579617591114574** yields a development FPR of 4.95%. Applying it to DetectRL-X gives **6.95% FPR and 70.35% TPR**.

This CPU-only check reads the reference CSVs for HART draft origin, heavily edited human drafts on GEN, and polished AI drafts on DetectRL-X.

```bash
python - <<'PY'
import csv
from pathlib import Path

root = Path("reference_results")
checks = [
    ("summary_hart_content.csv", "task", "HA_H_vs_AH_A"),
    ("summary_GEN_edit.csv", "negative_categories", "HA_60_80"),
    ("summary_AA.csv", "dataset", "DetectRLX_english_test_AA_polishing.jsonl"),
]
for name, key, group in checks:
    with (root / name).open(encoding="utf-8", newline="") as handle:
        row, = [r for r in csv.DictReader(handle) if r["method"] == "CPD_k1" and r[key] == group]
    auroc = 100 * float(row["ROC_AUC"])
    tpr = 100 * float(row["TPR_at_5FPR"])
    print(f"{group}  AUROC {auroc:.2f}%  TPR@5% {tpr:.2f}%")
PY
```

The expected CPD AUROC / TPR@5% pairs, in percent and in the order shown, are **89.15 / 62.39**, **77.71 / 52.35**, and **84.59 / 44.93**. Each scoring run writes its per-sample scores locally. Use the reference CSVs to compare the aggregate metrics.
