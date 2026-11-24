# Reproducing the main experiments

Run the commands from the repository root. The runner reads prepared JSONL files from `benchmark_2026/` and local model weights from `models/`, and writes results to `outputs/`. For installation, data preparation, and checkpoint locations, see the [README](../README.md).

## Direct detection and threshold transfer

```bash
python scripts/run_experiments.py main
python scripts/run_experiments.py single-fast
python report_main_results.py \
    --main-csv outputs/cpd_results_main/summary.csv \
    --single-fast-csv outputs/cpd_results_single_fast/summary.csv \
    --output outputs/main_table.csv

python scripts/run_experiments.py dev
python scripts/run_experiments.py calibrate
```

`main` compares unedited human texts (H) with directly generated AI texts (A) on three test sets. It saves per-sample scores, AUROC, and the true positive rate at a 5% false positive rate (TPR@5%) for eight detectors. `single-fast` supplies the ninth detector, Single-Fast, under the method name `Fast_DetectGPT_single`. It uses Falcon-7B for both the reference distribution and scoring, with eager attention. The main table gives each dataset equal weight in the macro-average. The archived CPD AUROCs are 94.18% on GEN, 91.90% on DetectRL-X, and 98.31% on HART.

Scoring uses BF16, a maximum input length of 1,024 tokens, and a random seed of 42. CPD uses a one-token short context. Binoculars, dual-model Fast-DetectGPT, and Lastde++ also use Falcon-7B-Instruct. For each comparison, the scorer removes empty texts, samples both classes to the size of the smaller class, then applies the length and score-validity filters. For the main table, scoring begins at the second token and requires at least nine tokens per input. The GEN development run uses the original minimum input length setting of zero. The shortest development text with a valid archived score had 33 tokens.

`dev` scores the GEN development split with the eight standard detectors. For each detector, `calibrate` selects the development threshold with the highest TPR among those with FPR at or below 5%. It freezes that threshold and reports the observed FPR, TPR, balanced accuracy, and confusion matrix on each test set in `outputs/cpd_results_main_calibrated_GEN_dev/<method>/`. TPR@5% in the main table comes from each test set's own ROC curve.

## Editing and rewriting

```bash
python scripts/run_experiments.py hart
python scripts/run_experiments.py editing
python scripts/run_experiments.py rewriting
```

On HART, HA denotes a human draft edited by AI, and AH denotes an AI draft rewritten to sound human. In the comparisons below, the first group is the negative class and the second is the positive class.

| Command | Comparison | Summary under `outputs/` |
| --- | --- | --- |
| `hart` | H vs AH, HA vs A, HA vs AH, H+HA vs AH+A (draft origin), and H+AH vs HA+A (final expression). Each input must contain at least 15 tokens before truncation. | `cpd_results_HART/summary.csv` |
| `editing` | Each GEN condition (H, HA_20_40, HA_40_60, HA_60_80) compares 936 negative samples with the same 936 AI-generated references (A). | `cpd_results_GEN_edit/summary_by_group.csv` |
| `rewriting` | DetectRL-X compares human texts (H) with AI rewrites (AA) for condensing, expanding, and polishing, both separately and pooled. The pooled run samples 6,000 of the 18,000 rewrites to balance the 6,000 human texts. | `cpd_results_AA/summary.csv` |

Each scoring run saves `run_config.json`, `*_per_sample.csv`, and `summary.csv`. Use these files to check the configuration and sample counts.

## Custom paths and small runs

```bash
python scripts/run_experiments.py main --data-dir /path/to/benchmark_2026 \
    --model-dir /path/to/models --output-root outputs/quick \
    --methods CPD --sample-size 5
python scripts/run_experiments.py main --dry-run
```

`--sample-size 5` selects up to five H texts and five A texts per test set. Use the same output root and `--methods` selection for `main`, `dev`, and `calibrate` so calibration can find the corresponding scores. Use a separate output root for small runs. Preview the underlying commands with `--dry-run` and set `CUDA_VISIBLE_DEVICES` to select a GPU.

All seven experiment commands compute fresh scores for their scoring runs by default. When calling `CPD.py --score-dir` directly, the scorer reuses cached values by sample ID and method column. Choose a new cache directory when changing the model, text, precision, truncation settings, or Single-Fast implementation version.
