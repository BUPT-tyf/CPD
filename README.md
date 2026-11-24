# Context Perturbation Discrepancy (CPD)

**English** | [简体中文](README.zh-CN.md)

Code for the main experiments in *Beyond Fixed-Context Likelihood: Zero-Shot AI-Generated Text Detection via Context Perturbation Discrepancy*. CPD uses the same Falcon-7B model to score each observed token with its full prefix and a shorter prefix. It subtracts the expected log-probability gain at each position, sums the deviations across the document, and normalizes by their reference standard deviation. The experiments cover direct detection on GEN, DetectRL-X, and HART, draft origin on HART, rewriting on GEN and DetectRL-X, and fixed-threshold evaluation.

The main experiment runner is `scripts/run_experiments.py`. From the repository root, preview the scoring command for all three benchmarks before downloading any data or models. [Experiment runner](scripts/run_experiments.py)

```bash
python scripts/run_experiments.py main --dry-run
```

## Rebuild the main table from saved results

The following command uses only the Python standard library. It reads the archived summaries for eight detectors and writes `outputs/main_table.csv`. CPD's macro-average AUROC is **94.80%** across the three datasets, and its macro-average recall at 5% FPR is **79.60%**. The full nine-detector table also uses the separate Single-Fast run described below. [Reference results and column definitions](docs/results.md)

```bash
python report_main_results.py
```

## Set up the scoring environment

The experiments used Linux, Python 3.10.16, PyTorch 2.4.1 with CUDA 12.4, and Transformers 4.51.1. Scoring in BF16 requires an NVIDIA GPU with BF16 support. CPD loads one model. The dual-model baselines load two models at once. The experiments used a 48 GB RTX A6000.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Download the two Falcon checkpoints into `models/`. The weight shards, tokenizer files, and configurations at these revisions match the original experiment files. If you already have the checkpoints in Hugging Face format, pass the directory containing `falcon-7b/` and `falcon-7b-instruct/` to the runner with `--model-dir`. CPD and Single-Fast each require only the first checkpoint. [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) · [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

```bash
huggingface-cli download tiiuae/falcon-7b \
  --revision ec89142b67d748a1865ea4451372db8313ada0d8 --local-dir models/falcon-7b \
  --include '*.json' '*.safetensors' '*.py'
huggingface-cli download tiiuae/falcon-7b-instruct \
  --revision 8782b5c5d8c9290412416618f36a133653e85285 --local-dir models/falcon-7b-instruct \
  --include '*.json' '*.safetensors' '*.py'
```

## Run the experiments from source data

Build the GEN development and test splits, the English DetectRL-X test split, and the HART test split. The build checks each file's SHA-256 hash against the paper inputs. If you already have prepared splits, run `python prepare_data.py verify --data-dir /path/to/benchmark_2026` and pass the same `--data-dir` to the experiment runner. [Data sources and offline preparation](docs/data.md)

```bash
python prepare_data.py build
python scripts/run_experiments.py main --methods CPD --sample-size 2 --output-root outputs/smoke
```

The small run scores two human texts and two AI texts per dataset to check model loading, inference, and output. To generate the full paper table, score the eight standard detectors, run Single-Fast separately, and combine both summaries.

```bash
python scripts/run_experiments.py main
python scripts/run_experiments.py single-fast
python report_main_results.py --main-csv outputs/cpd_results_main/summary.csv \
  --single-fast-csv outputs/cpd_results_single_fast/summary.csv \
  --output outputs/main_table_all_methods.csv
```

Fixed-threshold evaluation uses scores from both `main` and `dev`. The HART and rewriting experiments run independently once their input files are ready. [Experiment settings and output paths](docs/experiments.md)

| Experiment | Commands |
| --- | --- |
| Score GEN development data and transfer thresholds to the test sets | Run `python scripts/run_experiments.py dev`, then `python scripts/run_experiments.py calibrate`. |
| Compare draft origin and final expression on HART | `python scripts/run_experiments.py hart` |
| Evaluate AI edits of human GEN drafts | Run `python prepare_data.py build gen-edit`, then `python scripts/run_experiments.py editing`. |
| Evaluate three rewrites of AI drafts on DetectRL-X | Run `python prepare_data.py build rewriting`, then `python scripts/run_experiments.py rewriting`. |

## Repository layout and validation

| Location | Contents |
| --- | --- |
| `CPD.py`, `cpd_cache.py`, `cpd_paths.py` | Scoring, optional per-sample caching, and output naming |
| `prepare_data.py`, `data_prep/` | Data conversion, GEN split indices, file hashes, and class counts |
| `scripts/run_experiments.py` | Seven experiment commands with common data, model, and output directory options |
| `evaluate_cpd_thresholds.py`, `report_main_results.py` | Threshold selection on the development set and cross-dataset reporting |
| `reference_results/`, `docs/`, `tests/` | Five archived summary tables, reproduction guides, and regression tests |
| `outputs/` | Run configurations, per-sample CSVs, and summaries generated locally and ignored by Git |

Run the 50 regression tests with `python -m unittest discover -s tests` after installing the dependencies. For this release, the four main input files were rebuilt from pinned sources and matched the archived inputs byte for byte. Recomputing metrics from 20,000 archived test documents reproduced all 24 detector–dataset results. A separate Falcon-7B inference check reproduced the archived CPD scores for one human and one AI text on each benchmark.

## Baseline implementations

Higher scores indicate stronger evidence of an AI-origin draft. Loglikelihood, LRR, TempTest, and NTS use Falcon-7B. Binoculars, dual-model Fast-DetectGPT, and Lastde++ use Falcon-7B and Falcon-7B-Instruct in their method-specific roles. [GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) · [DetectLLM](https://github.com/mbzuai-nlp/DetectLLM) · [TempTest](https://github.com/TMKempton/TempTest) · [NTS](https://github.com/Shixuan-Ma/NTS) · [Binoculars](https://github.com/ahans30/Binoculars) · [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) · [Lastde++](https://github.com/TrustMedia-zju/Lastde_Detector)
