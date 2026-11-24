# Data for the main experiments

Run `python prepare_data.py build` from the repository root to prepare the four JSONL files in the first three groups below. All five HART editing comparisons use the same test file. Prepare the optional `gen-edit` and `rewriting` groups when running the GEN and DetectRL-X rewriting experiments.

| Data group | Output files under `benchmark_2026/` | Class counts before scoring |
| --- | --- | --- |
| `gen` | `GEN_dev.jsonl`, `GEN_test.jsonl` | 2,000 H and 2,000 A in each split |
| `detectrlx` | `DetectRLX_english_test_general.jsonl` | 6,000 H and 6,000 A |
| `hart` | `HART_test.jsonl` | 2,000 each of H, A, HA, and AH |
| `gen-edit` | `GEN_edit_test.jsonl` | 936 H, 936 A, and 936 HA in each of three edit-ratio ranges |
| `rewriting` | `AA/DetectRLX_english_test_AA_{polishing,expanding,condensing,all}.jsonl` | Each operation has 6,000 H and 6,000 AA. The pooled file has 6,000 H and 18,000 AA. |

The preparation scripts convert files from the original dataset releases using the split definitions in this repository. For GEN and DetectRL-X, the downloader checks the upstream file hashes and caches the raw files in `~/.cache/cpd/`. Set `CPD_DATA_CACHE` to use another directory. After conversion, the build checks each output's SHA-256 against `data_prep/manifest.json`. On subsequent runs it skips any group whose files already match.

## Data sources and splits

- **GEN** uses revision `a0f143c1ae9c35b684898a5f9c0ab6efb49be486`. The development and direct-detection test splits use disjoint prompt groups. The direct-detection build reads only `human.jsonl` and `ai_generated.jsonl`. It reconstructs both splits in their original sample order using `data_prep/gen_direct_rows.csv`, where each row gives a zero-based index into the source file for its category. The full sampling procedure is in `data_prep/gen.py`. The editing experiment also reads `ai_edited.jsonl` and pairs revisions of the same draft across three edit-ratio ranges using `seed_id`. Its 936 AI controls come from the direct-detection test split. [GEN dataset card](https://huggingface.co/datasets/szyszy/GEN)
- **DetectRL-X** uses revision `ccc9a0a74793dd3448e0da1650b430f5df1b0402`. The converter takes the English pairs from the official test split in the `Binary/` files. The three rewriting operations use the same human controls, so the converter keeps one human text per pair ID when pooling them. Direct detection needs only the general file. [DetectRL-X dataset card](https://huggingface.co/datasets/WUJUNCHAO/DetectRL-X)
- **HART** uses the four official test files under `benchmark/hart` in the truth-mirror repository at commit `3b9055a52a9c84196bc6acde160234c32fc3e3fe`. The four labels follow the authors' annotations of content and expression. H denotes an original human text, A a directly generated AI text, HA an AI-edited human draft, and AH a humanized AI draft. [HART source files](https://github.com/baoguangsheng/truth-mirror/tree/3b9055a52a9c84196bc6acde160234c32fc3e3fe/benchmark/hart)

## Use a local copy

Use `--data-dir` with any build command to set the output directory. To convert previously downloaded source files offline, pass one data group and point `--raw-dir` at its source directory.

```bash
python prepare_data.py build gen --raw-dir /path/to/GEN
python prepare_data.py build detectrlx --raw-dir /path/to/DetectRL-X
python prepare_data.py build hart --raw-dir /path/to/truth-mirror/benchmark/hart
python prepare_data.py build gen-edit --raw-dir /path/to/GEN
python prepare_data.py build rewriting --raw-dir /path/to/DetectRL-X
```

| Dataset | Expected layout under `--raw-dir` |
| --- | --- |
| GEN | `human.jsonl` and `ai_generated.jsonl` at the top level. The editing group also needs `ai_edited.jsonl`. |
| DetectRL-X | A `Binary/` directory containing `binary_general_open.json` for direct detection and `binary_{polishing,expanding,condensing}_open.json` for rewriting. |
| HART | `essay.test.json`, `arxiv.test.json`, `writing.test.json`, and `news.test.json` at the top level. |

If you already have the prepared JSONL files, verify all nine against the manifest with the following command. Verification uses only the Python standard library.

```bash
python prepare_data.py verify gen detectrlx hart gen-edit rewriting --data-dir /path/to/benchmark_2026
```

The scorer reads `generation` as the model input and uses `category` to select the positive and negative classes for each comparison. The `id` field identifies each sample and serves as a key for optional score caching. The local data directory holds the texts and prompts. Run outputs contain the per-sample scores.
