"""Cache/control-flow checks; model inference is mocked, no GPU is required."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import sklearn.metrics  # Load before temporarily substituting optional GPU modules.

from cpd_cache import ScoreCache, ScoreDirectory


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def load_cpd(filename="CPD.py"):
    replacements = {}
    if importlib.util.find_spec("torch") is None:
        torch = types.ModuleType("torch")
        torch.inference_mode = lambda: lambda function: function
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        replacements = {
            "torch": torch,
            "torch.nn": types.ModuleType("torch.nn"),
            "torch.nn.functional": types.ModuleType("torch.nn.functional"),
        }
        transformers = types.ModuleType("transformers")
        for name in ("AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig"):
            setattr(transformers, name, Mock(name=name))
        replacements["transformers"] = transformers
    spec = importlib.util.spec_from_file_location(Path(filename).stem + "_under_test", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements) if replacements else contextlib.nullcontext():
        spec.loader.exec_module(module)
    return module


cpd = load_cpd()


class DatasetNameTests(unittest.TestCase):
    def test_absolute_and_relative_benchmark_datasets_share_the_result_prefix(self):
        for dataset in (Path("benchmark_2026/AA/edit.jsonl"),
                        Path("/somewhere/benchmark_2026/AA/edit.jsonl")):
            self.assertEqual(cpd.safe_stem(dataset), "benchmark_2026_AA_edit")
        self.assertEqual(cpd.safe_stem("/somewhere/custom/edit.jsonl"), "custom_edit")

    def test_reject_output_filename_collisions_before_scoring(self):
        with (patch.object(sys, "argv", ["CPD.py", "--datasets",
                                         "benchmark_2026/a.b.jsonl", "benchmark_2026/a_b.jsonl"]),
              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
            cpd.parse_args()


class FakeIDs:
    def __init__(self, length):
        self.length = length

    def numel(self):
        return self.length

    def __getitem__(self, key):
        return self


class FakeScorer:
    ref_model = None

    def __init__(self, offset=0):
        self.offset = offset
        self.tokenized = []
        self.calls = []

    def tokenize(self, text, max_length=None):
        self.tokenized.append(text)
        length = len(text.split())
        return FakeIDs(length if max_length is None else min(length, max_length))

    def evaluate(self, text, input_ids, selected_names):
        self.calls.append((text, list(selected_names)))
        sign = -1 if text.startswith("h ") else 1
        return {name: sign + self.offset + i / 10
                for i, name in enumerate(selected_names)}, input_ids.numel()


class CacheStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "scores.jsonl"
        self.sample = {"id": "h1", "category": "H", "source_index": 9}

    def test_merge_overwrite_restart_and_compaction(self):
        cache = ScoreCache(self.path)
        cache.put(self.sample, 20, {"CPD_k1": 1.2345678901234567})
        cache.put(self.sample, 20, {"CPD_k1": 99, "NTS": 2})
        self.assertEqual(cache.get("h1")["scores"],
                         {"CPD_k1": 1.2345678901234567, "NTS": 2})
        cache.put({**self.sample, "id": "a1", "category": "A"}, 20, {"CPD_k1": 3})
        cache.put(self.sample, 20, {"CPD_k1": 4}, overwrite=True)
        restarted = ScoreCache(self.path)
        self.assertEqual(restarted.get("h1")["scores"], {"CPD_k1": 4, "NTS": 2})
        self.assertEqual(restarted.get("a1")["scores"], {"CPD_k1": 3})
        restarted.compact()
        self.assertEqual(len(self.path.read_text().splitlines()), 2)
        self.assertEqual(ScoreCache(self.path).records, restarted.records)

    def test_partial_final_write_is_recoverable(self):
        cache = ScoreCache(self.path)
        cache.put(self.sample, 20, {"CPD_k1": 1})
        with self.path.open("ab") as handle:
            handle.write(b'{"id": "unfinished"')
        with self.assertWarns(UserWarning):
            restarted = ScoreCache(self.path)
        restarted.put({**self.sample, "id": "a1"}, 20, {"CPD_k1": 2})
        self.assertEqual(set(ScoreCache(self.path).records), {"h1", "a1"})

    def test_missing_final_newline_and_nonfinite_values(self):
        self.path.write_text(json.dumps({"id": "h1", "tokens": 20,
                                        "scores": {"CPD_k1": float("nan"), "NTS": 2}}))
        cache = ScoreCache(self.path)
        self.assertEqual(cache.get("h1")["scores"], {"NTS": 2})
        cache.put(self.sample, 20, {"CPD_k1": 1})
        self.assertEqual(ScoreCache(self.path).get("h1")["scores"], {"NTS": 2, "CPD_k1": 1})
        with self.assertRaises(ValueError):
            cache.put(self.sample, 20, {"NTS": float("inf")}, overwrite=True)
        with self.assertRaises(ValueError):
            cache.put(self.sample, 21, {"NTS": 1})

    def test_corrupt_complete_record_is_not_silently_ignored(self):
        self.path.write_text("invalid\n")
        with self.assertRaises(ValueError):
            ScoreCache(self.path)

    def test_directory_merges_methods_and_imports_csv_with_jsonl_precedence(self):
        folder = self.path.parent
        ScoreCache(folder / "first_scores.jsonl").put(self.sample, 20, {"CPD_k1": 1})
        ScoreCache(folder / "second_scores.jsonl").put(self.sample, 20, {"NTS": 2})
        pd.DataFrame([{"id": "h1", "tokens": 20, "label": "positive", "CPD_k1": 99,
                       "LogLikelihood": -3, "Entropy": float("nan")},
                      {"id": "0001", "tokens": 20, "CPD_k1": 4}]).to_csv(
                          folder / "renamed_results.csv", index=False, encoding="utf-8-sig")
        shared = ScoreDirectory(folder)
        self.assertEqual(shared.get("h1")["scores"],
                         {"CPD_k1": 1, "NTS": 2, "LogLikelihood": -3})
        self.assertEqual(shared.get("0001")["scores"]["CPD_k1"], 4)

    def test_directory_does_not_search_parent_siblings_subfolders_or_external_symlinks(self):
        parent = self.path.parent
        first = parent / "first"
        second = parent / "second"
        nested = first / "nested"
        first.mkdir()
        ScoreCache(parent / "parent_scores.jsonl").put(self.sample, 20, {"CPD_k1": 1})
        ScoreCache(second / "other_scores.jsonl").put(self.sample, 20, {"CPD_k1": 2})
        ScoreCache(nested / "nested_scores.jsonl").put(self.sample, 20, {"CPD_k1": 3})
        (first / "linked_scores.jsonl").symlink_to(parent / "parent_scores.jsonl")
        shared = ScoreDirectory(first)
        self.assertIsNone(shared.get("h1"))
        with self.assertRaises(ValueError):
            shared.dataset("../outside_scores.jsonl")

    def test_directory_overwrite_updates_shared_jsonl_copies_and_keeps_source_positions(self):
        folder = self.path.parent
        first_path = folder / "first_scores.jsonl"
        second_path = folder / "second_scores.jsonl"
        ScoreCache(first_path).put(self.sample, 20, {"CPD_k1": 1})
        ScoreCache(second_path).put({**self.sample, "source_index": 99}, 20,
                                    {"CPD_k1": 1, "NTS": 2})
        shared = ScoreDirectory(folder)
        target = shared.dataset("new_scores.jsonl")
        target.put(self.sample, 20, {"CPD_k1": 99})
        self.assertFalse(target.path.exists())
        self.assertEqual(shared.get("h1")["scores"]["CPD_k1"], 1)
        target.put({**self.sample, "source_index": 100}, 20, {"CPD_k1": 4}, overwrite=True)
        target.compact()
        self.assertEqual(ScoreDirectory(folder).get("h1")["scores"], {"CPD_k1": 4, "NTS": 2})
        self.assertEqual(ScoreCache(first_path).get("h1")["source_index"], 9)
        self.assertEqual(ScoreCache(second_path).get("h1")["source_index"], 99)
        self.assertEqual(ScoreCache(second_path).get("h1")["scores"], {"CPD_k1": 4, "NTS": 2})

    def test_conflicting_token_counts_are_checked_for_the_requested_id(self):
        folder = self.path.parent
        ScoreCache(folder / "first_scores.jsonl").put(self.sample, 20, {"CPD_k1": 1})
        ScoreCache(folder / "second_scores.jsonl").put(self.sample, 21, {"NTS": 2})
        shared = ScoreDirectory(folder)
        self.assertIsNone(shared.get("another_id"))
        with self.assertRaisesRegex(ValueError, "Conflicting token counts"):
            shared.get("h1")


class CPDWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset.jsonl"
        rows = [
            {"id": "h1", "category": "H", "generation": "h short text"},
            {"id": "h2", "category": "H", "generation": "h long text x y"},
            {"id": "a1", "category": "A", "generation": "a short text"},
            {"id": "a2", "category": "A", "generation": "a long text x y"},
            {"id": "ah1", "category": "AH", "generation": "ah short text"},
            {"id": "ah2", "category": "AH", "generation": "ah long text x y"},
        ]
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.score_dir = self.root / "cache"
        self.cache_path = self.score_dir / f"{cpd.safe_stem(self.dataset)}_scores.jsonl"

    def arguments(self, *extra):
        argv = ["CPD.py", "--datasets", str(self.dataset), "--methods", "CPD",
                "--score-dir", str(self.score_dir), "--output-dir", str(self.root / "output"),
                *extra]
        with patch.object(sys, "argv", argv):
            return cpd.parse_args()

    def run_cpd(self, *extra, scorer=None):
        scorer = scorer or FakeScorer()
        args = self.arguments(*extra)
        with (patch.object(cpd, "parse_args", return_value=args),
              patch.object(cpd, "LazyTextScorer", return_value=scorer),
              patch.object(cpd, "tqdm", side_effect=lambda items, **kwargs: items),
              contextlib.redirect_stdout(io.StringIO())):
            cpd.main()
        output = Path(args.output_dir) / f"{cpd.safe_stem(args.datasets[0])}_per_sample.csv"
        return scorer, pd.read_csv(output)

    def another_dataset(self, extra_rows=()):
        path = self.root / "aggregate.jsonl"
        rows = [json.loads(line) for line in self.dataset.read_text().splitlines()]
        rows = [*extra_rows, *reversed(rows)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path, rows

    def test_other_dataset_filename_reuses_scores_and_exports_current_source_positions(self):
        self.run_cpd()
        original = self.cache_path.read_bytes()
        original_time = self.cache_path.stat().st_mtime_ns
        other, rows = self.another_dataset()
        scorer, frame = self.run_cpd("--datasets", str(other))
        self.assertFalse(scorer.tokenized)
        expected_positions = {row["id"]: index for index, row in enumerate(rows)
                              if row["category"] in {"H", "A"}}
        self.assertEqual(frame.set_index("id").source_index.to_dict(), expected_positions)
        self.assertEqual(self.cache_path.read_bytes(), original)
        self.assertEqual(self.cache_path.stat().st_mtime_ns, original_time)
        self.assertFalse((self.score_dir / f"{cpd.safe_stem(other)}_scores.jsonl").exists())

    def test_two_datasets_in_one_run_compute_only_unique_missing_ids(self):
        other, _ = self.another_dataset([
            {"id": "h3", "category": "H", "generation": "h third long text"},
            {"id": "a3", "category": "A", "generation": "a third long text"},
        ])
        scorer, _ = self.run_cpd("--datasets", str(self.dataset), str(other))
        self.assertEqual(len(scorer.calls), 6)
        second_path = self.score_dir / f"{cpd.safe_stem(other)}_scores.jsonl"
        self.assertEqual(set(ScoreCache(second_path).records), {"h3", "a3"})
        reused, frame = self.run_cpd("--datasets", str(other))
        self.assertFalse(reused.tokenized)
        self.assertEqual(set(frame.id), {"h1", "h2", "a1", "a2", "h3", "a3"})

    def test_missing_methods_can_be_merged_from_different_dataset_files(self):
        self.run_cpd()
        other, _ = self.another_dataset()
        additional, _ = self.run_cpd("--datasets", str(other), "--methods", "CPD", "NTS",
                                     scorer=FakeScorer(offset=10))
        self.assertEqual(len(additional.calls), 4)
        self.assertTrue(all(names == ["NTS"] for _, names in additional.calls))
        reused, frame = self.run_cpd("--methods", "CPD", "NTS")
        self.assertFalse(reused.tokenized)
        self.assertEqual(frame.loc[frame.id == "h1", "CPD_k1"].item(), -1)
        self.assertEqual(frame.loc[frame.id == "h1", "NTS"].item(), 9)
        self.assertEqual(set(ScoreCache(self.cache_path).get("h1")["scores"]), {"CPD_k1"})

    def test_renamed_csv_in_score_folder_is_reused_without_rewriting_it(self):
        self.run_cpd()
        output = self.root / "output" / f"{cpd.safe_stem(self.dataset)}_per_sample.csv"
        imported = self.score_dir / "previous_experiment.csv"
        imported.write_bytes(output.read_bytes())
        self.cache_path.unlink()
        csv_before = imported.read_bytes()
        other, _ = self.another_dataset()
        scorer, frame = self.run_cpd("--datasets", str(other))
        self.assertFalse(scorer.tokenized)
        self.assertEqual(set(frame.id), {"h1", "h2", "a1", "a2"})
        self.assertEqual(imported.read_bytes(), csv_before)
        self.assertFalse((self.score_dir / f"{cpd.safe_stem(other)}_scores.jsonl").exists())

    def test_reuses_h_across_tasks_and_keeps_separate_outputs(self):
        first, _ = self.run_cpd("--positive", "AH", "--output-dir", str(self.root / "H_AH"))
        self.assertEqual(len(first.calls), 4)
        manifest = (self.score_dir / "config.json").read_bytes()
        second, frame = self.run_cpd("--output-dir", str(self.root / "H_A"))
        self.assertEqual(len(second.calls), 2)
        self.assertTrue(all(text.startswith("a ") for text, _ in second.calls))
        self.assertEqual(set(frame.id), {"h1", "h2", "a1", "a2"})
        self.assertEqual(frame.source_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(set(ScoreCache(self.cache_path).records),
                         {"h1", "h2", "a1", "a2", "ah1", "ah2"})
        self.assertEqual((self.score_dir / "config.json").read_bytes(), manifest)
        saved_config = json.loads(manifest)
        self.assertNotIn("initial_run", saved_config)
        self.assertEqual(saved_config["model_settings"]["ref_name"], "falcon")
        self.assertEqual(saved_config["model_settings"]["max_length"], 1024)
        for key in ("methods", "contexts", "same_context_start", "negative_categories",
                    "positive_categories", "datasets", "sample_size", "min_tokens_before_truncation",
                    "max_tokens_before_truncation"):
            self.assertNotIn(key, saved_config["model_settings"])
        evaluation = json.loads((self.root / "H_A" / "run_config.json").read_text())
        self.assertEqual(evaluation["methods"], ["CPD"])
        self.assertEqual(evaluation["contexts"], [1])
        self.assertEqual(evaluation["positive_categories"], ["A"])

    def test_legacy_metadata_is_cleaned_without_recomputing_scores(self):
        self.run_cpd()
        original = ScoreCache(self.cache_path).records
        legacy_settings = vars(self.arguments()).copy()
        legacy_settings["base_dir"] = "/original/model/directory"
        (self.score_dir / "config.json").write_text(json.dumps({
            "format_version": 1, "initial_run": legacy_settings,
        }))
        with self.cache_path.open("w") as handle:
            for record in original.values():
                handle.write(json.dumps({**record, "category": "H", "label": "negative"}) + "\n")
        scorer, frame = self.run_cpd()
        self.assertFalse(scorer.tokenized)
        self.assertEqual(ScoreCache(self.cache_path).records, original)
        cleaned = [json.loads(line) for line in self.cache_path.read_text().splitlines()]
        self.assertTrue(all("category" not in row and "label" not in row for row in cleaned))
        self.assertTrue(all(set(row) == {"id", "source_index", "tokens", "scores"}
                            for row in cleaned))
        self.assertEqual(set(frame.category), {"H", "A"})
        model_settings = json.loads((self.score_dir / "config.json").read_text())["model_settings"]
        self.assertEqual(model_settings["base_dir"], "/original/model/directory")

    def test_complete_hit_does_not_load_a_tokenizer_or_model(self):
        self.run_cpd()
        args = self.arguments()
        with (patch.object(cpd, "parse_args", return_value=args),
              patch.object(cpd, "load_model_and_tokenizer", side_effect=AssertionError("model loaded")),
              patch.object(cpd.AutoTokenizer, "from_pretrained", side_effect=AssertionError("tokenizer loaded")),
              patch.object(cpd, "tqdm", side_effect=lambda items, **kwargs: items),
              contextlib.redirect_stdout(io.StringIO())):
            cpd.main()

    def test_missing_methods_are_computed_without_overwriting_existing_scores(self):
        self.run_cpd()
        initial = ScoreCache(self.cache_path).get("h1")["scores"]["CPD_k1"]
        scorer, frame = self.run_cpd("--methods", "CPD", "NTS", scorer=FakeScorer(offset=10))
        self.assertEqual(len(scorer.calls), 4)
        self.assertTrue(all(names == ["NTS"] for _, names in scorer.calls))
        cached = ScoreCache(self.cache_path).get("h1")["scores"]
        self.assertEqual(cached["CPD_k1"], initial)
        self.assertEqual(cached["NTS"], 9)
        self.assertEqual(set(frame.columns) & {"NTS", "CPD_k1"}, {"NTS", "CPD_k1"})

    def test_single_fast_scores_are_cached_separately_from_dual_fast(self):
        self.run_cpd("--methods", "Fast_DetectGPT")
        original = ScoreCache(self.cache_path).get("h1")["scores"]["Fast_DetectGPT"]
        scorer, frame = self.run_cpd("--methods", "Fast_DetectGPT", "Fast_DetectGPT_single",
                                     scorer=FakeScorer(offset=10))
        self.assertEqual(len(scorer.calls), 4)
        self.assertTrue(all(names == ["Fast_DetectGPT_single"] for _, names in scorer.calls))
        self.assertEqual(ScoreCache(self.cache_path).get("h1")["scores"],
                         {"Fast_DetectGPT": original, "Fast_DetectGPT_single": 9})
        self.assertEqual(set(frame.columns) & {"Fast_DetectGPT", "Fast_DetectGPT_single"},
                         {"Fast_DetectGPT", "Fast_DetectGPT_single"})
        reused, frame = self.run_cpd("--methods", "Fast_DetectGPT_single")
        self.assertFalse(reused.tokenized)
        self.assertEqual(frame.loc[frame.id == "h1", "Fast_DetectGPT_single"].item(), 9)
        summary = pd.read_csv(self.root / "output" / "summary.csv")
        self.assertEqual(summary.method.tolist(), ["Fast_DetectGPT_single"])

    def test_overwrite_only_recomputes_selected_ids_and_methods(self):
        self.run_cpd("--methods", "CPD", "NTS", "--positive", "AH")
        self.run_cpd("--methods", "CPD", "NTS")
        before = ScoreCache(self.cache_path)
        old_nts = before.get("h1")["scores"]["NTS"]
        old_ah = before.get("ah1").copy()
        scorer, frame = self.run_cpd("--overwrite-cache", scorer=FakeScorer(offset=10))
        self.assertEqual(len(scorer.calls), 4)
        self.assertTrue(all(names == ["CPD_k1"] for _, names in scorer.calls))
        after = ScoreCache(self.cache_path)
        self.assertEqual(after.get("h1")["scores"], {"CPD_k1": 9, "NTS": old_nts})
        self.assertEqual(after.get("ah1"), old_ah)
        self.assertEqual(frame.loc[frame.id == "h1", "CPD_k1"].item(), 9)

    def test_length_filter_applies_to_hits_and_retains_exact_boundary(self):
        self.run_cpd()
        scorer, frame = self.run_cpd("--min-tokens-before-truncation", "5")
        self.assertFalse(scorer.tokenized)
        self.assertEqual(set(frame.id), {"h2", "a2"})
        self.assertTrue((frame.tokens == 5).all())
        _, empty = self.run_cpd("--min-tokens-before-truncation", "6")
        self.assertTrue(empty.empty)
        self.assertTrue(pd.read_csv(self.root / "output" / "summary.csv").empty)
        self.assertEqual(len(ScoreCache(self.cache_path).records), 4)

    def test_original_length_filter_prevents_new_inference_and_can_be_disabled(self):
        scorer, frame = self.run_cpd("--min-tokens-before-truncation", "5")
        self.assertEqual(len(scorer.calls), 2)
        self.assertEqual(set(frame.id), {"h2", "a2"})
        other, unfiltered = self.run_cpd("--min-tokens-before-truncation", "0")
        self.assertEqual(len(other.calls), 2)
        self.assertEqual(set(unfiltered.id), {"h1", "h2", "a1", "a2"})

    def range_dataset(self):
        rows = [
            {"id": f"{category}_{length}", "category": category,
             "generation": " ".join([category.lower()] + ["x"] * (length - 1))}
            for category in ("H", "A") for length in (299, 300, 500, 501)
        ]
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows

    def test_original_range_filters_inference_and_default_removes_limits(self):
        rows = self.range_dataset()
        scorer, frame = self.run_cpd("--min-tokens-before-truncation", "300",
                                     "--max-tokens-before-truncation", "500")
        expected = {"H_300", "H_500", "A_300", "A_500"}
        self.assertEqual(set(frame.id), expected)
        self.assertEqual(len(scorer.calls), 4)
        self.assertEqual(set(ScoreCache(self.cache_path).records), expected)
        unfiltered, frame = self.run_cpd()
        self.assertEqual(set(frame.id), {row["id"] for row in rows})
        self.assertEqual(len(unfiltered.calls), 4)

    def test_original_range_filters_cache_hits_without_tokenizing(self):
        self.range_dataset()
        self.run_cpd()
        original = self.cache_path.read_bytes()
        scorer, frame = self.run_cpd("--min-tokens-before-truncation", "300",
                                     "--max-tokens-before-truncation", "500")
        self.assertFalse(scorer.tokenized)
        self.assertFalse(scorer.calls)
        self.assertEqual(set(frame.id), {"H_300", "H_500", "A_300", "A_500"})
        self.assertEqual(self.cache_path.read_bytes(), original)
        summary = pd.read_csv(self.root / "output" / "summary.csv")
        self.assertEqual(summary.Skipped_negative.item(), 2)
        self.assertEqual(summary.Skipped_positive.item(), 2)

    def test_original_maximum_rejects_long_text_hidden_by_truncation(self):
        for max_length in (64, 500):
            for overwrite in (False, True):
                for original_length in (299, 300, 499, 500, 501, 1000):
                    with self.subTest(max_length=max_length, overwrite=overwrite,
                                      original_length=original_length):
                        args = self.arguments(
                            "--min-tokens-before-truncation", "300",
                            "--max-tokens-before-truncation", "500",
                            "--max-length", str(max_length),
                            *(["--overwrite-cache"] if overwrite else []),
                        )
                        sample = {"id": "sample", "generation": " ".join(["x"] * original_length)}
                        cache = Mock()
                        cache.get.return_value = {
                            "tokens": min(original_length, max_length), "scores": {"CPD_k1": 2},
                        }
                        scorer = cpd.LazyTextScorer(args)
                        scorer.ref_tokenizer = object()
                        with (patch.object(cpd, "tokenize_once", side_effect=lambda tokenizer, text, limit:
                                           FakeIDs(min(len(text.split()), limit))),
                              patch.object(scorer, "evaluate", side_effect=lambda text, ids, names:
                                           ({name: 3 for name in names}, ids.numel())) as evaluate):
                            scores, tokens, status = cpd.score_sample(
                                sample, args, ["CPD_k1"], cache, scorer)
                        self.assertEqual(tokens, min(original_length, max_length))
                        if 300 <= original_length <= 500:
                            self.assertEqual(status, "overwritten" if overwrite else "hit")
                            self.assertEqual(scores, {"CPD_k1": 3 if overwrite else 2})
                            self.assertEqual(evaluate.call_count, int(overwrite))
                        else:
                            self.assertIsNone(scores)
                            self.assertEqual(status, "short" if original_length < 300 else "long")
                            evaluate.assert_not_called()
                            cache.get.assert_not_called()
                            cache.put.assert_not_called()

    def test_original_minimum_keeps_same_samples_at_all_truncation_lengths(self):
        rows = [
            {"id": f"{category}_{length}", "category": category,
             "generation": " ".join([category.lower()] + ["x"] * (length - 1))}
            for category in ("H", "A") for length in (255, 256, 300)
        ]
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
        expected_ids = {"H_256", "H_300", "A_256", "A_300"}
        for length in (256, 128, 64):
            with self.subTest(length=length):
                args = self.arguments("--min-tokens-before-truncation", "256",
                                      "--max-length", str(length))
                scorer = cpd.LazyTextScorer(args)
                scorer.ref_tokenizer = object()
                scored_lengths = []

                def tokenize(tokenizer, text, max_length):
                    return FakeIDs(min(len(text.split()), max_length))

                def evaluate(text, input_ids, selected_names):
                    scored_lengths.append(input_ids.numel())
                    return {name: 1 for name in selected_names}, input_ids.numel()

                selected_ids = set()
                with (patch.object(cpd, "tokenize_once", side_effect=tokenize),
                      patch.object(scorer, "evaluate", side_effect=evaluate)):
                    for row in rows:
                        scores, tokens, status = cpd.score_sample(
                            row, args, ["CPD_k1"], None, scorer)
                        if scores is not None:
                            selected_ids.add(row["id"])
                            self.assertEqual(tokens, length)
                        else:
                            self.assertEqual(status, "short")
                self.assertEqual(selected_ids, expected_ids)
                self.assertEqual(scored_lengths, [length] * 4)

    def test_original_minimum_is_checked_on_cache_hits_and_overwrites(self):
        options = ("--min-tokens-before-truncation", "5", "--max-length", "3")
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite):
                args = self.arguments(*options, *(["--overwrite-cache"] if overwrite else []))
                sample = {"id": "h1", "generation": "h short text"}
                cache = Mock()
                scorer = FakeScorer()
                scores, tokens, status = cpd.score_sample(sample, args, ["CPD_k1"], cache, scorer)
                self.assertIsNone(scores)
                self.assertEqual((tokens, status), (3, "short"))
                self.assertFalse(scorer.calls)
                cache.get.assert_not_called()

        args = self.arguments(*options)
        cache = Mock()
        cache.get.return_value = {"tokens": 3, "scores": {"CPD_k1": 2}}
        scorer = FakeScorer()
        scores, tokens, status = cpd.score_sample(
            {"id": "h2", "generation": "h long text x y"}, args, ["CPD_k1"], cache, scorer)
        self.assertEqual((scores, tokens, status), ({"CPD_k1": 2}, 3, "hit"))
        self.assertFalse(scorer.calls)

    def test_one_scoring_token_is_sufficient_and_empty_targets_are_skipped(self):
        for context, length in ((1, 1), (1, 2), (3, 3), (3, 4)):
            with self.subTest(context=context, length=length):
                args = self.arguments("--contexts", str(context))
                scorer = FakeScorer()
                sample = {"id": "sample", "generation": " ".join(["x"] * length)}
                scores, tokens, status = cpd.score_sample(
                    sample, args, [f"CPD_k{context}"], None, scorer)
                self.assertEqual(tokens, length)
                if length > context:
                    self.assertIsNotNone(scores)
                    self.assertEqual(status, "computed")
                    self.assertEqual(len(scorer.calls), 1)
                else:
                    self.assertIsNone(scores)
                    self.assertEqual(status, "short")
                    self.assertFalse(scorer.calls)

    def test_new_folder_does_not_read_another_configuration(self):
        self.run_cpd()
        scorer, frame = self.run_cpd("--score-dir", str(self.root / "second_cache"),
                                     "--base-name", "llama2-instruct", scorer=FakeScorer(offset=5))
        self.assertEqual(len(scorer.calls), 4)
        self.assertEqual(frame.loc[frame.id == "h1", "CPD_k1"].item(), 4)
        self.assertEqual(ScoreCache(self.cache_path).get("h1")["scores"]["CPD_k1"], -1)

    def test_nonfinite_recalculation_does_not_destroy_previous_scores(self):
        self.run_cpd()
        previous = self.cache_path.read_text()
        scorer = FakeScorer()
        scorer.evaluate = Mock(return_value=(None, 3))
        _, frame = self.run_cpd("--overwrite-cache", scorer=scorer)
        self.assertTrue(frame.empty)
        self.assertEqual(self.cache_path.read_text(), previous)


class DataAndDispatchTests(unittest.TestCase):
    def test_ids_survive_filtering_sampling_and_leading_zeroes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            rows = [{"id": "0000", "category": "H", "generation": ""},
                    {"id": "0001", "category": "H", "generation": "human"},
                    {"id": "0002", "category": "A", "generation": "machine"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                negative, positive = cpd.load_data(path, return_records=True, require_ids=True)
                legacy = cpd.load_data(path)
            self.assertEqual(negative[0]["id"], "0001")
            self.assertEqual(negative[0]["source_index"], 1)
            self.assertEqual(positive[0]["source_index"], 2)
            self.assertEqual(legacy, (["human"], ["machine"]))
            rows[2]["id"] = "0001"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "unique"):
                cpd.load_data(path, require_ids=True)

    def test_cache_requires_ids_but_uncached_legacy_data_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text('{"label":0,"generation":"human"}\n'
                            '{"label":1,"generation":"machine"}\n')
            with self.assertRaisesRegex(ValueError, "id.*column"):
                cpd.load_data(path, require_ids=True)
            with contextlib.redirect_stdout(io.StringIO()):
                negative, positive = cpd.load_data(path, return_records=True)
            self.assertEqual(negative[0]["id"], "row:0")
            self.assertEqual(positive[0]["id"], "row:1")

    def test_invalid_ids_are_rejected_only_when_caching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            for ids in ((None, "a1"), ("same", "same")):
                with self.subTest(ids=ids):
                    path.write_text(json.dumps({"id": ids[0], "label": 0, "generation": "human"})
                                    + "\n" + json.dumps({"id": ids[1], "label": 1, "generation": "machine"})
                                    + "\n")
                    with self.assertRaises(ValueError):
                        cpd.load_data(path, require_ids=True)
                    with contextlib.redirect_stdout(io.StringIO()):
                        negative, positive = cpd.load_data(path, return_records=True)
                    self.assertEqual(negative[0]["generation"], "human")
                    self.assertEqual(positive[0]["generation"], "machine")

    def test_partial_context_scoring_preserves_shared_target_start(self):
        with patch.object(sys, "argv", ["CPD.py", "--methods", "CPD", "NTS",
                                       "--contexts", "1", "3", "--same-context-start"]):
            args = cpd.parse_args()
        with (patch.object(cpd, "get_full_logits", return_value=object()),
              patch.object(cpd, "context_scores", return_value={"CPD_k1": 7}) as context,
              patch.object(cpd, "single_model_baselines") as single,
              patch.object(cpd, "tokenize_once", side_effect=AssertionError("re-tokenized"))):
            scores, length = cpd.evaluate_text(
                "sample", object(), object(), None, None, args, {},
                selected_names=["CPD_k1"], input_ids=FakeIDs(20))
        self.assertEqual(scores, {"CPD_k1": 7})
        self.assertEqual(length, 20)
        self.assertEqual(context.call_count, 1)
        self.assertEqual(context.call_args.args[3:5], (1, 3))
        single.assert_not_called()

    def test_lazy_scorer_only_loads_dual_model_when_missing_scores_need_it(self):
        with patch.object(sys, "argv", ["CPD.py", "--methods", "CPD", "Fast_DetectGPT_single", "Binoculars"]):
            args = cpd.parse_args()
        tokenizer = Mock()
        tokenizer.get_vocab.return_value = {"word": 0}
        with (patch.object(cpd, "load_model_and_tokenizer",
                           side_effect=[(object(), tokenizer), (object(), tokenizer)]) as loader,
              patch.object(cpd, "last_logits_forward_kwargs", return_value={}),
              patch.object(cpd, "evaluate_text", return_value=({}, 20)),
              contextlib.redirect_stdout(io.StringIO())):
            scorer = cpd.LazyTextScorer(args)
            scorer.evaluate("sample", FakeIDs(20), ["CPD_k1"])
            self.assertEqual(loader.call_count, 1)
            scorer.evaluate("sample", FakeIDs(20), ["Fast_DetectGPT_single"])
            self.assertEqual(loader.call_count, 1)
            scorer.evaluate("sample", FakeIDs(20), ["Binoculars"])
            self.assertEqual(loader.call_count, 2)
            scorer.evaluate("sample", FakeIDs(20), ["Binoculars"])
            self.assertEqual(loader.call_count, 2)

    def test_single_fast_uses_only_the_reference_model(self):
        with patch.object(sys, "argv", ["CPD.py", "--methods", "Fast_DetectGPT_single"]):
            args = cpd.parse_args()
        self.assertTrue(args.use_single)
        self.assertFalse(args.use_dual)
        self.assertFalse(args.use_pair)
        self.assertFalse(args.use_short)
        self.assertEqual(cpd.all_method_names(args), ["Fast_DetectGPT_single"])

    def test_cli_defaults_and_invalid_combinations(self):
        with patch.object(sys, "argv", ["CPD.py"]):
            args = cpd.parse_args()
        self.assertIsNone(args.score_dir)
        self.assertFalse(args.overwrite_cache)
        self.assertEqual(args.min_tokens_before_truncation, 0)
        self.assertEqual(args.max_tokens_before_truncation, 0)
        self.assertNotIn("min_tokens", vars(args))
        self.assertNotIn("min_scored_tokens", vars(args))
        for extra in (["--overwrite-cache"], ["--min-tokens-before-truncation", "-1"],
                      ["--max-tokens-before-truncation", "-1"], ["--max-tokens-before-truncation"],
                      ["--min-tokens-before-truncation", "501", "--max-tokens-before-truncation", "500"],
                      ["--min-tokens-before-truncation"], ["--max-length", "1"],
                      ["--min-tokens", "256"], ["--min-scored-tokens", "8"]):
            with (self.subTest(extra=extra), patch.object(sys, "argv", ["CPD.py", *extra]),
                  contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                cpd.parse_args()

    def test_cli_accepts_original_minimum_at_all_truncation_lengths(self):
        for length in (256, 128, 64):
            with (self.subTest(length=length), patch.object(sys, "argv", [
                    "CPD.py", "--min-tokens-before-truncation", "256",
                    "--max-length", str(length)])):
                args = cpd.parse_args()
                self.assertEqual(args.min_tokens_before_truncation, 256)
                self.assertEqual(args.max_length, length)

    def test_direct_evaluation_checks_original_minimum_then_truncates(self):
        with patch.object(sys, "argv", ["CPD.py", "--min-tokens-before-truncation", "256",
                                       "--max-length", "64"]):
            args = cpd.parse_args()
        for original_length in (255, 256, 300):
            with (self.subTest(original_length=original_length),
                  patch.object(cpd, "tokenize_once", side_effect=lambda tokenizer, text, limit:
                               FakeIDs(min(original_length, limit))),
                  patch.object(cpd, "get_full_logits") as logits):
                scores, length = cpd.evaluate_text(
                    "sample", None, object(), None, None, args, {}, selected_names=[])
                self.assertEqual(length, 64)
                self.assertEqual(scores, None if original_length < 256 else {})
                logits.assert_not_called()

    def test_direct_evaluation_filters_original_range_before_truncating(self):
        for max_length in (64, 500, 1024):
            with patch.object(sys, "argv", [
                    "CPD.py", "--min-tokens-before-truncation", "300",
                    "--max-tokens-before-truncation", "500", "--max-length", str(max_length)]):
                args = cpd.parse_args()
            for original_length in (299, 300, 500, 501):
                with (self.subTest(max_length=max_length, original_length=original_length),
                      patch.object(cpd, "tokenize_once", side_effect=lambda tokenizer, text, limit:
                                   FakeIDs(min(original_length, limit))),
                      patch.object(cpd, "get_full_logits") as logits):
                    scores, length = cpd.evaluate_text(
                        "sample", None, object(), None, None, args, {}, selected_names=[])
                    self.assertEqual(scores, {} if 300 <= original_length <= 500 else None)
                    self.assertEqual(length, min(original_length, max_length))
                    logits.assert_not_called()


@unittest.skipUnless(HAS_TORCH, "PyTorch is needed for numerical checks")
class FastSingleNumericalTests(unittest.TestCase):
    def arguments(self, methods):
        with patch.object(sys, "argv", ["CPD.py", "--methods", *methods]):
            return cpd.parse_args()

    def test_matches_exact_distribution_and_same_model_dual_fast_across_chunks(self):
        import torch

        probabilities = [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.25] * 4]
        targets = [0, 2, 1]
        observed = []
        means = []
        variances = []
        for distribution, target in zip(probabilities, targets):
            logs = [math.log(value) for value in distribution]
            mean = sum(probability * log_value for probability, log_value in zip(distribution, logs))
            observed.append(logs[target])
            means.append(mean)
            variances.append(sum(probability * (log_value - mean) ** 2
                                 for probability, log_value in zip(distribution, logs)))
        expected = (sum(observed) - sum(means)) / math.sqrt(sum(variances))
        logits = torch.tensor(probabilities, dtype=torch.float64).log()
        labels = torch.tensor(targets)
        args = self.arguments(["Fast_DetectGPT_single", "LogLikelihood", "Entropy"])
        for chunk_size in (1, 2, 64):
            with self.subTest(chunk_size=chunk_size):
                actual = cpd.single_model_baselines(
                    logits, labels, set(args.methods), chunk_size, args)
                paired = cpd.paired_baselines(
                    logits, logits, labels, torch.device("cpu"), {"Fast_DetectGPT"}, chunk_size)
                self.assertAlmostEqual(actual["Fast_DetectGPT_single"], expected, places=5)
                self.assertAlmostEqual(actual["Fast_DetectGPT_single"], paired["Fast_DetectGPT"], places=6)
                self.assertAlmostEqual(actual["LogLikelihood"], sum(observed) / len(targets), places=6)
                self.assertAlmostEqual(actual["Entropy"], sum(means) / len(targets), places=6)

    def test_uniform_distribution_returns_finite_zero(self):
        import torch

        args = self.arguments(["Fast_DetectGPT_single"])
        actual = cpd.single_model_baselines(
            torch.zeros(5, 4), torch.zeros(5, dtype=torch.long), set(args.methods), 2, args)
        self.assertEqual(actual["Fast_DetectGPT_single"], 0)

    def test_single_fast_matches_paper_reduction_with_small_variances(self):
        import torch

        # A near-uniform distribution makes the historical variance floor observable.
        logits = torch.tensor([[0., 1e-5, 3e-5, -1e-5]]).repeat(9, 1)
        labels = torch.zeros(9, dtype=torch.long)
        logp = logits.log_softmax(dim=-1)
        probs = logp.exp()
        means = (probs * logp).sum(dim=-1)
        variances = (probs * (logp - means[:, None]).square()).sum(dim=-1)
        expected = ((logp[:, 0] - means).double().sum()
                    / variances.double().sum().clamp_min(1e-12).sqrt()).item()
        self.assertLess(variances.sum().item(), 1e-8)
        args = self.arguments(["Fast_DetectGPT_single"])
        for chunk_size in (1, 3, 64):
            result = cpd.single_model_baselines(logits, labels, set(args.methods), chunk_size, args)
            self.assertEqual(result["Fast_DetectGPT_single"], expected)

    def test_single_and_mixed_runs_forward_each_required_model_once(self):
        import torch

        class FixedModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.register_buffer("fixed_logits", logits)
                self.forward_count = 0

            def forward(self, input_ids, use_cache=False):
                self.forward_count += 1
                return types.SimpleNamespace(logits=self.fixed_logits[None, :input_ids.shape[1]])

        ids = torch.tensor([0, 1, 2, 1, 3, 0, 2, 3, 1, 0])
        rows = torch.tensor([[0.1, 0.8, -0.4, 0.2], [0.4, -0.6, 0.7, 0.2]])
        for methods in (["Fast_DetectGPT_single"], ["Fast_DetectGPT", "Fast_DetectGPT_single"]):
            with self.subTest(methods=methods):
                args = self.arguments(methods)
                reference = FixedModel(rows.repeat(5, 1))
                scoring = FixedModel(rows.flip(-1).repeat(5, 1))
                tokenizer = Mock()
                tokenizer.return_value = {"input_ids": ids[None]}
                tokenizer.get_vocab.return_value = {str(index): index for index in range(4)}
                with (patch.object(cpd, "load_model_and_tokenizer",
                                   side_effect=[(reference, tokenizer), (scoring, tokenizer)]) as loader,
                      contextlib.redirect_stdout(io.StringIO())):
                    scorer = cpd.LazyTextScorer(args)
                    result, length = scorer.evaluate("sample", ids, methods)
                self.assertEqual(reference.forward_count, 1)
                self.assertEqual(scoring.forward_count, int(args.use_dual))
                self.assertEqual(loader.call_count, 1 + int(args.use_dual))
                self.assertEqual(loader.call_args_list[0].args[1], args.ref_name)
                self.assertEqual(set(result), set(methods))
                self.assertEqual(length, len(ids))
                self.assertTrue(all(math.isfinite(value) for value in result.values()))
                expected_single = cpd.paired_baselines(
                    reference.fixed_logits[:-1], reference.fixed_logits[:-1], ids[1:],
                    torch.device("cpu"), {"Fast_DetectGPT"}, args.score_chunk_size)["Fast_DetectGPT"]
                self.assertAlmostEqual(result["Fast_DetectGPT_single"], expected_single, places=6)
                if args.use_dual:
                    expected_dual = cpd.paired_baselines(
                        scoring.fixed_logits[:-1], reference.fixed_logits[:-1], ids[1:],
                        torch.device("cpu"), {"Fast_DetectGPT"}, args.score_chunk_size)["Fast_DetectGPT"]
                    self.assertAlmostEqual(result["Fast_DetectGPT"], expected_dual, places=6)
                    self.assertNotAlmostEqual(result["Fast_DetectGPT_single"], result["Fast_DetectGPT"], places=4)


if __name__ == "__main__":
    unittest.main()
