import os

import argparse
import gc
import hashlib
import json
import inspect
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cpd_paths import safe_stem, validate_output_names

from cpd_cache import ScoreDirectory, prepare_score_directory


MODEL_PATHS = {
    "llama3.1": "Llama-3.1-8B",
    "llama3.1-instruct": "Llama-3.1-8B-Instruct",
    "llama2": "Llama-2-7b-hf",
    "llama2-instruct": "Llama-2-7b-chat-hf",
    "falcon": "falcon-7b",
    "falcon-instruct": "falcon-7b-instruct",
    "qwen2.5": "Qwen2.5-0.5B",
    'gpt-neo': 'gpt-neo-2.7B',
}

PAIRED_METHODS = {"Binoculars", "Fast_DetectGPT"}
DUAL_MODEL_METHODS = PAIRED_METHODS | {"Lastde++"}
SINGLE_CONTEXT_METHODS = {"CR", "CPD", "Block_CPD"}
SINGLE_MODEL_METHODS = {
    "NTS", "TempTest", "Lastde", "LRR", "LogRank",
    "LogLikelihood", "Rank", "Entropy", "Fast_DetectGPT_single",
}
METHOD_CHOICES = [
    "CR", "Block_CPD", "Fast_DetectGPT_single",
    "Binoculars", "Fast_DetectGPT", "CPD", 
    "NTS", "TempTest", "Lastde", "Lastde++", "LRR",
    "LogRank", "LogLikelihood", "Rank", "Entropy",
]
DEFAULT_METHODS = [
    "Lastde++", "Binoculars", "Fast_DetectGPT",
    "NTS","TempTest", "LRR", "LogLikelihood", 
    "CPD",
]

DEFAULT_DATASETS = [
    "benchmark_2026/GEN_dev.jsonl",
]


def parse_args():
    parser = argparse.ArgumentParser(description="CPD and zero-shot detection baselines",
                                     allow_abbrev=False)
    parser.add_argument("--base-dir", default=os.environ.get("MODEL_DIR", "models"), help="Local model parent dir")
    parser.add_argument("--base-name", default="falcon-instruct")
    parser.add_argument("--ref-name", default="falcon")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Random samples per binary class per dataset; None or >= available means all")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument(
        "--negative-categories", "--negative",
        nargs="+", type=lambda x: x.strip().upper(), default=["H"],
        metavar="CATEGORY",
        help="One or more categories treated as negative (default: H)."
    )
    parser.add_argument(
        "--positive-categories", "--positive",
        nargs="+", type=lambda x: x.strip().upper(), default=["A"],
        metavar="CATEGORY",
        help="One or more categories treated as positive (default: A)."
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHOD_CHOICES, default=DEFAULT_METHODS.copy(),
        help="Choose detectors; Fast_DetectGPT_single uses only --ref-name for "
             "sampling and scoring. CR/CPD/Block_CPD run at each --contexts value.",
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[1],
                        help="Short-context lengths (default: 1)")
    parser.add_argument(
        "--same-context-start",
        action="store_true",
        help=("Ablation option: score every context length from the same target "
              "position (max context length). By default, each k starts at its "
              "earliest valid target position.")
    )
    parser.add_argument("--block-batch-size", type=int, default=64,
                        help="Number of k+1-token blocks per forward; auto-shrinks on OOM")
    parser.add_argument("--short-batch-size", type=int, default=1024,
                        help="Sliding-window batch size. Try 128, 512, 1024. "
                             "Halves automatically on CUDA OOM.")
    parser.add_argument("--precision", choices=["int8", "bf16", "fp16", "nf4"], default="bf16",
                        help="int8 saves model VRAM; bf16/fp16 may run faster if VRAM allows")
    parser.add_argument("--attn-impl", choices=["auto", "sdpa", "eager", "flash_attention_2"],
                        default="auto", help="Attention backend; auto uses model default")
    parser.add_argument("--full-logits-device", choices=["gpu", "cpu"], default="gpu",
                        help="Keep full-context logits on GPU for speed (CPU saves VRAM)")
    parser.add_argument("--disable-last-logits-opt", action="store_true",
                        help="Do not request final-position logits only even if supported")
    parser.add_argument("--score-chunk-size", type=int, default=64,
                        help="Chunks for paired-model baseline calculations")
    parser.add_argument("--nts-low-temp", type=float, default=0.7)
    parser.add_argument("--nts-high-temp", type=float, default=1.4)
    parser.add_argument("--temptest-temp", type=float, default=0.8,
                        help="Assumed generation temperature, 0 < T < 1")
    parser.add_argument("--lastde-s", type=int, default=3)
    parser.add_argument("--lastde-scales", type=int, default=5)
    parser.add_argument("--lastde-bins-factor", type=int, default=10)
    parser.add_argument("--lastdepp-s", type=int, default=4)
    parser.add_argument("--lastdepp-scales", type=int, default=10,
                        help="Official README uses 10; ICLR paper text mentions 15")
    parser.add_argument("--lastdepp-bins-factor", type=int, default=8)
    parser.add_argument("--lastdepp-samples", type=int, default=100,
                        help="Monte Carlo samples per text (ICLR default 100)")
    parser.add_argument("--min-tokens-before-truncation", type=int, default=0, metavar="N",
                        help="Discard original texts with fewer than N tokens before "
                             "--max-length truncation; 0 disables this filter (default: 0)")
    parser.add_argument("--max-tokens-before-truncation", type=int, default=0, metavar="N",
                        help="Discard original texts with more than N tokens before "
                             "--max-length truncation; 0 disables this filter (default: 0)")
    parser.add_argument("--score-dir", default=None,
                        help="Search score JSONL/CSV files in this folder by ID and method; "
                             "omitted disables caching. "
                             "Use a new folder when scoring settings or sample texts change.")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Recompute selected samples/methods and replace their cached "
                             "scores; other IDs and methods are preserved")
    parser.add_argument("--output-dir", default="outputs/main")
    args = parser.parse_args()
    validate_output_names(parser, args.datasets)

    args.contexts = sorted(set(args.contexts))
    args.methods = list(dict.fromkeys(args.methods))
    args.negative_categories = list(dict.fromkeys(args.negative_categories))
    args.positive_categories = list(dict.fromkeys(args.positive_categories))
    if not args.negative_categories or not args.positive_categories:
        parser.error("Both negative and positive category lists must be non-empty")
    if set(args.negative_categories) & set(args.positive_categories):
        parser.error("Negative and positive categories must not overlap")

    args.use_pair = bool(PAIRED_METHODS.intersection(args.methods))
    args.use_dual = bool(DUAL_MODEL_METHODS.intersection(args.methods))
    args.use_short = bool(SINGLE_CONTEXT_METHODS.intersection(args.methods))
    args.use_single = bool(SINGLE_MODEL_METHODS.intersection(args.methods))

    if not args.contexts or min(args.contexts) < 1:
        parser.error("--contexts must contain positive integers")
    if args.use_short and args.max_length <= max(args.contexts):
        parser.error("--max-length must be greater than max(--contexts)")
    if args.short_batch_size < 1 or args.block_batch_size < 1 or args.score_chunk_size < 1:
        parser.error("Batch sizes must be positive")
    if not 0 < args.nts_low_temp < args.nts_high_temp:
        parser.error("NTS temperatures must satisfy 0 < low < high")
    if not 0 < args.temptest_temp < 1:
        parser.error("--temptest-temp must be in (0,1)")
    for name in ("lastde_s", "lastde_scales", "lastde_bins_factor",
                 "lastdepp_s", "lastdepp_scales", "lastdepp_bins_factor",
                 "lastdepp_samples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if args.min_tokens_before_truncation < 0:
        parser.error("--min-tokens-before-truncation must be non-negative")
    if args.max_tokens_before_truncation < 0:
        parser.error("--max-tokens-before-truncation must be non-negative")
    if (args.max_tokens_before_truncation
            and args.min_tokens_before_truncation > args.max_tokens_before_truncation):
        parser.error("--min-tokens-before-truncation must not exceed --max-tokens-before-truncation")
    if args.overwrite_cache and not args.score_dir:
        parser.error("--overwrite-cache requires --score-dir")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be a positive integer, or omitted for all")
    return args


def get_model_path(base_dir, name):
    if os.path.isdir(name):
        return name
    return os.path.join(base_dir, MODEL_PATHS.get(name, name))


def load_model_and_tokenizer(base_dir, name, precision="int8", attn_impl="auto"):
    path = get_model_path(base_dir, name)
    print(f"Loading {name}: {path} | precision={precision} | attention={attn_impl}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This setup needs an NVIDIA CUDA GPU")

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16. Use --precision fp16/int8")
    dtype = (torch.bfloat16 if precision == "bf16" else torch.float16)
    if precision == "int8" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    model_kwargs = dict(
        device_map={"": 0},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if precision == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
        )
    if precision == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    if attn_impl != "auto":
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, **model_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def last_logits_forward_kwargs(model, disabled=False):
    """Request last-position logits where the architecture supports it."""
    if disabled:
        return {}
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    if "logits_to_keep" in parameters:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {"num_logits_to_keep": 1}
    return {}


def load_data(
    json_file,
    sample_size=None,
    seed=42,
    negative=("H",),
    positive=("A",),
    return_records=False,
    require_ids=False,
):
    """Balanced sampling for a configurable negative-vs-positive task."""
    df = pd.read_json(json_file, lines=True, encoding="utf-8", dtype={"id": "string"})

    if "generation" not in df.columns:
        raise ValueError("Missing 'generation' column")

    if "category" in df.columns:
        categories = df["category"].astype("string").str.strip().str.upper()
    elif "label" in df.columns:
        # Numeric labels retain the original mapping; string labels such as
        # H/A/HA/AA are also accepted directly.
        raw_labels = df["label"].astype("string").str.strip().str.upper()
        numeric_labels = pd.to_numeric(df["label"], errors="coerce")
        mapped_labels = numeric_labels.map({0: "H", 1: "A", 2: "AH"})
        categories = mapped_labels.fillna(raw_labels.where(numeric_labels.isna()))
    else:
        raise ValueError("Missing 'category' or 'label' column")

    df = df.copy()
    df["_category"] = categories
    df["source_index"] = np.arange(len(df))

    negative = tuple(str(x).strip().upper() for x in negative)
    positive = tuple(str(x).strip().upper() for x in positive)

    if not negative or not positive:
        raise ValueError("Negative and positive categories must both be non-empty")
    if set(negative) & set(positive):
        raise ValueError("Positive and negative categories overlap")

    valid = df["generation"].map(
        lambda x: isinstance(x, str) and bool(x.strip())
    )
    df = df.loc[valid]

    if "id" not in df:
        if require_ids:
            raise ValueError(f"{json_file}: --score-dir requires an 'id' column")
        df["id"] = df["source_index"].map(lambda index: f"row:{index}")
    else:
        invalid_ids = df["id"].isna() | df["id"].str.strip().eq("")
        if invalid_ids.any():
            if require_ids:
                raise ValueError(f"{json_file}: sample IDs must not be missing or empty")
            df.loc[invalid_ids, "id"] = df.loc[invalid_ids, "source_index"].map(
                lambda index: f"row:{index}")
    if require_ids and df["id"].duplicated().any():
        raise ValueError(f"{json_file}: sample IDs must be unique within the dataset")

    available_categories = set(df["_category"].dropna().unique())
    requested_categories = set(negative) | set(positive)
    missing_categories = requested_categories - available_categories
    if missing_categories:
        raise ValueError(
            f"Requested categories not found in {json_file}: "
            f"{sorted(missing_categories)}. "
            f"Available categories: {sorted(available_categories)}"
        )

    columns = ["id", "source_index", "_category", "generation"]
    items = df[columns].rename(columns={"_category": "category"})
    negative_items = items.loc[df["_category"].isin(negative)].to_dict("records")
    positive_items = items.loc[df["_category"].isin(positive)].to_dict("records")

    n_neg = len(negative_items)
    n_pos = len(positive_items)

    target = min(n_neg, n_pos)

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        target = min(target, sample_size)

    if target == 0:
        raise ValueError(
            f"No valid samples: negative={n_neg}, positive={n_pos}"
        )

    rng = np.random.default_rng(seed)

    def sample(items):
        if len(items) <= target:
            return items

        indices = np.sort(
            rng.choice(len(items), size=target, replace=False)
        )
        return [items[i] for i in indices]

    negative_data = sample(negative_items)
    positive_data = sample(positive_items)

    print(
        f"Task: {negative} vs {positive}\n"
        f"Available: Negative={n_neg}, Positive={n_pos}\n"
        f"Selected: Negative={len(negative_data)}, "
        f"Positive={len(positive_data)}\n"
        f"Seed={seed}",
        flush=True
    )

    if return_records:
        return negative_data, positive_data
    return ([item["generation"] for item in negative_data],
            [item["generation"] for item in positive_data])


def tokenize_once(tokenizer, text, max_length):
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=max_length, return_tensors="pt")["input_ids"][0]
    return ids.long()


@torch.inference_mode()
def get_full_logits(model, input_ids, placement="gpu"):
    """Logits at position t-1 predict token x[t]."""
    dev = next(model.parameters()).device
    outputs = model(input_ids=input_ids.unsqueeze(0).to(dev), use_cache=False)
    logits = outputs.logits[0, :-1]
    if placement == "cpu":
        logits = logits.cpu()
    del outputs
    return logits


@torch.inference_mode()
def paired_baselines(score_logits, ref_logits, labels, device, required, chunk_size=64):
    """Only compute requested paired-model baselines."""
    need_bino = "Binoculars" in required
    need_fast = "Fast_DetectGPT" in required
    if not (need_bino or need_fast):
        return {}
    if score_logits.shape != ref_logits.shape:
        raise ValueError("Paired baselines require equal sequence and vocab dimensions")

    sum_obs_ce = 0.0
    sum_exp_ce = 0.0
    fast_numerator = 0.0
    fast_variance = 0.0
    for i in range(0, labels.numel(), chunk_size):
        j = min(i + chunk_size, labels.numel())
        s = score_logits[i:j].to(device=device, dtype=torch.float32)
        r = ref_logits[i:j].to(device=device, dtype=torch.float32)
        y = labels[i:j].to(device)
        log_s = F.log_softmax(s, dim=-1)
        p_r = F.softmax(r, dim=-1)
        token_ll = log_s.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        expected_ll = (p_r * log_s).sum(-1)
        if need_bino:
            sum_obs_ce += -token_ll.sum().item()
            sum_exp_ce += -expected_ll.sum().item()
        if need_fast:
            token_var = (p_r * (log_s - expected_ll.unsqueeze(-1)).square()).sum(-1)
            fast_numerator += (token_ll - expected_ll).sum().item()
            fast_variance += token_var.sum().item()

    result = {}
    if need_bino:
        result["Binoculars"] = -sum_obs_ce / max(sum_exp_ce, 1e-8)
    if need_fast:
        result["Fast_DetectGPT"] = fast_numerator / np.sqrt(max(fast_variance, 1e-8))
    return result


# Single-model baselines: larger scores indicate more likely AI.

def score_loglikelihood(token_ll):
    return float(np.mean(token_ll))


def score_rank(token_rank):
    return float(-np.mean(token_rank))


def score_logrank(token_rank):
    return float(-np.log(token_rank).mean())


def score_lrr(token_ll, token_rank):
    # DetectLLM-LRR: -mean(log p) / mean(log rank).
    denom = np.log(token_rank).mean()
    return float(-np.mean(token_ll) / max(denom, 1e-10))


def score_entropy(token_entropy):
    # Keep entropy as a *negative* quantity, so large -> likely machine.
    return float(-np.mean(token_entropy))


def score_nts(ts, expectation, std):
    # Official NTS: average token-wise standard deviation in the denominator.
    return float((ts - expectation) / max(std, 1e-12))


def score_temptest(mean_log_norm, mean_ll, temperature):
    # Paper's TempTest is negative for temperature-sampled machine text.
    # Negate it to conform to the existing higher-is-machine ROC convention.
    return float((1.0 / temperature - 1.0) * mean_ll - mean_log_norm)


def diversity_entropy_multiscale(tps, window, scales, bins_factor):
    """Multiscale diversity entropy of token log-probabilities."""
    tps = np.asarray(tps, dtype=np.float64)
    n = len(tps)
    bins = bins_factor * n
    result = []
    for tau in range(1, scales + 1):
        if n - tau - window + 1 < 1:
            return None
        coarse = np.convolve(tps, np.ones(tau, dtype=np.float64) / tau, mode="valid")
        segments = np.lib.stride_tricks.sliding_window_view(coarse, window)
        if len(segments) < 2:
            return None
        left, right = segments[:-1], segments[1:]
        numer = np.einsum("ij,ij->i", left, right)
        denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cos = np.clip(numer / np.maximum(denom, 1e-15), -1.0, 1.0)
        indexes = np.minimum(((cos + 1.0) * (bins / 2.0)).astype(np.int64), bins - 1)
        hist = np.bincount(indexes, minlength=bins)
        prob = hist[hist > 0].astype(np.float64) / len(cos)
        result.append(float(-(prob * np.log(prob)).sum() / math.log(bins)))
    return np.asarray(result, dtype=np.float64)


def score_lastde(tps, window=3, scales=5, bins_factor=10):
    de = diversity_entropy_multiscale(tps, window, scales, bins_factor)
    if de is None:
        return float("nan")
    # Match official PyTorch torch.std default: sample standard deviation.
    std = de.std(ddof=1)
    return float(np.mean(tps) / max(std, 1e-10))


def score_lastdepp(observed_tps, sampled_tps, window=4, scales=10, bins_factor=8):
    """Z-score Lastde against sampled token log-probability sequences."""
    observed = score_lastde(observed_tps, window, scales, bins_factor)
    sample_scores = np.asarray([
        score_lastde(seq, window, scales, bins_factor)
        for seq in sampled_tps
    ], dtype=np.float64)
    if not np.isfinite(observed) or not np.isfinite(sample_scores).all():
        return float("nan")
    # Match official PyTorch torch.std default: sample standard deviation.
    return float((observed - sample_scores.mean()) / max(sample_scores.std(ddof=1), 1e-10))


@torch.inference_mode()
def single_model_baselines(full_logits, labels, required, chunk_size, args):
    """Share one chunked logits traversal across single-model detectors."""
    chosen = SINGLE_MODEL_METHODS & set(required)
    if not chosen:
        return {}
    need_rank = bool(chosen & {"Rank", "LogRank", "LRR"})
    need_tps = bool(chosen & {"LogLikelihood", "LRR", "Lastde"})
    need_entropy = "Entropy" in chosen
    need_nts = "NTS" in chosen
    need_temptest = "TempTest" in chosen
    need_fast_single = "Fast_DetectGPT_single" in chosen

    length = labels.numel()
    dev = full_logits.device
    token_ll = np.empty(length, dtype=np.float64) if need_tps else None
    token_rank = np.empty(length, dtype=np.float64) if need_rank else None
    token_entropy = np.empty(length, dtype=np.float64) if need_entropy else None

    nts_ts_low_sum = nts_ts_high_sum = 0.0
    nts_expectation_sum = nts_std_sum = 0.0
    nts_token_count = 0
    temptest_norm_sum = temptest_ll_sum = 0.0
    if need_fast_single:
        fast_residuals = torch.empty(length, device=dev, dtype=torch.float64)
        fast_variances = torch.empty_like(fast_residuals)

    for i in range(0, length, chunk_size):
        j = min(length, i + chunk_size)
        logits = full_logits[i:j].float()
        y = labels[i:j].to(dev)
        log_p = F.log_softmax(logits, dim=-1)
        observed = log_p.gather(1, y[:, None]).squeeze(1)
        if need_tps:
            token_ll[i:j] = observed.cpu().double().numpy()
        if need_rank:
            # Exact rank, without sorting the full vocabulary.
            y_logits = logits.gather(1, y[:, None])
            token_rank[i:j] = (logits > y_logits).sum(dim=1).add(1).double().cpu().numpy()
        if need_entropy:
            token_entropy[i:j] = (-(log_p.exp() * log_p).sum(1)).double().cpu().numpy()
        if need_fast_single:
            # The Single-Fast paper run computes token moments in FP32,
            # then sums over all prediction positions in FP64.
            probs = log_p.exp()
            expected_ll = (probs * log_p).sum(dim=-1)
            token_variance = (probs * (log_p - expected_ll[:, None]).square()).sum(dim=-1)
            fast_residuals[i:j] = observed - expected_ll
            fast_variances[i:j] = token_variance.clamp_min(0)
            del probs, expected_ll, token_variance
        if need_nts:
            # Temperature sensitivity of the observed token log-probabilities.
            low_lp = F.log_softmax(logits / args.nts_low_temp, dim=-1)
            high_lp = F.log_softmax(logits / args.nts_high_temp, dim=-1)
            nts_ts_low_sum += low_lp.gather(1, y[:, None]).double().sum().item()
            nts_ts_high_sum += high_lp.gather(1, y[:, None]).double().sum().item()

            # Official NTS moments:
            # p is the original (T=1) distribution, not a temperature-scaled one.
            probs = log_p.exp()
            delta = low_lp - high_lp
            token_expectation = (probs * delta).sum(dim=-1)
            token_second_moment = (probs * delta.square()).sum(dim=-1)
            token_variance = (
                token_second_moment - token_expectation.square()
            ).clamp_min(0.0)
            token_std = token_variance.sqrt()

            # Average token-wise expectation and token-wise standard deviation.
            nts_expectation_sum += token_expectation.double().sum().item()
            nts_std_sum += token_std.double().sum().item()
            nts_token_count += y.numel()
            del probs, delta, token_expectation, token_second_moment
            del token_variance, token_std, low_lp, high_lp
        if need_temptest:
            t = args.temptest_temp
            temptest_norm_sum += torch.logsumexp(log_p / t, dim=-1).double().sum().item()
            temptest_ll_sum += observed.double().sum().item()
        del logits, log_p, observed

    scores = {}
    if need_fast_single:
        scores["Fast_DetectGPT_single"] = (
            fast_residuals.sum() / fast_variances.sum().clamp_min(1e-12).sqrt()
        ).item()
    if "LogLikelihood" in chosen:
        scores["LogLikelihood"] = score_loglikelihood(token_ll)
    if "Rank" in chosen:
        scores["Rank"] = score_rank(token_rank)
    if "LogRank" in chosen:
        scores["LogRank"] = score_logrank(token_rank)
    if "LRR" in chosen:
        scores["LRR"] = score_lrr(token_ll, token_rank)
    if "Entropy" in chosen:
        scores["Entropy"] = score_entropy(token_entropy)
    if "NTS" in chosen:
        expectation = nts_expectation_sum / nts_token_count
        std = nts_std_sum / nts_token_count
        ts = abs(nts_ts_low_sum / length - nts_ts_high_sum / length)
        scores["NTS"] = score_nts(ts, expectation, std)
    if "TempTest" in chosen:
        scores["TempTest"] = score_temptest(
            temptest_norm_sum / length, temptest_ll_sum / length,
            args.temptest_temp)
    if "Lastde" in chosen:
        scores["Lastde"] = score_lastde(
            token_ll, args.lastde_s, args.lastde_scales, args.lastde_bins_factor)
    return scores


@torch.inference_mode()
def lastdepp_dual_baseline(sampling_logits, scoring_logits, labels,
                            chunk_size, args, input_ids):
    """Base samples tokens from original prefixes; Instruct scores the draws."""
    if sampling_logits.shape != scoring_logits.shape:
        raise ValueError("Lastde++ requires identical sampling/scoring logits shape and vocabulary")
    n = labels.numel()
    if sampling_logits.shape[0] != n:
        raise ValueError("Lastde++ labels and logit sequence lengths do not match")
    dev = sampling_logits.device
    observed_tps = np.empty(n, dtype=np.float64)
    sampled_tps = np.empty((args.lastdepp_samples, n), dtype=np.float64)

    payload = input_ids.contiguous().cpu().numpy().tobytes()
    signature = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    generator = torch.Generator(device=dev).manual_seed(
        (args.seed + signature) % (2**63 - 1))

    for i in range(0, n, chunk_size):
        j = min(n, i + chunk_size)
        sample_logits = sampling_logits[i:j].to(dev, dtype=torch.float32)
        sample_probs = F.softmax(sample_logits, dim=-1)
        draws = torch.multinomial(sample_probs, args.lastdepp_samples,
                                  replacement=True, generator=generator)

        log_score = F.log_softmax(scoring_logits[i:j].to(dev, dtype=torch.float32),
                                  dim=-1)
        target = labels[i:j].to(dev)
        observed_tps[i:j] = log_score.gather(1, target[:, None]).squeeze(1).cpu().double().numpy()
        sampled_tps[:, i:j] = log_score.gather(1, draws).cpu().double().numpy().T

    return score_lastdepp(
        observed_tps, sampled_tps, args.lastdepp_s, args.lastdepp_scales,
        args.lastdepp_bins_factor)


@torch.inference_mode()
def cpd_token_terms(logits_full, logits_short, targets, need_observed=False):
    if logits_full.shape != logits_short.shape:
        raise ValueError("Full/short logits shape mismatch")

    full = logits_full.float()
    short = logits_short.float()
    diff = full - short
    probs = F.softmax(full, dim=-1)
    mean = (probs * diff).sum(dim=-1, keepdim=True)
    diff.sub_(mean)

    residual = diff.gather(-1, targets[:, None]).squeeze(-1)
    variance = (probs * diff.square()).sum(-1).clamp_min(0.0)

    observed = None
    if need_observed:
        observed = (
            F.log_softmax(full, dim=-1).gather(-1, targets[:, None])
            - F.log_softmax(short, dim=-1).gather(-1, targets[:, None])
        ).squeeze(-1)
    return observed, residual, variance


@torch.inference_mode()
def context_scores(model, input_ids, full_logits, k, start,
                   batch_size, need_cp, need_cpd, last_logit_kwargs):
    if not (need_cp or need_cpd):
        return {}

    device = next(model.parameters()).device
    length = input_ids.numel()
    ids = input_ids.to(device)
    totals = torch.zeros(3, device=device, dtype=torch.float64)

    # Reuse identical single-token short contexts; otherwise use regular batches.
    use_dedup = k == 1
    if use_dedup:
        unique_ids, inverse = torch.unique(
            ids[start - 1:length - 1], sorted=True, return_inverse=True
        )
        unique_count = unique_ids.numel()
        short_lookup = None
        cursor = 0
        unique_batch = min(batch_size, unique_count)
        try:
            short_lookup = torch.empty(
                (unique_count, full_logits.shape[-1]),
                device=device, dtype=full_logits.dtype,
            )
            while cursor < unique_count:
                end = min(cursor + unique_batch, unique_count)
                try:
                    outputs = model(
                        input_ids=unique_ids[cursor:end, None],
                        use_cache=False, **last_logit_kwargs,
                    )
                    short_lookup[cursor:end].copy_(outputs.logits[:, -1, :])
                    del outputs
                    cursor = end
                except torch.cuda.OutOfMemoryError:
                    if unique_batch == 1:
                        raise
                    unique_batch = max(1, unique_batch // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            use_dedup = False
            del short_lookup
            gc.collect()
            torch.cuda.empty_cache()
    if not use_dedup:
        windows = ids.unfold(0, k, 1)

    batch_limit = min(batch_size, length - start)
    cursor = start
    while cursor < length:
        end = min(cursor + batch_limit, length)
        try:
            positions = torch.arange(cursor, end, device=device)
            if use_dedup:
                short = short_lookup.index_select(
                    0, inverse[cursor - start:end - start]
                ).float()
            else:
                contexts = windows.index_select(0, positions - k).contiguous()
                outputs = model(
                    input_ids=contexts, use_cache=False, **last_logit_kwargs
                )
                short = outputs.logits[:, -1, :].float()
                del outputs, contexts

            full = full_logits.index_select(
                0, (positions - 1).to(full_logits.device)
            ).to(device=device, dtype=torch.float32)
            targets = ids.index_select(0, positions)

            if need_cpd:
                observed, residual, variance = cpd_token_terms(
                    full, short, targets, need_observed=need_cp
                )
                totals[1] += residual.double().sum()
                totals[2] += variance.double().sum()
            else:
                observed = (
                    F.log_softmax(full, dim=-1).gather(-1, targets[:, None])
                    - F.log_softmax(short, dim=-1).gather(-1, targets[:, None])
                ).squeeze(-1)
            if need_cp:
                totals[0] += observed.double().sum()
            cursor = end
        except torch.cuda.OutOfMemoryError:
            if batch_limit == 1:
                raise
            batch_limit = max(1, batch_limit // 2)
            print(f"[CUDA OOM] short k={k}: batch -> {batch_limit}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    observed_sum, residual_sum, variance_sum = totals.cpu().tolist()
    scores = {}
    if need_cp:
        scores[f"CR_k{k}"] = observed_sum / (length - start)
    if need_cpd:
        scores[f"CPD_k{k}"] = residual_sum / math.sqrt(max(variance_sum, 1e-8))
    return scores


@torch.inference_mode()
def block_cpd_scores(model, input_ids, full_logits, k, start,
                     block_batch_size):
    """Evaluate nonoverlapping k-target blocks with a shared boundary token."""
    dev = next(model.parameters()).device
    length = input_ids.numel()
    input_ids_gpu = input_ids.to(dev)
    accum = torch.zeros(2, device=dev, dtype=torch.float64)
    counted = 0

    starts_by_size = {}
    for t in range(1, length, k):
        m = min(k, length-t)
        if t + m - 1 >= start:
            starts_by_size.setdefault(m, []).append(t)

    for m, starts in sorted(starts_by_size.items(), reverse=True):
        batch_limit = min(block_batch_size, len(starts))
        cursor = 0
        while cursor < len(starts):
            end = min(len(starts), cursor + batch_limit)
            try:
                t0 = torch.tensor(starts[cursor:end], device=dev, dtype=torch.long)
                offsets = torch.arange(m, device=dev)
                ctx_idx = t0[:, None] - 1 + torch.arange(m+1, device=dev)[None, :]
                contexts = input_ids_gpu[ctx_idx]
                outputs = model(input_ids=contexts, use_cache=False)
                if outputs.logits.size(1) != m+1:
                    raise ValueError("Block_CPD requires all per-position short logits; "
                                     "disable logits_to_keep and use a compatible model")

                token_positions = (t0[:, None] + offsets[None, :]).reshape(-1)
                valid = token_positions >= start
                tvalid = token_positions[valid]
                short = outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1))[valid]
                full_idx = (tvalid - 1).to(full_logits.device)
                full = full_logits.index_select(0, full_idx).to(dev, dtype=torch.float32)
                targets = input_ids_gpu.index_select(0, tvalid)
                _, residual, var = cpd_token_terms(full, short, targets)
                increments = torch.stack([residual.double().sum(), var.double().sum()])
                accum += increments
                counted += int(tvalid.numel())
                del outputs, contexts, short, full, increments
            except torch.cuda.OutOfMemoryError:
                if batch_limit == 1:
                    raise
                new_limit = max(1, batch_limit // 2)
                print(f"[CUDA OOM] block k={k}: blocks/batch {batch_limit} -> "
                      f"{new_limit}", flush=True)
                batch_limit = new_limit
                gc.collect()
                torch.cuda.empty_cache()
                continue
            cursor = end

    assert counted == length-start, (counted, length-start)
    residual_sum, variance_sum = accum.cpu().tolist()
    return {f"Block_CPD_k{k}": residual_sum / math.sqrt(max(variance_sum, 1e-8))}


def all_method_names(args):
    chosen = set(args.methods)
    names = [m for m in METHOD_CHOICES
             if m in chosen and m not in SINGLE_CONTEXT_METHODS]
    for k in args.contexts:
        names.extend(f"{m}_k{k}" for m in ("CR", "CPD", "Block_CPD") if m in chosen)
    return names


def sample_is_too_short(token_count, args):
    max_context = max(args.contexts) if args.use_short else 1
    # The original-text minimum is checked separately when it exceeds max_length.
    return (token_count < min(args.min_tokens_before_truncation, args.max_length)
            or token_count <= max_context)


def sample_is_too_long(token_count, args):
    return bool(args.max_tokens_before_truncation
                and token_count > args.max_tokens_before_truncation)


def original_length_probe_limit(args):
    # A count capped at max_length cannot distinguish an exact upper boundary
    # from an overlong original text. Probe one token beyond that boundary.
    maximum = args.max_tokens_before_truncation
    return max(args.min_tokens_before_truncation,
               maximum + 1 if maximum and maximum >= args.max_length else 0)


@torch.inference_mode()
def evaluate_text(text, ref_model, ref_tokenizer, score_model,
                  score_tokenizer, args, last_logit_kwargs,
                  selected_names=None, input_ids=None):
    # Supplied IDs have already passed the original-text filter in score_sample.
    filter_limit = original_length_probe_limit(args)
    if input_ids is None and filter_limit > args.max_length:
        filter_ids = tokenize_once(ref_tokenizer, text, filter_limit)
        if (filter_ids.numel() < args.min_tokens_before_truncation
                or sample_is_too_long(filter_ids.numel(), args)):
            return None, min(filter_ids.numel(), args.max_length)
    ids = (tokenize_once(ref_tokenizer, text, args.max_length)
           if input_ids is None else input_ids)
    L = ids.numel()
    max_context = max(args.contexts) if args.use_short else 1
    if sample_is_too_short(L, args) or sample_is_too_long(L, args):
        return None, L

    selected = set(all_method_names(args) if selected_names is None else selected_names)
    if not selected:
        return {}, L
    ref_logits = get_full_logits(ref_model, ids, args.full_logits_device)
    scores = {}
    required = selected

    score_logits = None
    if required & DUAL_MODEL_METHODS:
        if score_model is ref_model:
            score_logits = ref_logits
        else:
            score_ids = tokenize_once(score_tokenizer, text, args.max_length)
            if not torch.equal(score_ids, ids):
                raise ValueError("Dual-model baselines require identical input token IDs. "
                                 "Choose models with identical tokenizers.")
            score_logits = get_full_logits(score_model, score_ids, args.full_logits_device)

        labels = ids[1:]
        dev = next(ref_model.parameters()).device
        if required & PAIRED_METHODS:
            scores.update(paired_baselines(score_logits, ref_logits, labels,
                                           dev, required, args.score_chunk_size))
        if "Lastde++" in required:
            scores["Lastde++"] = lastdepp_dual_baseline(
                ref_logits, score_logits, labels, args.score_chunk_size, args, ids)
        if score_logits is not ref_logits:
            del score_logits

    if required & SINGLE_MODEL_METHODS:
        scores.update(single_model_baselines(
            ref_logits, ids[1:], required, args.score_chunk_size, args))

    if args.use_short:
        for k in args.contexts:
            start_k = max_context if args.same_context_start else k

            if required & {f"CR_k{k}", f"CPD_k{k}"}:
                scores.update(context_scores(
                    ref_model, ids, ref_logits, k, start_k,
                    args.short_batch_size, f"CR_k{k}" in required,
                    f"CPD_k{k}" in required, last_logit_kwargs,
                ))
            if f"Block_CPD_k{k}" in required:
                scores.update(block_cpd_scores(
                    ref_model, ids, ref_logits, k, start_k,
                    args.block_batch_size,
                ))

    scores = {name: val for name, val in scores.items() if name in selected}
    if set(scores) != selected or any(not np.isfinite(v) for v in scores.values()):
        return None, L
    del ref_logits
    return scores, L


class LazyTextScorer:
    """A cache-only run needs neither model weights nor a CUDA device."""

    def __init__(self, args):
        self.args = args
        self.ref_model = self.ref_tokenizer = None
        self.score_model = self.score_tokenizer = None
        self.last_logit_kwargs = {}

    def tokenize(self, text, max_length=None):
        if self.ref_tokenizer is None:
            path = get_model_path(self.args.base_dir, self.args.ref_name)
            self.ref_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            if self.ref_tokenizer.pad_token_id is None:
                self.ref_tokenizer.pad_token = self.ref_tokenizer.eos_token
        return tokenize_once(self.ref_tokenizer, text,
                             self.args.max_length if max_length is None else max_length)

    def evaluate(self, text, input_ids, selected_names):
        args = self.args
        if self.ref_model is None:
            self.ref_model, self.ref_tokenizer = load_model_and_tokenizer(
                args.base_dir, args.ref_name, args.precision, args.attn_impl)
            self.last_logit_kwargs = last_logits_forward_kwargs(
                self.ref_model, args.disable_last_logits_opt)
            print(f"Short-window last-logits optimization: "
                  f"{self.last_logit_kwargs or 'not available'}")
        score_model, score_tokenizer = self.ref_model, self.ref_tokenizer
        if set(selected_names) & DUAL_MODEL_METHODS and args.base_name != args.ref_name:
            if self.score_model is None:
                self.score_model, self.score_tokenizer = load_model_and_tokenizer(
                    args.base_dir, args.base_name, args.precision, args.attn_impl)
                if self.ref_tokenizer.get_vocab() != self.score_tokenizer.get_vocab():
                    raise ValueError("Dual-model baselines need identical tokenizer ID->token mappings")
            score_model, score_tokenizer = self.score_model, self.score_tokenizer
        return evaluate_text(
            text, self.ref_model, self.ref_tokenizer, score_model, score_tokenizer,
            args, self.last_logit_kwargs, selected_names=selected_names,
            input_ids=input_ids,
        )


def score_sample(sample, args, methods, cache, scorer):
    """Read a complete hit or compute only the missing method columns."""
    filter_limit = original_length_probe_limit(args)
    if filter_limit > args.max_length:
        # Cached token counts describe truncated inputs, so check original text
        # even on a complete cache hit. No model weights are needed for this.
        filter_ids = scorer.tokenize(sample["generation"], max_length=filter_limit)
        if filter_ids.numel() < args.min_tokens_before_truncation:
            return None, min(filter_ids.numel(), args.max_length), "short"
        if sample_is_too_long(filter_ids.numel(), args):
            return None, min(filter_ids.numel(), args.max_length), "long"
    cached = cache.get(sample["id"]) if cache is not None else None
    scores = {}
    if cached is not None and not args.overwrite_cache:
        token_count = cached["tokens"]
        if sample_is_too_short(token_count, args):
            return None, token_count, "short"
        if sample_is_too_long(token_count, args):
            return None, token_count, "long"
        scores = {name: cached["scores"][name] for name in methods
                  if name in cached["scores"]}
    missing = [name for name in methods if name not in scores]
    if not missing:
        return scores, token_count, "hit"

    input_ids = scorer.tokenize(sample["generation"])
    token_count = input_ids.numel()
    if sample_is_too_short(token_count, args):
        return None, token_count, "short"
    if sample_is_too_long(token_count, args):
        return None, token_count, "long"
    if cached is not None and cached["tokens"] != token_count:
        raise ValueError(
            f"Token count changed for ID {sample['id']!r}. "
            "Use a new --score-dir for changed data or scoring settings."
        )
    fresh, token_count = scorer.evaluate(sample["generation"], input_ids, missing)
    if fresh is None:
        return None, token_count, "nonfinite"
    if cache is not None:
        cache.put(sample, token_count, fresh, overwrite=args.overwrite_cache)
    scores.update(fresh)
    status = ("overwritten" if cached is not None and args.overwrite_cache
              else "partial" if cached is not None else "computed")
    return scores, token_count, status


def get_metrics(negative_scores, positive_scores):
    """Positive=1; higher scores indicate the configured positive class.

    Max_F1 is optimistically chosen on this same dataset (diagnostic only).
    For publishable F1, choose threshold on an independent validation split.
    """
    negative = np.asarray(negative_scores, dtype=np.float64)
    positive = np.asarray(positive_scores, dtype=np.float64)
    if negative.size == 0 or positive.size == 0:
        return None
    y_true = np.concatenate([np.zeros(len(negative)), np.ones(len(positive))])
    y_score = np.concatenate([negative, positive])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    valid = fpr <= 0.05
    return {
        "ROC_AUC": float(auc(fpr, tpr)),
        "Max_F1_oracle": float(f1.max()),
        "TPR_at_5FPR": float(tpr[valid].max()) if valid.any() else 0.0,
        "Negative_mean": float(negative.mean()),
        "Positive_mean": float(positive.mean()),
        "N_negative": len(negative),
        "N_positive": len(positive),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = Path(args.score_dir) if args.score_dir else None
    if score_dir is not None:
        prepare_score_directory(score_dir, vars(args))
    score_directory = ScoreDirectory(score_dir) if score_dir is not None else None

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in vars(args).items()}, f,
                  ensure_ascii=False, indent=2)

    print(f"Selected methods: {args.methods}")
    print(f"Binary task: negative={args.negative_categories}, "
          f"positive={args.positive_categories}")
    print(f"Sampling: sample_size={args.sample_size} per binary class, seed={args.seed}")
    maximum_text = (str(args.max_tokens_before_truncation)
                    if args.max_tokens_before_truncation else "unlimited")
    print(f"Length filter: {args.min_tokens_before_truncation} <= original tokens <= "
          f"{maximum_text} before truncation (inclusive; applied after sampling)")
    print(f"Score directory: {score_dir.resolve() if score_dir else 'disabled'} | "
          f"overwrite selected scores={args.overwrite_cache}")
    print(f"GPU optimization: batch={args.short_batch_size}, "
          f"precision={args.precision}, attention={args.attn_impl}, "
          f"full_logits={args.full_logits_device}")
    if args.use_short:
        print(f"Context lengths: {args.contexts}")
        if args.same_context_start:
            print(
                f"Ablation mode: shared target start index={max(args.contexts)} "
                "(0-based)"
            )
        else:
            print(
                "Default mode: each context starts at its earliest valid target "
                "index (start=k, 0-based)"
            )
    if "Lastde++" in args.methods:
        print(f"Lastde++ dual-model: sampling={args.ref_name}, "
              f"scoring={args.base_name}, n={args.lastdepp_samples}; "
              f"s={args.lastdepp_s}, scales={args.lastdepp_scales}")
    if "Fast_DetectGPT_single" in args.methods:
        print(f"Fast_DetectGPT_single: sampling=scoring={args.ref_name} (analytic moments)")
    if not args.use_dual:
        print("No dual-model baselines selected")

    scorer = LazyTextScorer(args)
    methods = all_method_names(args)
    summary_rows = []
    summary_columns = [
        "dataset", "method", "ROC_AUC", "Max_F1_oracle", "TPR_at_5FPR",
        "Negative_mean", "Positive_mean", "N_negative", "N_positive",
        "Skipped_negative", "Skipped_positive", "target_start_index",
        "same_context_start", "mean_effective_scored_tokens",
    ]

    for data_path in args.datasets:
        negative_data, positive_data = load_data(
            data_path,
            args.sample_size,
            args.seed,
            negative=args.negative_categories,
            positive=args.positive_categories,
            return_records=True,
            require_ids=score_dir is not None,
        )
        file_tag = safe_stem(data_path)
        cache = (score_directory.dataset(f"{file_tag}_scores.jsonl")
                 if score_directory is not None else None)
        print(f"\nDataset: {data_path} | Negative: {len(negative_data)}, "
              f"Positive: {len(positive_data)}", flush=True)
        started = time.perf_counter()
        records = []
        skipped = {"negative": 0, "positive": 0}
        cache_stats = dict.fromkeys(("hit", "computed", "partial", "overwritten",
                                    "short", "long", "nonfinite"), 0)

        for label, samples in (("negative", negative_data), ("positive", positive_data)):
            for i, sample in enumerate(tqdm(samples, desc=f"{label}: {Path(data_path).stem}")):
                scores, token_count, status = score_sample(
                    sample, args, methods, cache, scorer)
                cache_stats[status] += 1
                if scores is None:
                    skipped[label] += 1
                    continue
                record = {
                    "id": sample["id"],
                    "source_index": sample["source_index"],
                    "category": sample["category"],
                    "label": label,
                    "row_number_in_class": i,
                    "tokens": token_count,
                    **scores,
                }
                if args.use_short:
                    max_context = max(args.contexts)
                    for k in args.contexts:
                        start_k = max_context if args.same_context_start else k
                        record[f"scored_tokens_k{k}"] = token_count - start_k
                records.append(record)

        used_gpu = scorer.ref_model is not None and torch.cuda.is_available()
        if used_gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        gpu_usage = (f"peak GPU allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB"
                     if used_gpu else "no GPU inference")
        print(f"Elapsed: {elapsed:.1f}s | samples/s: {len(records)/max(elapsed,1e-8):.3f} "
              f"| {gpu_usage}")
        if used_gpu:
            torch.cuda.reset_peak_memory_stats()
        print(f"Scores: reused={cache_stats['hit']}, new={cache_stats['computed']}, "
              f"missing-method updates={cache_stats['partial']}, "
              f"overwritten={cache_stats['overwritten']}")
        print(f"Skipped (length filter or nonfinite): negative={skipped['negative']}, "
              f"positive={skipped['positive']} | "
              f"too short={cache_stats['short']}, too long={cache_stats['long']}, "
              f"nonfinite={cache_stats['nonfinite']}")
        if cache is not None:
            cache.compact()
        columns = ["id", "source_index", "category", "label", "row_number_in_class",
                   "tokens", *methods]
        if args.use_short:
            columns.extend(f"scored_tokens_k{k}" for k in args.contexts)
        df = pd.DataFrame(records, columns=columns)
        df.to_csv(out_dir / f"{file_tag}_per_sample.csv", index=False,
                  encoding="utf-8-sig")
        if (df.empty or len(df[df.label == "negative"]) == 0
                or len(df[df.label == "positive"]) == 0):
            print("No valid negative/positive pair remains. Check length filters and --contexts.")
            pd.DataFrame(summary_rows, columns=summary_columns).to_csv(
                out_dir / "summary.csv", index=False, encoding="utf-8-sig")
            continue

        print("=" * 99)
        for name in methods:
            negative_scores = df.loc[df.label == "negative", name].to_numpy()
            positive_scores = df.loc[df.label == "positive", name].to_numpy()
            met = get_metrics(negative_scores, positive_scores)
            print(f"{name:<22} | AUC {met['ROC_AUC']:.4f} | "
                  f"MaxF1* {met['Max_F1_oracle']:.4f} | "
                  f"TPR@5%FPR {met['TPR_at_5FPR']:.4f} | "
                  f"Neg {met['Negative_mean']:+.5f} | "
                  f"Pos {met['Positive_mean']:+.5f}")
            context_prefix = next(
                (prefix for prefix in ("CR", "CPD", "Block_CPD")
                 if name.startswith(f"{prefix}_k")),
                None,
            )
            if context_prefix is not None:
                context_k = int(name.rsplit("_k", 1)[1])
                target_start = (
                    max(args.contexts) if args.same_context_start else context_k
                )
                mean_scored_tokens = float(
                    df[f"scored_tokens_k{context_k}"].mean()
                )
            else:
                target_start = 1
                mean_scored_tokens = float((df["tokens"] - 1).mean())

            summary_rows.append({
                "dataset": data_path,
                "method": name,
                **met,
                "Skipped_negative": skipped["negative"],
                "Skipped_positive": skipped["positive"],
                "target_start_index": target_start,
                "same_context_start": bool(args.same_context_start and args.use_short),
                "mean_effective_scored_tokens": mean_scored_tokens,
            })
        print("=" * 99)
        pd.DataFrame(summary_rows, columns=summary_columns).to_csv(
            out_dir / "summary.csv", index=False, encoding="utf-8-sig")
        gc.collect()
        if used_gpu:
            torch.cuda.empty_cache()

    print(f"\nFinished. CSV outputs saved under: {out_dir.resolve()}")
    print("*MaxF1 selects threshold using test labels: report as oracle only. "
          "Use a held-out validation set for formal F1 comparisons.")


if __name__ == "__main__":
    main()
