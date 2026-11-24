"""Build the GEN benchmark.

Partitions
----------
development:
    H=2000, A=2000

detection_test:
    H=2000, A=2000

edit_test:
    H=N, A=N, HA_20_40=N, HA_40_60=N, HA_60_80=N

Edit-test A is reused from detection-test A.
Only the A samples are intentionally shared between the two test sets.
All other source partitions remain disjoint.

Sampling
--------
- Five domains.
- 400 H/A per domain in development and detection_test.
- Edit-test H uses complete official seed_id values.
- Each seed_id has one HA sample in each edit-ratio interval.
- Edit-test A is selected from detection-test A.
- Edit-test A matches the number of human drafts in each domain.
- HA bins use measured edit_ratio: [20,40), [40,60), [60,80).
"""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm.auto import tqdm


# ============================== Settings ==============================

DATASET_ID = "szyszy/GEN"
REVISION = "a0f143c1ae9c35b684898a5f9c0ab6efb49be486"

SEED = 42
EXPECTED_DOMAINS = 5

PER_CLASS_PER_SPLIT = 2000
PER_DOMAIN = PER_CLASS_PER_SPLIT // EXPECTED_DOMAINS

TARGET_EDIT_SEEDS = 1000
MAX_A_PER_SOURCE = 2

EDIT_RATIO_FIELD = "edit_ratio"

EDIT_LEVELS = ("20_40", "40_60", "60_80")

EDIT_RANGES = {
    "20_40": (20.0, 40.0),
    "40_60": (40.0, 60.0),
    "60_80": (60.0, 80.0),
}

EXPECTED_ROWS = {
    "human": 5000,
    "ai_generated": 39391,
    "ai_edited": 273418,
}

CLASS_INFO = {
    "H": {
        "label": 0,
        "binary_label": 0,
        "category": "H",
        "edit_level": None,
    },
    "A": {
        "label": 1,
        "binary_label": 1,
        "category": "A",
        "edit_level": None,
    },
    "HA_20_40": {
        "label": 2,
        "binary_label": 1,
        "category": "HA_20_40",
        "edit_level": "20_40",
    },
    "HA_40_60": {
        "label": 3,
        "binary_label": 1,
        "category": "HA_40_60",
        "edit_level": "40_60",
    },
    "HA_60_80": {
        "label": 4,
        "binary_label": 1,
        "category": "HA_60_80",
        "edit_level": "60_80",
    },
}


# ============================== Basic utilities ==============================

def stream_rows(config, raw_dir, revision=REVISION):
    if raw_dir is None:
        from .download import public_file
        path = public_file(DATASET_ID, revision, f"{config}.jsonl")
    else:
        path = Path(raw_dir) / f"{config}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def progress_rows(config, phase, stream):
    return tqdm(
        stream(config),
        desc=f"GEN {phase} {config}",
        total=EXPECTED_ROWS.get(config),
        unit="rows",
    )


def scalar(value):
    if value is None:
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def metadata_str(value):
    value = scalar(value)
    return "unknown" if value is None else str(value)


def source_id(row):
    value = row.get("prompt_id")

    if value is None or str(value).strip() == "":
        raise ValueError(
            "GEN row has no prompt_id; source leakage cannot be checked"
        )

    return str(value)


def domain_name(row):
    return str(row.get("source") or "unknown")


def has_text(row):
    return (
        isinstance(row.get("text"), str)
        and bool(row["text"].strip())
    )


def check_label(config, row, idx):
    expected = {
        "human": 0,
        "ai_generated": 1,
        "ai_edited": 2,
    }[config]

    label = row.get("label")

    if label is not None and int(label) != expected:
        raise ValueError(
            f"Bad label in {config} row {idx}: "
            f"{label}; expected {expected}"
        )


# ============================== Edit ratio ==============================

def measured_edit_ratio(row):
    try:
        ratio = float(row.get(EDIT_RATIO_FIELD))
    except (TypeError, ValueError):
        return None

    return ratio if math.isfinite(ratio) else None


def edit_level_from_row(row):
    ratio = measured_edit_ratio(row)

    if ratio is None:
        return None

    for level, (lo, hi) in EDIT_RANGES.items():
        if lo <= ratio < hi:
            return level

    return None


# ============================== Human collection ==============================

def get_humans(stream):
    humans = {}

    for idx, row in enumerate(progress_rows("human", "collect", stream)):
        check_label("human", row, idx)

        if not has_text(row):
            continue

        pid = source_id(row)

        if pid in humans:
            raise ValueError(f"Duplicated human prompt_id: {pid}")

        humans[pid] = (idx, row)

    if not humans:
        raise RuntimeError("No valid human rows")

    return humans


# ============================== Reservoir sampling ==============================

def reservoir_add(bucket, key, row_index, rng):
    state = bucket.get(key)

    if state is None:
        bucket[key] = [row_index, 1]
    else:
        state[1] += 1

        if rng.randrange(state[1]) == 0:
            state[0] = row_index


# ============================== AI indexing ==============================

def index_ai_candidates(humans, stream):
    rng = random.Random(SEED + 101)

    candidates = defaultdict(dict)
    stats = Counter()

    for idx, row in enumerate(progress_rows("ai_generated", "index", stream)):
        check_label("ai_generated", row, idx)

        if not has_text(row):
            stats["empty_text"] += 1
            continue

        pid = source_id(row)

        if pid not in humans:
            stats["prompt_id_not_in_human"] += 1
            continue

        if domain_name(row) != domain_name(humans[pid][1]):
            raise ValueError(f"Domain mismatch for AI prompt_id={pid}")

        reservoir_add(
            candidates[pid],
            (metadata_str(row.get("model")),),
            idx,
            rng,
        )

        stats["valid_rows"] += 1

    return dict(candidates), stats


# ============================== HA indexing ==============================

def index_ha_candidates(humans, stream):
    rng = random.Random(SEED + 103)

    candidates = defaultdict(lambda: defaultdict(dict))
    seed_info = {}
    seed_to_pid = {}
    stats = Counter()

    for idx, row in enumerate(progress_rows("ai_edited", "index", stream)):
        check_label("ai_edited", row, idx)

        if not has_text(row):
            stats["empty_text"] += 1
            continue

        pid = source_id(row)

        raw_seed_id = scalar(row.get("seed_id"))

        if raw_seed_id is None or str(raw_seed_id).strip() == "":
            stats["missing_seed_id"] += 1
            continue

        seed_id = str(raw_seed_id)

        previous_pid = seed_to_pid.setdefault(seed_id, pid)

        if previous_pid != pid:
            raise ValueError(
                f"seed_id={seed_id} maps to multiple prompt_id values"
            )

        if pid not in humans:
            stats["seed_source_not_in_human"] += 1
            continue

        if domain_name(row) != domain_name(humans[pid][1]):
            raise ValueError(f"Domain mismatch for HA prompt_id={pid}")

        seed_text = row.get("seed_text")

        if not isinstance(seed_text, str) or not seed_text.strip():
            stats["missing_seed_text"] += 1
            continue

        info = {
            "seed_id": seed_id,
            "prompt_id": pid,
            "domain": domain_name(row),
            "seed_text": seed_text,
            "prompt": (
                row.get("prompt")
                if isinstance(row.get("prompt"), str)
                else ""
            ),
        }

        previous = seed_info.setdefault(seed_id, info)

        if previous["seed_text"].strip() != seed_text.strip():
            raise ValueError(f"Inconsistent seed_text: {seed_id}")

        if previous["prompt_id"] != pid:
            raise ValueError(f"Inconsistent prompt_id: {seed_id}")

        level = edit_level_from_row(row)

        if level is None:
            stats["outside_bins_or_missing_edit_ratio"] += 1
            continue

        key = (
            metadata_str(row.get("model")),
            metadata_str(row.get("edit_category")),
            metadata_str(row.get("target_ratio")),
        )

        reservoir_add(
            candidates[seed_id][level],
            key,
            idx,
            rng,
        )

        stats[f"valid_{level}_rows"] += 1

    normalized = {
        sid: {
            level: dict(strata)
            for level, strata in levels.items()
        }
        for sid, levels in candidates.items()
    }

    return normalized, stats, seed_to_pid, seed_info


# ============================== Source partitions ==============================

def make_source_partitions(
    humans,
    ha_candidates,
    seed_to_pid,
    seed_info,
    domains,
    output_dir,
):
    rng = random.Random(SEED + 307)

    by_domain = defaultdict(list)

    for pid, (_, row) in humans.items():
        by_domain[domain_name(row)].append(pid)

    mapped_seed_ids = {
        sid: pid
        for sid, pid in seed_to_pid.items()
        if pid in humans and sid in seed_info
    }

    eligible_seed_ids = [
        sid for sid in mapped_seed_ids
        if all(
            ha_candidates.get(sid, {}).get(level)
            for level in EDIT_LEVELS
        )
    ]

    missing_by_level = Counter()

    for sid in mapped_seed_ids:
        for level in EDIT_LEVELS:
            if not ha_candidates.get(sid, {}).get(level):
                missing_by_level[level] += 1

    eligible_set = set(eligible_seed_ids)

    audit = {
        "target_seed_count": TARGET_EDIT_SEEDS,
        "distinct_seed_ids_seen": len(seed_to_pid),
        "seed_ids_mapped_to_human": len(mapped_seed_ids),
        "complete_three_level_seed_count": len(eligible_seed_ids),
        "missing_bin_counts": dict(missing_by_level),
        "seeds_missing_one_or_more_bins": {
            sid: [
                level for level in EDIT_LEVELS
                if not ha_candidates.get(sid, {}).get(level)
            ]
            for sid in mapped_seed_ids
            if sid not in eligible_set
        },
        "seed_texts_by_seed_id": {
            sid: {
                "prompt_id": seed_info[sid]["prompt_id"],
                "domain": seed_info[sid]["domain"],
                "matches_human_jsonl_text": (
                    seed_info[sid]["seed_text"].strip()
                    == str(
                        humans[seed_info[sid]["prompt_id"]][1].get("text") or ""
                    ).strip()
                ),
            }
            for sid in eligible_seed_ids
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = output_dir / "GEN_seed_coverage_audit.json"

    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not eligible_seed_ids:
        raise RuntimeError(
            f"No complete paired edit seeds found: {audit_path}"
        )

    print(
        f"Complete seeds: {len(eligible_seed_ids)}; "
        f"official IDs: {len(seed_to_pid)}"
    )

    seed_ids_by_domain = defaultdict(list)

    for sid in sorted(
        eligible_seed_ids,
        key=lambda x: (
            not x.isdigit(),
            int(x) if x.isdigit() else x,
        ),
    ):
        seed_ids_by_domain[seed_info[sid]["domain"]].append(sid)

    pools = {
        name: defaultdict(list)
        for name in ("development", "detection_test", "edit_test")
    }

    for domain in domains:
        pids = sorted(by_domain[domain])

        edit_pids = {
            seed_info[sid]["prompt_id"]
            for sid in seed_ids_by_domain.get(domain, [])
        }

        remaining = [
            pid for pid in pids
            if pid not in edit_pids
        ]

        if len(remaining) < 2 * PER_DOMAIN:
            raise RuntimeError(
                f"Insufficient sources in {domain}: "
                f"{len(remaining)} < {2 * PER_DOMAIN}"
            )

        rng.shuffle(remaining)

        pools["edit_test"][domain] = list(
            seed_ids_by_domain.get(domain, [])
        )

        pools["development"][domain] = remaining[:PER_DOMAIN]

        pools["detection_test"][domain] = (
            remaining[PER_DOMAIN:2 * PER_DOMAIN]
        )

        print(
            f"{domain}: "
            f"dev={len(pools['development'][domain])}, "
            f"det={len(pools['detection_test'][domain])}, "
            f"edit={len(pools['edit_test'][domain])}"
        )

    # Check H-source isolation across all three partitions.
    dev = {
        pid
        for values in pools["development"].values()
        for pid in values
    }

    det = {
        pid
        for values in pools["detection_test"].values()
        for pid in values
    }

    edit = {
        seed_info[sid]["prompt_id"]
        for values in pools["edit_test"].values()
        for sid in values
    }

    for name1, set1, name2, set2 in (
        ("development", dev, "detection_test", det),
        ("development", dev, "edit_test", edit),
        ("detection_test", det, "edit_test", edit),
    ):
        overlap = set1 & set2

        if overlap:
            raise AssertionError(
                f"Human source overlap {name1}/{name2}: {len(overlap)}"
            )

    return pools


# ============================== Sample selection ==============================

def select_humans(humans, source_pools, split):
    selected = {}

    for domain in sorted(source_pools[split]):
        pids = list(source_pools[split][domain])

        random.Random(
            SEED + sum(map(ord, split)) + sum(map(ord, domain))
        ).shuffle(pids)

        selected.update({
            pid: humans[pid][0]
            for pid in pids
        })

    if len(selected) != PER_CLASS_PER_SPLIT:
        raise AssertionError(
            f"{split} H count {len(selected)} != {PER_CLASS_PER_SPLIT}"
        )

    return selected


def select_balanced_ai(candidates, source_pools, split):
    rng = random.Random(
        SEED + (501 if split == "development" else 503)
    )

    selected = {}

    for domain in sorted(source_pools[split]):
        pids = [
            pid for pid in source_pools[split][domain]
            if candidates.get(pid)
        ]

        rng.shuffle(pids)

        capacity = sum(
            min(MAX_A_PER_SOURCE, len(candidates[pid]))
            for pid in pids
        )

        if capacity < PER_DOMAIN:
            raise RuntimeError(
                f"Insufficient A capacity {split}/{domain}: "
                f"{capacity} < {PER_DOMAIN}"
            )

        model_counts = Counter()
        source_counts = Counter()
        available = {
            pid: dict(candidates[pid])
            for pid in pids
        }

        chosen = 0

        for _ in range(MAX_A_PER_SOURCE):
            order = pids[:]
            rng.shuffle(order)

            for pid in order:
                if chosen >= PER_DOMAIN:
                    break

                if source_counts[pid] >= MAX_A_PER_SOURCE:
                    continue

                if not available[pid]:
                    continue

                options = list(available[pid].items())
                rng.shuffle(options)

                model, state = min(
                    options,
                    key=lambda kv: model_counts[kv[0]],
                )

                selected[state[0]] = pid

                del available[pid][model]

                model_counts[model] += 1
                source_counts[pid] += 1
                chosen += 1

        if chosen != PER_DOMAIN:
            raise RuntimeError(
                f"Selected {chosen} A rows for {split}/{domain}; "
                f"expected {PER_DOMAIN}"
            )

        print(
            f"A {split}/{domain}: {chosen}; "
            f"models={dict(model_counts)}"
        )

    return selected


def select_ha_level(level, ha_candidates, seed_ids_by_domain):
    rng = random.Random(
        SEED + 601 + EDIT_LEVELS.index(level)
    )

    selected = {}

    global_counts = [Counter(), Counter(), Counter()]
    global_options = [set(), set(), set()]
    source_options = {}

    for domain, seed_ids in seed_ids_by_domain.items():
        for sid in seed_ids:
            strata = ha_candidates[sid][level]
            source_options[sid] = dict(strata)

            for key in strata:
                for d in range(3):
                    global_options[d].add(key[d])

    for domain, seed_ids in seed_ids_by_domain.items():
        local_counts = [Counter(), Counter(), Counter()]
        local_options = [set(), set(), set()]

        for sid in seed_ids:
            for key in source_options[sid]:
                for d in range(3):
                    local_options[d].add(key[d])

        order = list(seed_ids)
        rng.shuffle(order)

        for sid in order:
            options = list(source_options[sid].items())
            rng.shuffle(options)

            def imbalance(pair):
                key = pair[0]

                secondary = sum(
                    2 * max(1, len(global_options[d]))
                    * global_counts[d][key[d]]
                    + max(1, len(local_options[d]))
                    * local_counts[d][key[d]]
                    for d in (1, 2)
                )

                return (
                    global_counts[0][key[0]],
                    local_counts[0][key[0]],
                    secondary,
                )

            key, state = min(options, key=imbalance)

            row_idx = state[0]

            if row_idx in selected:
                raise AssertionError(f"Duplicate HA row index {row_idx}")

            selected[row_idx] = sid

            for d in range(3):
                global_counts[d][key[d]] += 1
                local_counts[d][key[d]] += 1

    expected = sum(len(v) for v in seed_ids_by_domain.values())

    if len(selected) != expected:
        raise AssertionError(
            f"HA_{level} count {len(selected)} != {expected}"
        )

    return selected


# ============================== Recovery ==============================

def recover_selected(config, selected_index_to_source, stream, expected_level=None):
    needed = dict(selected_index_to_source)
    found = {}

    for idx, row in enumerate(progress_rows(config, "retrieve", stream)):
        expected_key = needed.get(idx)

        if expected_key is None:
            continue

        actual_key = (
            str(scalar(row.get("seed_id")))
            if config == "ai_edited"
            else source_id(row)
        )

        if actual_key != str(expected_key) or not has_text(row):
            raise RuntimeError(
                f"Selected row mismatch {config}/{idx}: "
                f"expected {expected_key}, got {actual_key}"
            )

        check_label(config, row, idx)

        if (
            config == "ai_edited"
            and edit_level_from_row(row) != expected_level
        ):
            raise RuntimeError(
                f"Edit bin changed at {config}/{idx}"
            )

        found[idx] = (str(expected_key), row)

        if len(found) == len(needed):
            break

    if len(found) != len(needed):
        missing = sorted(set(needed) - set(found))[:20]

        raise RuntimeError(
            f"Could not recover {config} rows: {missing}"
        )

    return found


# ============================== NEW: Reuse Detection Test A ==============================

def select_edit_a_from_detection(detection_a, target_by_domain):
    """
    Reuse A rows already selected for detection_test.

    Select exactly N rows for edit_test, matching H_seed counts
    in each domain. Balance generator models within each domain.

    No new AI indexing or recovery is needed.
    """

    rng = random.Random(SEED + 801)

    by_domain = defaultdict(list)

    for idx, (pid, row) in detection_a.items():
        by_domain[domain_name(row)].append((idx, pid, row))

    selected = {}

    for domain in sorted(target_by_domain):
        target = target_by_domain[domain]
        candidates = by_domain.get(domain, [])

        if len(candidates) < target:
            raise RuntimeError(
                f"Detection A insufficient for edit_test/{domain}: "
                f"{len(candidates)} < {target}"
            )

        rng.shuffle(candidates)

        model_counts = Counter()
        chosen = []

        # Preserve the original seeded ordering of the fixed AI controls.
        for item in sorted(
            candidates,
            key=lambda x: (
                model_counts[str(x[2].get("model") or "unknown")],
                rng.random(),
            ),
        ):
            if len(chosen) >= target:
                break

            idx, pid, row = item
            chosen.append(item)

            model = str(row.get("model") or "unknown")
            model_counts[model] += 1

        for idx, pid, row in chosen:
            selected[idx] = (pid, row)

        if len(chosen) != target:
            raise AssertionError(
                f"Edit A count {domain}: {len(chosen)} != {target}"
            )

        print(
            f"Edit A {domain}: {len(chosen)}; "
            f"models={dict(model_counts)}"
        )

    return selected


# ============================== Record creation ==============================

def to_record(
    row_index,
    row,
    split,
    class_name,
    pid,
    role=None,
    seed_id=None,
):
    info = CLASS_INFO[class_name]

    record = {
        "id": (
            f"gen_{split}_{class_name.lower()}_seed_{seed_id}"
            if seed_id is not None
            else f"gen_{split}_{class_name.lower()}_{row_index:08d}"
        ),
        "prompt": (
            row.get("prompt")
            if isinstance(row.get("prompt"), str)
            else ""
        ),
        "generation": row["text"],
        "model": (
            "human"
            if class_name == "H"
            else str(row.get("model") or "unknown")
        ),
        "label": info["label"],
        "binary_label": info["binary_label"],
        "label_name": class_name,
        "category": info["category"],
        "edit_level": info["edit_level"],
        "dataset": "GEN",
        "domain": domain_name(row),
        "lang": "english",
        "split": split,
        "source_id": pid,
        "prompt_id": scalar(row.get("prompt_id")),
        "seed_id": (
            seed_id
            if seed_id is not None
            else scalar(row.get("seed_id"))
        ),
    }

    if role is not None:
        record["sample_role"] = role

    if class_name.startswith("HA_"):
        for field in (
            "seed_id",
            "edit_category",
            "edit_ratio",
            "edit_ratio_bin",
            "target_ratio",
            "edit_prompt",
        ):
            value = scalar(row.get(field))

            if value is not None:
                record[field] = value

    return record


# ============================== Output ==============================

def write_jsonl_atomic(records, path):
    tmp = path.with_name(path.name + ".tmp")

    try:
        with tmp.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )

        tmp.replace(path)

    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def validate_class_domain(records, expected_classes, per_domain):
    counts = Counter(r["label_name"] for r in records)

    for cls, expected in expected_classes.items():
        if counts[cls] != expected:
            raise AssertionError(
                f"{cls} count {counts[cls]} != {expected}"
            )

        by_domain = Counter(
            r["domain"]
            for r in records
            if r["label_name"] == cls
        )

        for domain, n in per_domain.items():
            if by_domain[domain] != n:
                raise AssertionError(
                    f"{cls}/{domain}: {by_domain[domain]} != {n}"
                )

    ids = [r["id"] for r in records]

    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate output record IDs")

    return counts


# ============================== Main ==============================

def build(output_dir, raw_dir=None, revision=REVISION):
    def stream(config):
        return stream_rows(config, raw_dir, revision)

    if PER_CLASS_PER_SPLIT % EXPECTED_DOMAINS != 0:
        raise ValueError(
            "PER_CLASS_PER_SPLIT must divide evenly across domains"
        )

    print("[1/8] Collecting Human records...")

    humans = get_humans(stream)

    domains = sorted({
        domain_name(row)
        for _, row in humans.values()
    })

    if len(domains) != EXPECTED_DOMAINS:
        raise RuntimeError(
            f"Expected {EXPECTED_DOMAINS} domains, "
            f"found {len(domains)}: {domains}"
        )

    print("[2/8] Indexing AI-generated candidates...")

    ai_candidates, ai_stats = index_ai_candidates(humans, stream)

    print("[3/8] Indexing AI-edited candidates...")

    ha_candidates, ha_stats, seed_to_pid, seed_info = (
        index_ha_candidates(humans, stream)
    )

    print("[4/8] Creating source partitions...")

    source_pools = make_source_partitions(
        humans,
        ha_candidates,
        seed_to_pid,
        seed_info,
        domains,
        output_dir=output_dir,
    )

    print("[5/8] Selecting H and A samples...")

    human_selected = {
        split: select_humans(humans, source_pools, split)
        for split in ("development", "detection_test")
    }

    ai_selected = {
        split: select_balanced_ai(
            ai_candidates,
            source_pools,
            split,
        )
        for split in ("development", "detection_test")
    }

    edit_seed_ids_by_domain = source_pools["edit_test"]

    edit_seed_ids = [
        sid
        for ids in edit_seed_ids_by_domain.values()
        for sid in ids
    ]

    actual_seed_total = len(edit_seed_ids)

    expected_by_domain = {
        domain: len(edit_seed_ids_by_domain[domain])
        for domain in domains
    }

    print("[6/8] Selecting HA samples...")

    ha_selected = {
        level: select_ha_level(
            level,
            ha_candidates,
            edit_seed_ids_by_domain,
        )
        for level in EDIT_LEVELS
    }

    print("[7/8] Recovering selected rows...")

    ai_found = {
        split: recover_selected(
            "ai_generated",
            selected,
            stream=stream,
        )
        for split, selected in ai_selected.items()
    }

    ha_found = {
        level: recover_selected(
            "ai_edited",
            selected,
            expected_level=level,
            stream=stream,
        )
        for level, selected in ha_selected.items()
    }

    # Reuse Detection Test A for Edit Test.
    # Match the number of H seeds in each domain.
    edit_a_found = select_edit_a_from_detection(
        ai_found["detection_test"],
        expected_by_domain,
    )

    print("[8/8] Writing outputs...")

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # ---------------- Development / Detection Test ----------------

    for split in ("development", "detection_test"):

        records = []

        for pid, idx in human_selected[split].items():
            records.append(
                to_record(
                    idx,
                    humans[pid][1],
                    split,
                    "H",
                    pid,
                )
            )

        for idx, (pid, row) in ai_found[split].items():
            records.append(
                to_record(
                    idx,
                    row,
                    split,
                    "A",
                    pid,
                )
            )

        random.Random(
            SEED + (701 if split == "development" else 702)
        ).shuffle(records)

        validate_class_domain(
            records,
            {"H": 2000, "A": 2000},
            {domain: PER_DOMAIN for domain in domains},
        )

        path = output_dir / {"development": "GEN_dev.jsonl", "detection_test": "GEN_test.jsonl"}[split]

        write_jsonl_atomic(records, path)

        outputs[split] = path

        print(
            f"{path}: {len(records)} rows; "
            f"{dict(Counter(r['label_name'] for r in records))}"
        )

    # ---------------- Edit Test ----------------

    records = []

    # H: one Human seed_text per complete seed_id.
    for sid in edit_seed_ids:

        info = seed_info[sid]

        synthetic_human = {
            "text": info["seed_text"],
            "prompt": info["prompt"],
            "source": info["domain"],
            "prompt_id": info["prompt_id"],
            "seed_id": sid,
        }

        records.append(
            to_record(
                int(sid) if sid.isdigit()
                else abs(hash(sid)) % (10**8),
                synthetic_human,
                "edit_test",
                "H",
                info["prompt_id"],
                role="human_seed",
                seed_id=sid,
            )
        )

    # A: reuse selected samples from Detection Test.
    for idx, (pid, row) in edit_a_found.items():
        records.append(
            to_record(
                idx,
                row,
                "edit_test",
                "A",
                pid,
            )
        )

    # HA: retain original seed_id pairing.
    for level in EDIT_LEVELS:

        class_name = f"HA_{level}"

        for idx, (sid, row) in ha_found[level].items():

            rec = to_record(
                idx,
                row,
                "edit_test",
                class_name,
                seed_info[sid]["prompt_id"],
                seed_id=sid,
            )

            rec["seed_source_id"] = seed_info[sid]["prompt_id"]

            records.append(rec)

    random.Random(SEED + 703).shuffle(records)

    # ---------------- Edit Test validation ----------------

    expected_edit = {
        "H": actual_seed_total,
        "A": actual_seed_total,
        **{
            f"HA_{level}": actual_seed_total
            for level in EDIT_LEVELS
        },
    }

    counts = Counter(r["label_name"] for r in records)

    if dict(counts) != expected_edit:
        raise AssertionError(
            f"Edit-test counts mismatch: "
            f"{dict(counts)} != {expected_edit}"
        )

    validate_class_domain(
        records,
        expected_edit,
        expected_by_domain,
    )

    # Verify HA seed pairing.
    selected_seed_ids = set(edit_seed_ids)

    for level in EDIT_LEVELS:

        ids = [
            sid
            for _, (sid, _) in ha_found[level].items()
        ]

        if len(ids) != len(set(ids)) or set(ids) != selected_seed_ids:
            raise AssertionError(
                f"HA_{level} is not one-to-one with H seed_ids"
            )

    # Verify reused A count and domain balance.
    if len(edit_a_found) != actual_seed_total:
        raise AssertionError("Edit-test A count mismatch")

    edit_a_domain_counts = Counter(
        domain_name(row)
        for _, (_, row) in edit_a_found.items()
    )

    if dict(edit_a_domain_counts) != expected_by_domain:
        raise AssertionError(
            f"Edit A domain mismatch: "
            f"{dict(edit_a_domain_counts)} != {expected_by_domain}"
        )

    edit_path = output_dir / "GEN_edit_test.jsonl"

    write_jsonl_atomic(records, edit_path)

    outputs["edit_test"] = edit_path

    print(
        f"{edit_path}: {len(records)} rows; "
        f"{dict(counts)}"
    )

    # ---------------- Source isolation audit ----------------

    dev_sources = (
        set(human_selected["development"])
        | set(ai_selected["development"].values())
    )

    det_sources = (
        set(human_selected["detection_test"])
        | set(ai_selected["detection_test"].values())
    )

    edit_h_sources = {
        seed_info[sid]["prompt_id"]
        for sid in edit_seed_ids
    }

    edit_a_sources = {
        pid for pid, _ in edit_a_found.values()
    }

    # Dev and Detection must remain disjoint.
    if dev_sources & det_sources:
        raise AssertionError("Development/Detection source leakage")

    # Edit H/HA sources must remain disjoint from Dev and Detection.
    if dev_sources & edit_h_sources:
        raise AssertionError("Development/Edit H source leakage")

    if det_sources & edit_h_sources:
        raise AssertionError("Detection/Edit H source leakage")

    # Edit A is intentionally reused from Detection A.
    if not edit_a_sources.issubset(det_sources):
        raise AssertionError(
            "Edit A contains samples not sourced from Detection Test"
        )

    overlaps = {
        "development__detection_test": len(dev_sources & det_sources),
        "development__edit_H_HA": len(dev_sources & edit_h_sources),
        "detection_test__edit_H_HA": len(det_sources & edit_h_sources),
        "detection_test__edit_A_intentional": len(
            det_sources & edit_a_sources
        ),
    }

    # ---------------- Coverage report ----------------

    report = {
        "dataset_id": DATASET_ID,

        "source_split_key": "prompt_id",
        "edit_pairing_key": "seed_id",

        "split_design": {
            "development": {
                "H": 2000,
                "A": 2000,
            },
            "detection_test": {
                "H": 2000,
                "A": 2000,
            },
            "edit_test": expected_edit,
        },

        "domains": domains,

        "per_domain": {
            "development_and_detection": 400,
            "edit_test": expected_by_domain,
        },

        "target_edit_seed_total": TARGET_EDIT_SEEDS,
        "actual_edit_seed_total": actual_seed_total,

        "official_seed_id_count": len(seed_to_pid),
        "official_seed_id_to_prompt_id": seed_to_pid,

        "edit_level_field": EDIT_RATIO_FIELD,

        "edit_level_ranges": {
            "20_40": "[20, 40)",
            "40_60": "[40, 60)",
            "60_80": "[60, 80)",
        },

        "ha_pairing": (
            "One HA row per official seed_id per edit level; "
            "Human text from seed_text"
        ),

        "edit_test_A_sampling": (
            "Reused from detection_test A; "
            "subsampled by domain to match H_seed counts"
        ),

        "source_overlap": overlaps,

        "edit_test_A_model_distribution": dict(
            Counter(
                str(row.get("model") or "unknown")
                for _, (_, row) in edit_a_found.items()
            )
        ),

        "ai_index_stats": dict(ai_stats),
        "ha_index_stats": dict(ha_stats),

        "output_files": {
            key: value.name
            for key, value in outputs.items()
        },
    }

    (output_dir / "GEN_coverage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---------------- Final summary ----------------

    print("\n=== GEN REVISED BENCHMARK COMPLETE ===")

    print("Development:    H=2000, A=2000")
    print("Detection test: H=2000, A=2000")

    print(
        f"Edit test: H={actual_seed_total}, "
        f"A={actual_seed_total}, "
        f"HA={actual_seed_total} x 3 levels"
    )

    print("Edit-test domain counts:", expected_by_domain)

    print(
        "Edit-test A model distribution:",
        report["edit_test_A_model_distribution"],
    )

    print("Source overlap audit:", overlaps)

    print("Outputs:", output_dir.resolve())

    return outputs
