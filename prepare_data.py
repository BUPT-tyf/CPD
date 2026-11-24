"""Prepare and verify the exact JSONL files used in the CPD experiments."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

MAIN_GROUPS = ("gen", "detectrlx", "hart")
GROUPS = (*MAIN_GROUPS, "gen-edit", "rewriting")
MANIFEST = Path(__file__).parent / "data_prep" / "manifest.json"


def digest(path):
    hash_state = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_state.update(chunk)
    return hash_state.hexdigest()


def mismatches(directory, files):
    failures = []
    for name, record in files.items():
        path = directory / name
        if not path.is_file():
            failures.append(f"missing {name}")
        elif digest(path) != record["sha256"]:
            failures.append(f"SHA-256 mismatch for {name}")
    return failures


def prepare(group, directory, raw_dir):
    if group == "gen":
        from data_prep.gen_direct import build
        build(directory, raw_dir)
    elif group == "gen-edit":
        from data_prep.gen import build
        build(directory, raw_dir)
    elif group == "hart":
        from data_prep.hart import build
        build(directory, raw_dir)
    else:
        from data_prep.detectrlx import build
        mode = "general" if group == "detectrlx" else "aa"
        build(directory, raw_dir, group=mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("groups", nargs="*", metavar="GROUP",
                        help="One or more of " + ", ".join(GROUPS) + " (default: gen detectrlx hart)")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark_2026"))
    parser.add_argument("--raw-dir", type=Path, help="Local upstream files for one build group")
    args = parser.parse_args()
    groups = list(dict.fromkeys(args.groups)) or list(MAIN_GROUPS)
    if any(group not in GROUPS for group in groups):
        parser.error("Unknown group. Choose from " + ", ".join(GROUPS))
    if args.raw_dir and (len(groups) != 1 or args.action != "build"):
        parser.error("--raw-dir requires build with one explicit group")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for group in groups:
        files = {k: v for k, v in manifest.items() if v["group"] == group}
        errors = mismatches(args.data_dir, files)
        if errors and args.action == "build":
            args.data_dir.mkdir(parents=True, exist_ok=True)
            # Keep existing outputs usable if a download or conversion fails.
            with TemporaryDirectory(dir=args.data_dir, prefix=".prepare-") as temp:
                stage = Path(temp)
                prepare(group, stage, args.raw_dir)
                errors = mismatches(stage, files)
                if not errors:
                    for name in files:
                        destination = args.data_dir / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        (stage / name).replace(destination)
        if errors:
            problems.extend(errors)
        else:
            print(f"{group}: verified {len(files)} file(s)")
    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()
