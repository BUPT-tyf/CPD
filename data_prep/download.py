"""Fetch pinned public dataset files and check their upstream SHA-256 hashes."""
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

# LFS object hashes published with the two pinned dataset revisions.
RAW_HASHES = {
    "human.jsonl": "8ee87cb209bc72dfea449068db29325f3544f83f39c34a850c5542809d19f63f",
    "ai_generated.jsonl": "a4e6e58313433d802d2669ebf3a787e9c0eef82b67fb426513d5c25995bb5b6a",
    "ai_edited.jsonl": "2e224f472103e9dc4f569eeba91451734985421d67cebd41f79491198211cf25",
    "Binary/binary_general_open.json": "815be1667a5e7404d9e513ab339c214c713bd7ac4eb46f375fa0a26bf2dfc15f",
    "Binary/binary_polishing_open.json": "2df191654f380cf3055dac25df9cfc88d502b6ef0564ba647b9a9f1baea377fc",
    "Binary/binary_expanding_open.json": "e1c1348c607c54f2896a08f29e5176b031139a4dd5b76c1560df8e34231c53f3",
    "Binary/binary_condensing_open.json": "6ebf5640a696e7b47f71a1e932b90320fe85d97aec4a20e6c25c4704580075e4",
}


@lru_cache(maxsize=None)
def public_file(repo, revision, filename):
    cache = Path(os.environ.get("CPD_DATA_CACHE", Path.home() / ".cache" / "cpd"))
    target = cache / repo / revision / filename
    expected = RAW_HASHES[filename]
    if target.is_file():
        with target.open("rb") as handle:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                checksum.update(chunk)
        if checksum.hexdigest() == expected:
            return target

    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo}/{filename}", flush=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        partial = Path(handle.name)
    try:
        with requests.get(url, params={"download": "true"}, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            checksum = hashlib.sha256()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError(f"Upstream SHA-256 mismatch for {filename}. Check your download or use --raw-dir.")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target
