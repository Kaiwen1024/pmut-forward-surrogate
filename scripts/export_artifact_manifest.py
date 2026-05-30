#!/usr/bin/env python
"""Write size and SHA-256 metadata for repository artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_PATTERNS = [
    "configs/*.json",
    "data/fullrange_cache/*",
    "artifacts/checkpoints/*",
    "results/fullrange_10k_retrain/*.csv",
    "results/fullrange_10k_retrain/*.json",
    "results/fullrange_10k_retrain/*.png",
    "results/fullrange_10k_retrain/eval_test/*.json",
    "figures/artifact/*.png",
    "figures/paper/*.md",
    "figures/paper/*/data/*.csv",
    "figures/paper/*/data/*.json",
    "figures/paper/*/data/*.md",
    "figures/paper/*/data/*.tex",
    "figures/paper/*/export/*.pdf",
    "figures/paper/*/export/*.png",
    "figures/paper/*/export/*.svg",
    "figures/paper/*/source/*.py",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path, patterns: List[str]) -> List[Path]:
    files = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files))


def main() -> None:
    p = argparse.ArgumentParser(description="Export artifact manifest and SHA-256 checksums.")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--out", type=Path, default=Path("artifact_manifest.json"))
    p.add_argument("--sha256-out", type=Path, default=Path("sha256sums.txt"))
    p.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS)
    args = p.parse_args()

    root = args.root.resolve()
    records: List[Dict[str, object]] = []
    sha_lines: List[str] = []
    for path in collect_files(root, list(args.patterns)):
        rel = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        sha_lines.append(f"{digest}  {rel}")

    payload = {"root": ".", "files": records}
    (root / args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / args.sha256_out).write_text("\n".join(sha_lines) + ("\n" if sha_lines else ""), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
