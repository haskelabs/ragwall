#!/usr/bin/env python3
"""Package the healthcare-focused RagWall release into a distributable archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List
import zipfile

# Paths relative to repository root that should be included in the release.
CONTENT_PATHS: List[str] = [
    "release/healthcare/README.md",
    "sanitizer/jailbreak/prr_gate.py",
    "docs/PATTERN_ENHANCEMENT_SUMMARY.md",
    "docs/AB_EVAL_ENHANCED_RESULTS.md",
    "docs/HEALTHCARE_RAG_ANALYSIS.md",
    "data/health_care_1000_queries.jsonl",
    "data/health_care_1000_queries_converted.jsonl",
    "data/health_care_1000_queries_sanitized.jsonl",
    "data/health_care_1000_queries_sanitized_v2.jsonl",
    "data/health_care_1000_queries_sanitized_v3.jsonl",
    "data/health_care_100_queries_matched.jsonl",
    "data/health_care_100_queries_matched_sanitized.jsonl",
    "experiments/results/tiny_jb_vectors.pt",
    "experiments/results/healthcare_vectors.pt",
    "reports/health_100_matched_eval",
    "reports/health_1000_enhanced_eval",
]


@dataclass
class ManifestEntry:
    relative_path: str
    size_bytes: int
    sha256: str


def iter_files(base_dir: Path, relative: Path) -> Iterable[Path]:
    """Yield all files under a relative path (recursing into directories)."""
    full_path = base_dir / relative
    if full_path.is_dir():
        for child in sorted(full_path.rglob("*")):
            if child.is_file():
                yield child
    else:
        yield full_path


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path, files: Iterable[Path]) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        entries.append(ManifestEntry(
            relative_path=rel,
            size_bytes=file_path.stat().st_size,
            sha256=sha256sum(file_path),
        ))
    return entries


def write_manifest(zip_file: zipfile.ZipFile, entries: List[ManifestEntry]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "files": [entry.__dict__ for entry in entries],
    }
    zip_file.writestr("MANIFEST.json", json.dumps(manifest, indent=2))


def collect_files(root: Path) -> List[Path]:
    missing: List[str] = []
    collected: List[Path] = []
    for rel_str in CONTENT_PATHS:
        rel_path = Path(rel_str)
        target = root / rel_path
        if not target.exists():
            missing.append(rel_str)
            continue
        collected.extend(iter_files(root, rel_path))
    if missing:
        raise FileNotFoundError(
            "Missing required release assets:\n" + "\n".join(f" - {m}" for m in missing)
        )
    return collected


def package_release(output: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(repo_root)
    entries = build_manifest(repo_root, files)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            file_path = repo_root / entry.relative_path
            zf.write(file_path, arcname=entry.relative_path)
        write_manifest(zf, entries)

    print(f"Created {output} with {len(entries)} files.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/ragwall-healthcare-release.zip"),
        help="Destination archive path (default: dist/ragwall-healthcare-release.zip)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing archive.",
    )
    args = parser.parse_args()

    output = args.output
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output archive {output} already exists. Use --overwrite to replace it."
        )

    package_release(output)


if __name__ == "__main__":
    main()
