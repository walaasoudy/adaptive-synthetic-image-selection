"""Additive local evidence snapshot; never writes to the source evidence tree.

This is NOT a claim of complete reproducibility: remote images/checkpoints must
also be recovered and verified. E4 is deliberately excluded from this copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

COMMITS = {
    "v1_stage345": "b6187d5577f5e51ee2600883f00079c948506da5",
    "stage2": "a1ddb26aef9a271649053c0a3277b600b0eb17a6",
}
SUFFIXES = {".json", ".jsonl", ".csv", ".parquet", ".yaml", ".yml",
            ".toml", ".txt", ".md", ".log", ".npz", ".pt", ".bundle"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(repo: Path, evidence: Path, destination: Path) -> dict:
    repo, evidence, destination = [p.resolve() for p in (repo, evidence, destination)]
    if destination == evidence or evidence in destination.parents:
        raise ValueError("Destination must be outside the historical evidence tree")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "asism-preservation-v2/1", "source": str(evidence),
                "complete_reproducibility": False, "files": [], "errors": [],
                "excluded": "E4/e4_audit, image payloads, unknown extensions; no remote inspection"}

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repo), *args])

    def record(path: Path, source: str) -> None:
        manifest["files"].append({"source": source,
                                  "copy": str(path.relative_to(destination)),
                                  "bytes": path.stat().st_size, "sha256": sha256(path)})

    try:
        commits = {**COMMITS, "audit_head": git("rev-parse", "HEAD").decode().strip()}
        manifest["commits"] = commits
        for name, commit in commits.items():
            path = destination / f"{name}-{commit}.zip"
            git("archive", "--format=zip", f"--output={path}", commit)
            record(path, f"git:{commit}")
        for name, args in {
            "working.patch": ("diff", "--binary", "HEAD"),
            "status.txt": ("status", "--porcelain=v1", "--untracked-files=all"),
            "history.txt": ("log", "--all", "--format=fuller", "--decorate"),
        }.items():
            path = destination / name
            path.write_bytes(git(*args))
            record(path, "git working state")
        # rglob can stop on an inaccessible directory; walk records each error.
        import os
        for source_root, prefix in ((evidence, "evidence"), (repo / "reports", "reports"),
                                    (repo / "research_notes", "research_notes"),
                                    (repo / "docs", "docs")):
            if not source_root.exists():
                manifest["errors"].append(f"Missing source: {source_root}")
                continue
            for folder, dirs, files in os.walk(source_root, onerror=lambda e: manifest["errors"].append(str(e))):
                dirs[:] = [d for d in dirs if d.lower() not in {"e4", "e4_audit"}]
                for name in files:
                    source = Path(folder) / name
                    if source.suffix.lower() not in SUFFIXES:
                        continue
                    target = destination / prefix / source.relative_to(source_root)
                    try:
                        before = sha256(source)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        if sha256(target) != before or sha256(source) != before:
                            raise ValueError(f"Source changed or copy mismatch: {source}")
                        record(target, str(source))
                    except (OSError, ValueError) as exc:
                        manifest["errors"].append(str(exc))
    finally:
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(args.repo, args.evidence, args.destination)
    print(json.dumps({"files": len(result["files"]), "errors": result["errors"]}, indent=2))
