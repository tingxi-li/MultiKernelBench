#!/usr/bin/env python3
"""Capture and verify immutable provenance for a follow-up campaign.

The script is standard-library-only.  It hashes source/configuration inputs and
records version-control, Python/toolchain, and GPU metadata.  Historical result
trees and build caches are deliberately excluded.

Usage:
  python capture.py snapshot --campaign ID --out snapshots/ID.json
  python capture.py verify snapshots/ID.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_INPUTS = (
    "ako_runs/CONTROLLED_CROSS_DSL_REPORT.md",
    "ako_runs/CONTROLLED_CROSS_DSL_REVIEW.md",
    "ako_runs/CONVERGENCE_PROTOCOL.md",
    "ako_runs/phase1_matmul",
    "ako_runs/phase2_fused_sdpa",
    "ako_runs/controlled_followup",
)
EXCLUDED_DIRS = {
    ".git", "__pycache__", ".torch_ext", "results", "inputs", "artifacts",
    "snapshots", "raw", "binaries",
}
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".json", ".md",
    ".jsonl", ".py", ".sh", ".toml", ".yaml", ".yml",
}


def _run(argv: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(
            argv, cwd=REPO, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _eligible(path: Path) -> bool:
    rel = path.relative_to(REPO)
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return False
    return path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES


def _resolve_files(inputs: list[str]) -> list[Path]:
    files: set[Path] = set()
    for item in inputs:
        path = (REPO / item).resolve()
        try:
            path.relative_to(REPO)
        except ValueError as exc:
            raise SystemExit(f"input escapes repository: {item}") from exc
        if not path.exists():
            raise SystemExit(f"input does not exist: {item}")
        if path.is_file():
            if _eligible(path):
                files.add(path)
        else:
            files.update(p for p in path.rglob("*") if _eligible(p))
    return sorted(files, key=lambda p: p.relative_to(REPO).as_posix())


def _packages() -> dict[str, str | None]:
    names = ("torch", "triton", "tilelang", "numpy", "scipy")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        f.write(encoded)
        tmp = Path(f.name)
    os.replace(tmp, path)


def snapshot(campaign: str, out: Path, inputs: list[str]) -> dict:
    files = _resolve_files(inputs)
    hashes = {
        p.relative_to(REPO).as_posix(): {
            "sha256": _sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in files
    }
    git_head = _run(["git", "rev-parse", "HEAD"])
    git_branch = _run(["git", "branch", "--show-current"])
    git_status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    payload = {
        "schema_version": 1,
        "campaign_id": campaign,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": str(REPO),
        "git": {
            "head": git_head.get("stdout"),
            "branch": git_branch.get("stdout"),
            "status_porcelain": git_status.get("stdout", "").splitlines(),
            "dirty": bool(git_status.get("stdout")),
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "packages": _packages(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "tool_queries": {
            "nvidia_smi": _run([
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id",
                "--format=csv,noheader",
            ]),
            "nvcc": _run(["nvcc", "--version"]),
        },
        "inputs": list(inputs),
        "files": hashes,
    }
    _atomic_json(out, payload)
    return payload


def verify(manifest: Path) -> int:
    data = json.loads(manifest.read_text())
    changed, missing = [], []
    for rel, expected in data["files"].items():
        path = REPO / rel
        if not path.is_file():
            missing.append(rel)
        elif _sha256(path) != expected["sha256"]:
            changed.append(rel)
    print(json.dumps({
        "manifest": str(manifest),
        "campaign_id": data.get("campaign_id"),
        "checked": len(data["files"]),
        "missing": missing,
        "changed": changed,
        "ok": not missing and not changed,
    }, indent=2, sort_keys=True))
    return 0 if not missing and not changed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--campaign", required=True)
    snap.add_argument("--out", type=Path, required=True)
    snap.add_argument("--input", action="append", dest="inputs")
    check = sub.add_parser("verify")
    check.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        inputs = args.inputs or list(DEFAULT_INPUTS)
        payload = snapshot(args.campaign, args.out, inputs)
        print(json.dumps({
            "campaign_id": payload["campaign_id"],
            "out": str(args.out),
            "files": len(payload["files"]),
            "git_dirty": payload["git"]["dirty"],
            "gpu_query_ok": payload["tool_queries"]["nvidia_smi"].get("returncode") == 0,
        }, indent=2, sort_keys=True))
        return 0
    return verify(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
