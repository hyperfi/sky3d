#!/usr/bin/env python3
"""Create immutable, exactly 6000 fm/c protocol prefixes for analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = 6000.0
CASES = ("reference_K0", "K0", "K1", "K2", "K0_half")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truncate(source: Path, target: Path) -> dict[str, object]:
    kept: list[str] = []
    endpoint = None
    source_endpoint = None
    rows = 0
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        value = float(stripped.split()[0])
        source_endpoint = value
        if value <= TARGET + 1.0e-8:
            kept.append(line)
            endpoint = value
            rows += 1
    if endpoint is None or abs(endpoint - TARGET) > 1.0e-8:
        raise ValueError(f"{source} has no exact {TARGET:g} fm/c row; last retained={endpoint}")
    target.write_text("".join(kept), encoding="utf-8")
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "source_endpoint_fm_c": source_endpoint,
        "analysis_endpoint_fm_c": endpoint,
        "analysis_rows": rows,
        "analysis_sha256": sha256(target),
    }


def main() -> None:
    base = ROOT / "response/prefix_runs"
    if base.exists():
        raise SystemExit(f"refusing to overwrite existing analysis prefixes: {base}")
    records = []
    for case in CASES:
        source = ROOT / "td" / case
        target = base / case
        target.mkdir(parents=True)
        shutil.copy2(source / "for005", target / "for005")
        record = truncate(source / "quadrupoles.res", target / "quadrupoles.res")
        record["case"] = case
        record["actual_input_sha256"] = sha256(source / "for005")
        records.append(record)
    (base / "manifest.json").write_text(
        json.dumps({"target_time_fm_c": TARGET, "cases": records}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
