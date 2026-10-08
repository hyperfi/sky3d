"""Readers for Sky3D namelists and response protocol files."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np


_GROUP_START_RE = re.compile(r"&(?P<name>[A-Za-z_]\w*)\s*")
_ASSIGNMENT_RE = re.compile(
    r"(?P<key>[A-Za-z_]\w*)\s*=\s*"
    r"(?P<value>(?:\d+\*)?'[^']*'|(?:\d+\*)?\"[^\"]*\"|[^,\s/]+)",
    re.DOTALL,
)


def _strip_fortran_comments(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        quote: str | None = None
        chars: list[str] = []
        for char in line:
            if char in {"'", '"'}:
                quote = None if quote == char else char if quote is None else quote
            if char == "!" and quote is None:
                break
            chars.append(char)
        cleaned.append("".join(chars))
    return "\n".join(cleaned)


def _fortran_value(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {".true.", "t", "true"}:
        return True
    if lowered in {".false.", "f", "false"}:
        return False
    numeric = re.sub(r"[dD]([+-]?\d+)$", r"e\1", value)
    try:
        return int(numeric)
    except ValueError:
        try:
            return float(numeric)
        except ValueError:
            return value


def _iter_groups(text: str):
    """Yield namelist groups, treating only an unquoted slash as terminator."""

    position = 0
    while True:
        match = _GROUP_START_RE.search(text, position)
        if match is None:
            return
        quote: str | None = None
        index = match.end()
        while index < len(text):
            char = text[index]
            if char in {"'", '"'}:
                quote = None if quote == char else char if quote is None else quote
            elif char == "/" and quote is None:
                yield match.group("name"), text[match.end() : index]
                position = index + 1
                break
            index += 1
        else:
            raise ValueError(f"Unterminated namelist group &{match.group('name')}")


def parse_sky3d_input(path: Path) -> dict[str, dict[str, Any]]:
    """Parse scalar metadata from a Sky3D Fortran namelist input."""

    text = _strip_fortran_comments(path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Any]] = {}
    for raw_name, body in _iter_groups(text):
        name = raw_name.lower()
        groups[name] = {
            item.group("key").lower(): _fortran_value(item.group("value"))
            for item in _ASSIGNMENT_RE.finditer(body)
        }
    return groups


def read_response_file(path: Path, column: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Read one numeric observable column from a Sky3D ``*.res`` file.

    ``column`` is zero-based in the numeric table: 0 is time, 1 is the first
    observable (normally IS), and 2 is the second observable (normally IV).
    """

    if column < 1:
        raise ValueError("Response column must be >= 1; column 0 is time")
    header = ""
    rows: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if not header:
                header = line[1:].strip()
            continue
        fields = line.replace("D", "E").replace("d", "e").split()
        if len(fields) <= column:
            raise ValueError(
                f"{path}: requested column {column}, but a row has only {len(fields)} columns"
            )
        rows.append([float(field) for field in fields])
    if len(rows) < 3:
        raise ValueError(f"{path}: at least three numeric samples are required")
    data = np.asarray(rows, dtype=float)
    # A Sky3D restart evaluates and appends the restart point once before the
    # next time step.  Keep the freshly evaluated copy so concatenated
    # protocol files remain a strictly increasing time series.
    duplicate = np.flatnonzero(np.diff(data[:, 0]) == 0.0)
    for index in duplicate:
        if not np.allclose(data[index, 1:], data[index + 1, 1:], rtol=1.0e-8, atol=1.0e-10):
            raise ValueError(
                f"{path}: duplicate time {data[index, 0]:g} has inconsistent values"
            )
    if duplicate.size:
        keep = np.ones(len(data), dtype=bool)
        keep[duplicate] = False
        data = data[keep]
    return data[:, 0], data[:, column], header


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
