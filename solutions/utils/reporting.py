"""Utility helpers for writing markdown reports and saving structured outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_dir(path: Path | str) -> Path:
    """Ensure that a directory exists and return it as a :class:`Path`."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_markdown(path: Path | str, lines: Iterable[str]) -> None:
    """Write a markdown file from an iterable of lines."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_table_md(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Return a GitHub-flavoured-markdown table."""
    header_line = " | ".join(headers)
    separator_line = " | ".join(["---"] * len(headers))
    row_lines = [
        " | ".join(str(value) for value in row)
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def dump_json(path: Path | str, payload: Mapping[str, object]) -> None:
    """Persist a dictionary as pretty-printed JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

