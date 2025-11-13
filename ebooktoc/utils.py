"""Utility helpers for path handling and JSON IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import tempfile
import requests


def ensure_file(path: str | Path) -> Path:
    """Return the path if it exists, otherwise raise FileNotFoundError."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved


def ensure_output_path(path: str | Path) -> Path:
    """Ensure output directory exists and return resolved path."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json(path: str | Path) -> Any:
    """Load JSON from disk using UTF-8 encoding."""

    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(data: Any, path: str | Path) -> None:
    """Write JSON to disk with UTF-8 encoding."""

    with Path(path).open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def coerce_positive_int(value: Any) -> int | None:
    """Return positive int converted from value, else None.

    This normalizes page numbers throughout the project.
    """
    try:
        if value is None:
            return None
        number = int(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def download_to_temp(url: str, *, prefix: str = "", suffix: str = "") -> Path:
    """Download URL content to a temporary file and return its path.

    Raises requests.RequestException on network errors.
    """
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=suffix)
    try:
        tmp.write(resp.content)
        tmp.flush()
    finally:
        tmp.close()
    return Path(tmp.name)
