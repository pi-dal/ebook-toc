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


def download_to_temp(
    url: str,
    *,
    prefix: str = "",
    suffix: str = "",
    chunk_size: int = 8192,
    timeout: int = 60,
) -> Path:
    """Download URL content to a temporary file using streaming.

    Parameters
    ----------
    url :
        URL to download.
    prefix, suffix :
        Temporary file name prefix/suffix.
    chunk_size :
        Streaming chunk size in bytes.
    timeout :
        Request timeout in seconds.

    Returns
    -------
    Path
        Path to the downloaded temporary file.

    Raises
    ------
    requests.RequestException
        On network errors.
    """
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
    except TypeError:
        # Backwards-compat for tests or environments where the stubbed
        # requests.get does not accept ``stream``.
        resp = requests.get(url, timeout=timeout)

    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=prefix,
        suffix=suffix,
    )
    try:
        iterator = getattr(resp, "iter_content", None)
        if callable(iterator):
            for chunk in iterator(chunk_size=chunk_size):  # type: ignore[call-arg]
                if chunk:
                    tmp.write(chunk)
        else:
            # Fallback for simple stubs that expose only ``content``.
            tmp.write(getattr(resp, "content", b""))
    finally:
        tmp.close()
        close = getattr(resp, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return Path(tmp.name)
