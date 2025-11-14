"""ebook-toc package."""

__all__ = [
    "cli",
    "siliconflow_api",
    "toc_parser",
    "utils",
    "pdf_writer",
    "fingerprints",
]

# Single source of truth: Read version from installed metadata.
# Falls back to parsing pyproject.toml when running from a source checkout
# without an installed distribution (e.g., `python -m ebooktoc.cli`).
from importlib import metadata as _md  # Python 3.10+

try:
    __version__ = _md.version("ebook-toc")
except _md.PackageNotFoundError:  # pragma: no cover - dev-only path
    try:
        import re
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"))
        __version__ = m.group(1) if m else "0.0.0"
    except Exception:
        __version__ = "0.0.0"
