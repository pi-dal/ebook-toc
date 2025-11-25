"""Page fingerprint utilities for aligning PDFs with inserted pages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


def build_page_fingerprint(page: Any, text: str) -> dict[str, Any]:
    """Return a lightweight fingerprint for *page* given extracted *text*."""

    text_clean = text or ""
    trimmed = text_clean[:2000].encode("utf-8", "ignore")
    text_hash = hashlib.sha1(trimmed).hexdigest() if trimmed else None

    image_count = 0
    try:
        images = page.get_images(full=True)
    except TypeError:
        images = page.get_images()
    except Exception:
        images = []
    if images:
        image_count = len(images)

    return {
        "text_len": len(text_clean),
        "text_hash": text_hash,
        "image_count": image_count,
        "width": round(page.rect.width),
        "height": round(page.rect.height),
    }


## Note: Prior direct baseline→current alignment helpers were removed.
## The project now relies on dominant size + canonical index mapping for robustness.


def dominant_dimensions(fps: Iterable[dict[str, Any]]) -> tuple[int, int] | None:
    counts: dict[tuple[int, int], int] = {}
    for fp in fps:
        w = fp.get("width")
        h = fp.get("height")
        if not w or not h:
            continue
        key = (int(w), int(h))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def build_canonical_map_for_dims(
    fps: list[dict[str, Any]],
    dims: tuple[int, int],
) -> dict[int, int]:
    """Return mapping canonical_index (1-based) -> pdf_page (1-based).

    Canonical index increments only on pages whose width/height match *dims*.
    """
    mapping: dict[int, int] = {}
    canonical = 0
    for pdf_idx, fp in enumerate(fps, start=1):
        w = int(fp.get("width") or 0)
        h = int(fp.get("height") or 0)
        if (w, h) == dims:
            canonical += 1
            mapping[canonical] = pdf_idx
    return mapping


def compute_pdf_fingerprints(
    pdf_path: Path,
    limit: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF (fitz) is required for fingerprinting") from exc

    resolved = Path(pdf_path).expanduser().resolve()
    fingerprints: list[dict[str, Any]] = []
    with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
        page_count = doc.page_count
        end_page = page_count if not limit or limit <= 0 else min(limit, page_count)
        for index in range(end_page):
            page = doc.load_page(index)
            text = page.get_text("text").strip()
            fingerprints.append(build_page_fingerprint(page, text))
            if progress_callback is not None:
                try:
                    progress_callback(index + 1, end_page)
                except Exception:
                    # Progress updates are best-effort only.
                    pass
    return fingerprints, page_count
