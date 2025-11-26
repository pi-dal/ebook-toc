"""Page fingerprint utilities for aligning PDFs with inserted pages."""

from __future__ import annotations

import concurrent.futures
import hashlib
import threading
from pathlib import Path
from typing import Any, Callable, Iterable


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
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    max_workers: int = 4,
) -> tuple[list[dict[str, Any]], int]:
    """Compute lightweight fingerprints for pages in a PDF.

    Parameters
    ----------
    pdf_path :
        Path to the source PDF file.
    limit :
        Optional upper bound on the number of pages to fingerprint. A value of
        ``0`` or ``None`` means "all pages".
    progress_callback :
        Optional callable receiving ``(completed, total)`` page counts.
    max_workers :
        Maximum number of worker threads to use when fingerprinting pages.

    Returns
    -------
    list[dict[str, Any]], int
        A tuple of ``(fingerprints, total_page_count)`` where ``fingerprints``
        contains one entry per scanned page in order.
    """
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF (fitz) is required for fingerprinting") from exc

    resolved = Path(pdf_path).expanduser().resolve()
    with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
        page_count = doc.page_count

    end_page = page_count if not limit or limit <= 0 else min(limit, page_count)
    if end_page <= 0:
        return [], page_count

    progress_lock = threading.Lock()
    completed = 0

    def _process_chunk(indices: list[int]) -> list[dict[str, Any]]:
        nonlocal completed
        local_results: list[dict[str, Any]] = []
        with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
            for index in indices:
                page = doc.load_page(index)
                text = page.get_text("text").strip()
                fp = build_page_fingerprint(page, text)
                local_results.append(fp)

                if progress_callback is not None:
                    with progress_lock:
                        completed += 1
                        try:
                            progress_callback(completed, end_page)
                        except Exception:
                            # Progress updates are best-effort only.
                            pass

        return local_results

    worker_count = max(1, int(max_workers) if max_workers is not None else 1)
    indices = list(range(end_page))
    if worker_count == 1:
        fingerprints = _process_chunk(indices)
    else:
        # Split work into chunks so each worker opens the document only once.
        # Use a modest chunk size to balance I/O and parallelism.
        chunk_size = max(1, min(32, end_page // worker_count or 1))
        chunks = [indices[i : i + chunk_size] for i in range(0, end_page, chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(_process_chunk, chunks)
        fingerprints = [fp for chunk in results for fp in chunk]

    return fingerprints, page_count
