"""Utilities for writing table-of-contents bookmarks into PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .utils import ensure_output_path, coerce_positive_int as _util_coerce_positive_int


@dataclass
class BookmarkResult:
    added: int
    skipped: List[str]
    output_path: Path


def write_pdf_toc(
    pdf_path: Path,
    entries: Iterable[dict[str, Any]],
    output_path: Path,
    page_offset: Optional[int] = None,
) -> BookmarkResult:
    """Write *entries* into the PDF at *pdf_path* and save to *output_path*."""

    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError(
            "PyMuPDF (fitz) is required to embed bookmarks into PDFs."
        ) from exc

    resolved_pdf = Path(pdf_path).expanduser().resolve()
    if not resolved_pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {resolved_pdf}")

    dest_path = ensure_output_path(output_path)

    doc = fitz.open(resolved_pdf)  # type: ignore[attr-defined]
    try:
        added = 0
        skipped: List[str] = []
        toc_rows: List[List[Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                skipped.append("Entry is not a dict")
                continue

            title = (entry.get("content") or "").strip()
            if not title:
                skipped.append("Missing title content")
                continue

            target_page = _coerce_page(entry.get("target_page"))
            fallback_page = _coerce_page(entry.get("page"))
            if target_page is None:
                target_page = fallback_page
            if target_page is None:
                skipped.append(f"Invalid page for '{title}'")
                continue

            resolved_page = target_page + page_offset if page_offset is not None else target_page
            if resolved_page <= 0:
                skipped.append(
                    f"Resolved page {resolved_page} out of range for '{title}'"
                )
                continue

            page_index = resolved_page - 1
            if page_index < 0 or page_index >= doc.page_count:
                skipped.append(
                    f"Page out of range for '{title}' -> {resolved_page}"
                )
                continue

            toc_rows.append([1, title, resolved_page])

        success = False
        if toc_rows and hasattr(doc, "set_toc"):
            try:
                doc.set_toc(toc_rows)
                added = len(toc_rows)
                success = True
            except Exception as exc:  # pragma: no cover
                skipped.append(f"set_toc failed: {exc}")

        if not success:
            added = 0
            for (level, title, resolved_page) in toc_rows:
                page_index = resolved_page - 1
                if _add_outline(doc, title, page_index):
                    added += 1
                else:
                    skipped.append(
                        f"Bookmark API unsupported for '{title}'"
                    )

        doc.save(dest_path, deflate=True)
    finally:
        doc.close()

    return BookmarkResult(added=added, skipped=skipped, output_path=dest_path)


def _coerce_page(value: Optional[Any]) -> Optional[int]:
    # Delegate to shared util to keep semantics consistent
    return _util_coerce_positive_int(value)


def _add_outline(doc: Any, title: str, page_index: int) -> bool:
    try:
        doc.add_outline(title, page_index)
        return True
    except AttributeError:
        pass
    except Exception:
        return False

    if hasattr(doc, "new_outline"):
        try:
            doc.new_outline(title, page=page_index)
            return True
        except Exception:
            return False

    if hasattr(doc, "insert_outline"):
        try:
            doc.insert_outline(-1, title=title, page=page_index)
            return True
        except Exception:
            return False

    if hasattr(doc, "insertBookmark"):
        try:
            doc.insertBookmark(title, page_index)
            return True
        except Exception:
            return False

    return False
