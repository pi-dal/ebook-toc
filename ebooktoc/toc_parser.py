"""Extract table-of-contents entries from SiliconFlow-generated JSON."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Pattern, Set
import re

_TOC_KEYWORDS = ("目录", "contents")


def extract_toc_entries(document: Any) -> List[Dict[str, Any]]:
    """Return TOC entries detected in the document JSON.

    SiliconFlow is prompted to return a JSON array of {"page", "content"} objects,
    but we keep compatibility with earlier MinerU-style payloads in case callers feed
    raw OCR results directly.
    """

    if isinstance(document, list):
        return _normalize_entries(document)

    if isinstance(document, dict):
        pages = document.get("pages", [])
        entries: List[Dict[str, Any]] = []

        for index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue

            page_number = (
                page.get("page")
                or page.get("page_number")
                or page.get("pageIndex")
                or page.get("pageNum")
                or index + 1
            )

            for block in page.get("blocks", []):
                if not isinstance(block, dict):
                    continue

                text = (block.get("text") or block.get("content") or "").strip()
                if not text:
                    continue

                block_type = (block.get("block_type") or block.get("type") or "").lower()
                if block_type == "title" or _contains_toc_keyword(text):
                    entries.append({"page": page_number, "content": text})

        return entries

    raise TypeError("Unsupported TOC document format; expected list or dict")


def _normalize_entries(items: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        page = item.get("page")
        content = item.get("content")
        target_page = item.get("target_page")

        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        entry: Dict[str, Any] = {"page": page_number, "content": content.strip()}

        target_value = _coerce_optional_int(target_page)
        if target_value is not None:
            entry["target_page"] = target_value

        normalized.append(entry)

    return normalized


def _contains_toc_keyword(text: str) -> bool:
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in _TOC_KEYWORDS)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def deduplicate_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[tuple[Any, ...]] = set()
    deduped: List[Dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        content = entry.get("content")
        if not isinstance(content, str):
            continue

        normalized_content = content.strip()
        if not normalized_content:
            continue

        target_page = entry.get("target_page")
        key = (normalized_content.lower(), target_page)
        if key in seen:
            continue
        seen.add(key)

        deduped.append({**entry, "content": normalized_content})

    return deduped


def filter_entries(
    entries: Iterable[Dict[str, Any]],
    contains: Optional[str] = None,
    pattern: Optional[Pattern[str]] = None,
) -> List[Dict[str, Any]]:
    contains_lc = contains.lower() if contains else None
    filtered: List[Dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        content = entry.get("content")
        if not isinstance(content, str):
            continue

        text = content.strip()
        if not text:
            continue

        if contains_lc and contains_lc not in text.lower():
            continue

        if pattern and not pattern.search(text):
            continue

        filtered.append(entry)

    return filtered


def infer_missing_targets(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Try to infer missing target_page from the entry content.

    Heuristics:
    - If content ends with dot leaders or spaces followed by digits (e.g., "...... 123"),
      capture that number as the target page.
    - Else, if content ends with bare digits and there are at least two spaces/dots before,
      capture the trailing number.
    We avoid capturing numeric outline parts like "1.2.3" by anchoring to the end and
    requiring separation before the digits.
    """
    inferred: List[Dict[str, Any]] = []
    # Patterns like "..... 123", "··· 45", or whitespace before trailing digits
    tail_num = re.compile(r"(?:[.·\s]{2,})(\d{1,4})\s*$")
    # Fallback: any trailing digits at end if reasonably isolated
    tail_num_loose = re.compile(r"(\d{1,4})\s*$")

    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("target_page") is not None:
            inferred.append(e)
            continue
        content = e.get("content")
        if not isinstance(content, str):
            inferred.append(e)
            continue
        text = content.strip()
        m = tail_num.search(text)
        page_val: Optional[int] = None
        if m:
            try:
                page_val = int(m.group(1))
            except Exception:
                page_val = None
        else:
            m2 = tail_num_loose.search(text)
            if m2 and len(text) >= len(m2.group(1)) + 2:  # ensure some prefix separation
                try:
                    page_val = int(m2.group(1))
                except Exception:
                    page_val = None
        if page_val is not None and page_val > 0:
            adj = dict(e)
            adj["target_page"] = page_val
            inferred.append(adj)
        else:
            inferred.append(e)
    return inferred
