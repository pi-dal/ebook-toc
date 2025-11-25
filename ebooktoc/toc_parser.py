"""Extract table-of-contents entries from SiliconFlow-generated JSON."""

from __future__ import annotations

from typing import Any, Iterable, Pattern, Set
import re
import difflib
import unicodedata

_TOC_KEYWORDS = ("目录", "contents")


def extract_toc_entries(document: Any) -> list[dict[str, Any]]:
    """Return TOC entries detected in the document JSON.

    SiliconFlow is prompted to return a JSON array of {"page", "content"} objects,
    but we keep compatibility with earlier MinerU-style payloads in case callers feed
    raw OCR results directly.
    """

    if isinstance(document, list):
        return _normalize_entries(document)

    if isinstance(document, dict):
        pages = document.get("pages", [])
        entries: list[dict[str, Any]] = []

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


def _normalize_entries(items: Iterable[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

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

        entry: dict[str, Any] = {"page": page_number, "content": content.strip()}

        target_value = _coerce_optional_int(target_page)
        if target_value is not None:
            entry["target_page"] = target_value

        normalized.append(entry)

    return normalized


def _contains_toc_keyword(text: str) -> bool:
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in _TOC_KEYWORDS)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _normalize_for_comparison(text: str) -> str:
    """Return a normalised string for TOC comparison.

    The normalisation pipeline is designed to treat minor formatting
    differences as equivalent while preserving enough structure to avoid
    over-merging distinct entries.

    Steps
    -----
    text :
        Raw TOC title text.

    The function applies:

    - Unicode normalisation (NFKC) to merge compatibility characters.
    - Lowercasing.
    - Removal of common chapter/section numbering prefixes such as
      "第一章", "第1章", "1.1", or "1 ".
    - Removal of trailing page numbers and dot leaders (e.g. "...... 123").
    """

    s = unicodedata.normalize("NFKC", text)
    s = s.strip().lower()
    if not s:
        return s

    # Strip leading Chinese/Arabic chapter numbering like "第一章", "第1章".
    prefix_patterns = [
        r"^第[一二三四五六七八九十百千0-9]+[章节卷部篇回]\s*",
        r"^[0-9]+(?:[．\.\-、][0-9]+)*\s+",
        r"^[0-9]+\s+",
    ]
    for pat in prefix_patterns:
        s = re.sub(pat, "", s, count=1)

    # Remove trailing page numbers and dot leaders, mirroring infer heuristics.
    s = re.sub(r"(?:[.·\s]{2,})(\d{1,4})\s*$", "", s)
    s = re.sub(r"\s+(\d{1,4})\s*$", "", s)

    return s.strip()


def deduplicate_entries(
    entries: Iterable[dict[str, Any]],
    *,
    fuzzy_threshold: float | None = 0.85,
) -> list[dict[str, Any]]:
    """Deduplicate TOC entries by normalised content and target_page.

    Parameters
    ----------
    entries :
        Iterable of TOC entry mappings with at least ``\"content\"`` and
        ``\"page\"`` keys, and an optional ``\"target_page\"``.
    fuzzy_threshold :
        Optional similarity threshold in the range [0, 1]. When set, a new
        entry is considered a duplicate if its normalised content is within
        ``fuzzy_threshold`` of any of the last 50 accepted entries with the
        same ``target_page`` according to :class:`difflib.SequenceMatcher`.
        When ``None``, only exact normalised matches are deduplicated.
    """

    seen_exact: Set[tuple[str, Any]] = set()
    history: list[tuple[str, Any]] = []
    deduped: list[dict[str, Any]] = []

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
        norm_key = _normalize_for_comparison(normalized_content)
        if not norm_key:
            # Fall back to simple lowercase content when everything was stripped.
            norm_key = normalized_content.lower()

        exact_key = (norm_key, target_page)
        if exact_key in seen_exact:
            continue

        is_duplicate = False
        if fuzzy_threshold is not None:
            # Compare against at most the last 50 accepted items to cap runtime.
            for prev_norm, prev_target in reversed(history[-50:]):
                if prev_target != target_page:
                    continue

                # Fast path: treat short prefix extensions as duplicates
                # (e.g. "绪论" vs "绪论与背景") to handle minor elaborations.
                shorter, longer = (
                    (norm_key, prev_norm)
                    if len(norm_key) <= len(prev_norm)
                    else (prev_norm, norm_key)
                )
                if len(shorter) >= 2 and longer.startswith(shorter):
                    if len(longer) - len(shorter) <= 6:
                        is_duplicate = True
                        break

                ratio = difflib.SequenceMatcher(None, norm_key, prev_norm).ratio()
                if ratio >= fuzzy_threshold:
                    is_duplicate = True
                    break

        if is_duplicate:
            continue

        seen_exact.add(exact_key)
        history.append((norm_key, target_page))
        deduped.append({**entry, "content": normalized_content})

    return deduped


def filter_entries(
    entries: Iterable[dict[str, Any]],
    contains: str | None = None,
    pattern: Pattern[str] | None = None,
) -> list[dict[str, Any]]:
    contains_lc = contains.lower() if contains else None
    filtered: list[dict[str, Any]] = []

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


def infer_missing_targets(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Try to infer missing target_page from the entry content.

    Heuristics:
    - If content ends with dot leaders or spaces followed by digits (e.g., "...... 123"),
      capture that number as the target page.
    - Else, if content ends with bare digits and there are at least two spaces/dots before,
      capture the trailing number.
    We avoid capturing numeric outline parts like "1.2.3" by anchoring to the end and
    requiring separation before the digits.
    """
    inferred: list[dict[str, Any]] = []
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
        page_val: int | None = None
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
