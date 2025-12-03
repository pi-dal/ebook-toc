"""Extract and post-process table-of-contents entries.

This module contains helpers for:

* Normalising and deduplicating TOC entries returned by the VLM.
* Detecting page-numbering segments (for example, preface vs body) in books
  that reset printed page numbers one or more times.
* Inferring and refining printed-page targets for TOC entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Pattern
import re
import difflib
import unicodedata

_TOC_KEYWORDS = ("目录", "contents")

# Shared heuristics for preface/body/appendix classification.
PREFACE_KEYWORDS = [
    "序",
    "前言",
    "引言",
    "致谢",
    "目录",
    "缩写",
    "符号表",
    "preface",
    "foreword",
    "introduction",
    "acknowledgment",
    "contents",
    "abbreviation",
]

BODY_PATTERNS = [
    r"第\s*\d+\s*章",
    r"第\s*[一二三四五六七八九十百]+\s*章",
    r"chapter\s*\d+",
    r"^\d+\.\d+",
]

APPENDIX_KEYWORDS = ["附录", "appendix", "索引", "index", "参考文献", "bibliography"]


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

    seen_exact: set[tuple[str, Any]] = set()
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
    """Infer missing ``target_page`` values from content and context.

    The inference proceeds in two stages:

    1. Extract explicit page numbers from TOC line content when they appear
       at the end of the line (for example, ``\"...... 123\"``). This
       preserves historical behaviour used by older tests and callers.
    2. Mark entries before the first numbered TOC entry as unnumbered
       preface items and attempt to infer ``target_page`` for null entries
       that occur *between* numbered entries (for example, sub-sections that
       omit a page number but clearly belong to the same page as their
       parent).

    Importantly, this function does **not** invent page numbers for
    leading preface entries that lack any numeric hint. Those entries keep
    ``target_page`` as ``None`` and are annotated with ``\"_unnumbered\"``
    so that downstream code (for example, multi-segment offset mapping)
    can treat them specially.

    Parameters
    ----------
    entries :
        Iterable of TOC entry mappings. Each entry should contain at least a
        ``\"content\"`` string and may optionally include ``\"target_page\"``.

    Returns
    -------
    list[dict[str, Any]]
        New list of entry dictionaries with ``\"target_page\"`` and
        auxiliary metadata updated where applicable.
    """
    raw_entries: list[dict[str, Any]] = [
        e for e in entries if isinstance(e, dict)
    ]
    if not raw_entries:
        return []

    # Stage 1: extract trailing page numbers from content where present.
    # Patterns like "..... 123", "··· 45", or whitespace before trailing digits.
    tail_num = re.compile(r"(?:[.·\s]{2,})(\d{1,4})\s*$")
    # Fallback: any trailing digits at end if reasonably isolated.
    tail_num_loose = re.compile(r"(\d{1,4})\s*$")

    items: list[dict[str, Any]] = []
    for e in raw_entries:
        entry = dict(e)
        if entry.get("target_page") is None:
            content = entry.get("content")
            if isinstance(content, str):
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
                    if m2 and len(text) >= len(m2.group(1)) + 2:
                        try:
                            page_val = int(m2.group(1))
                        except Exception:
                            page_val = None
                if page_val is not None and page_val > 0:
                    entry["target_page"] = page_val
        items.append(entry)

    # Stage 2: mark unnumbered prefix entries and infer null targets within
    # numbered sections from neighbouring entries.
    first_numbered_idx: int | None = None
    for idx, entry in enumerate(items):
        tp = entry.get("target_page")
        if tp is None:
            continue
        try:
            int(tp)
        except (TypeError, ValueError):
            continue
        first_numbered_idx = idx
        break

    if first_numbered_idx is None:
        # No numbered entries at all – treat everything as unnumbered preface.
        for i, entry in enumerate(items):
            entry["_unnumbered"] = True
            entry["_unnumbered_index"] = i
        return items

    # Mark entries before the first numbered entry as unnumbered preface.
    if first_numbered_idx > 0:
        for i in range(first_numbered_idx):
            items[i]["_unnumbered"] = True
            items[i]["_unnumbered_index"] = i

    # For null targets after the first numbered entry, try to borrow a page
    # number from neighbours (these are typically sub-entries on the same
    # page).
    for i in range(first_numbered_idx, len(items)):
        entry = items[i]
        if entry.get("target_page") is not None:
            continue
        if entry.get("_unnumbered"):
            continue

        prev_page: int | None = None
        for j in range(i - 1, -1, -1):
            tp = items[j].get("target_page")
            if tp is None:
                continue
            try:
                prev_page = int(tp)
                break
            except (TypeError, ValueError):
                continue

        next_page: int | None = None
        for j in range(i + 1, len(items)):
            tp = items[j].get("target_page")
            if tp is None:
                continue
            try:
                next_page = int(tp)
                break
            except (TypeError, ValueError):
                continue

        if prev_page is not None:
            entry["target_page"] = prev_page
            entry["_inferred"] = True
        elif next_page is not None:
            entry["target_page"] = next_page
            entry["_inferred"] = True

    return items


@dataclass
class PageSegment:
    """A segment of TOC entries sharing the same page-numbering sequence.

    Parameters
    ----------
    start_idx :
        Index of the first entry in the original TOC list belonging to this
        segment (inclusive, zero-based).
    end_idx :
        Index one past the last entry in the original TOC list belonging to
        this segment (exclusive, zero-based).
    entries :
        TOC entries that belong to this segment, in original order.
    segment_type :
        Heuristic label for this segment, one of ``\"preface\"``,
        ``\"body\"``, ``\"appendix\"``, or ``\"unknown\"``.
    offset :
        Optional offset between printed page numbers in this segment and PDF
        page indices, following ``pdf_page_1based = printed_page + offset``.
        This field is populated by VLM-based inference code in :mod:`vlm_api`
        and is left as ``None`` by detection helpers.
    """

    start_idx: int
    end_idx: int
    entries: list[dict[str, Any]]
    segment_type: str
    offset: int | None = None


def detect_page_segments(entries: list[dict[str, Any]]) -> list[PageSegment]:
    """Detect printed page-number segments by finding reset points.

    Many Chinese books use separate page-numbering sequences for preface and
    body sections (and sometimes appendices). For example, a document may
    number preface pages as 1–7, then restart numbering from 1 for the main
    body. This function scans TOC entries for such "resets" where
    ``target_page`` suddenly drops to a much smaller value, and splits the
    entries into contiguous segments accordingly.

    Parameters
    ----------
    entries :
        List of TOC entries with a ``\"target_page\"`` field (when present)
        interpreted as the printed page number shown in the book.

    Returns
    -------
    list[PageSegment]
        Segments in document order, each containing entries that share a
        continuous page-numbering sequence.

    Examples
    --------
    >>> entries = [
    ...     {"content": "序言", "target_page": 1},
    ...     {"content": "前言", "target_page": 3},
    ...     {"content": "第1章", "target_page": 1},  # reset
    ...     {"content": "1.1节", "target_page": 5},
    ... ]
    >>> segments = detect_page_segments(entries)
    >>> len(segments)
    2
    >>> segments[0].segment_type in {"preface", "body"}
    True
    """
    if not entries:
        return []

    pages: list[int | None] = []
    for entry in entries:
        tp = entry.get("target_page")
        if tp is None:
            pages.append(None)
            continue
        try:
            pages.append(int(tp))
        except (TypeError, ValueError):
            pages.append(None)

    # Always start a segment at index 0; subsequent reset indices mark
    # boundaries where printed page numbers drop significantly or repeat.
    reset_indices: list[int] = [0]
    prev_valid_page: int | None = None
    seen_small_pages: set[int] = set()

    for index, page in enumerate(pages):
        if page is None:
            continue

        # Heuristic 1: repeated small page numbers (for example, two separate
        # "page 1" entries) often indicate a reset between preface and body.
        if page <= 5 and page in seen_small_pages:
            reset_indices.append(index)
            seen_small_pages = {page}
        elif page <= 20:
            seen_small_pages.add(page)

        # Heuristic 2: a substantial drop to a small page number, such as
        # 20 -> 1 or 30 -> 5, also indicates a reset.
        if prev_valid_page is not None:
            if prev_valid_page - page >= 3 and page < 20:
                reset_indices.append(index)
                seen_small_pages = {page} if page <= 20 else set()

        prev_valid_page = page

    # Deduplicate and sort reset indices to avoid accidental duplicates.
    unique_indices = sorted(set(reset_indices))

    segments: list[PageSegment] = []
    total_segments = len(unique_indices)

    for seg_idx, start in enumerate(unique_indices):
        end = reset_indices[seg_idx + 1] if seg_idx + 1 < total_segments else len(entries)
        segment_entries = entries[start:end]
        segment_type = _classify_segment(segment_entries, seg_idx, total_segments)
        segments.append(
            PageSegment(
                start_idx=start,
                end_idx=end,
                entries=segment_entries,
                segment_type=segment_type,
            )
        )

    # If the very first segment begins with entries that have no target_page
    # and have been marked as ``_unnumbered`` by :func:`infer_missing_targets`,
    # split out that prefix as a dedicated unnumbered preface segment so that
    # it can receive its own offset later.
    if segments:
        first = segments[0]
        prefix_len = 0
        for entry in first.entries:
            if entry.get("target_page") is None and entry.get("_unnumbered"):
                prefix_len += 1
            else:
                break

        if prefix_len > 0:
            if prefix_len == len(first.entries):
                # Entire first segment is unnumbered.
                first.segment_type = "preface_unnumbered"
            else:
                pre_entries = first.entries[:prefix_len]
                rest_entries = first.entries[prefix_len:]
                pre_segment = PageSegment(
                    start_idx=first.start_idx,
                    end_idx=first.start_idx + prefix_len,
                    entries=pre_entries,
                    segment_type="preface_unnumbered",
                )
                rest_segment = PageSegment(
                    start_idx=first.start_idx + prefix_len,
                    end_idx=first.end_idx,
                    entries=rest_entries,
                    segment_type=_classify_segment(rest_entries, 1, max(2, total_segments)),
                )
                segments = [pre_segment, rest_segment, *segments[1:]]

    return segments


def _classify_segment(
    entries: list[dict[str, Any]],
    segment_index: int,
    total_segments: int,
) -> str:
    """Classify a page segment as preface, body, appendix, or unknown.

    Classification is heuristic and based on:

    * Keywords such as ``\"序\"``, ``\"前言\"``, or ``\"附录\"``.
    * Chapter-like patterns (for example, ``\"第1章\"``, ``\"chapter 2\"``).
    * The segment's position (first, middle, last) within the document.

    Parameters
    ----------
    entries :
        TOC entries belonging to the segment.
    segment_index :
        Zero-based index of this segment among all detected segments.
    total_segments :
        Total number of detected segments.

    Returns
    -------
    str
        One of ``\"preface\"``, ``\"body\"``, ``\"appendix\"``, or
        ``\"unknown\"``.
    """
    if not entries:
        return "unknown"

    contents = [str(entry.get("content", "")).lower() for entry in entries]
    all_text = " ".join(contents)

    preface_score = sum(1 for kw in PREFACE_KEYWORDS if kw in all_text)
    body_score = sum(1 for pattern in BODY_PATTERNS if re.search(pattern, all_text, re.IGNORECASE))
    appendix_score = sum(1 for kw in APPENDIX_KEYWORDS if kw in all_text)

    # Positional hints: first segment is likely preface; last segment with
    # appendix keywords is more likely an appendix.
    if segment_index == 0 and total_segments > 1:
        preface_score += 2
    if segment_index == total_segments - 1 and appendix_score > 0:
        appendix_score += 1

    if body_score > preface_score and body_score > appendix_score:
        return "body"
    if preface_score > body_score and preface_score >= appendix_score:
        return "preface"
    if appendix_score > 0:
        return "appendix"

    # Fallback: if uncertain, treat the first segment in a multi-segment TOC
    # as preface and others as body.
    if segment_index == 0 and total_segments > 1:
        return "preface"
    return "body"


def merge_segments_with_offsets(
    segments: list[PageSegment],
    page_count: int = 0,
    resolved_override: dict[int, int | None] | None = None,
) -> list[dict[str, Any]]:
    """Merge segments back into a flat list with resolved ``target_page`` values.

    Each entry's ``\"target_page\"`` is adjusted using the segment's offset so
    that the resulting value represents a PDF 1-based page index. For
    unnumbered preface segments, entries are assigned sequential pages
    starting from the segment's offset when it is known.

    The function attaches helpful metadata to each entry:

    * ``\"_original_target_page\"`` – the original printed page number (or
      ``None`` for unnumbered entries).
    * ``\"_segment_type\"`` – the segment classification label.
    * ``\"_segment_offset\"`` – the offset applied for this segment.
    * ``\"_needs_manual_offset\"`` – set when an unnumbered entry could not
      be resolved to a concrete PDF page.

    Parameters
    ----------
    segments :
        Segments with :attr:`PageSegment.offset` populated where available.
    page_count :
        Total number of pages in the PDF. When greater than zero, resolved
        pages are clamped into the inclusive range ``[1, page_count]``.

    Returns
    -------
    list[dict[str, Any]]
        Flat list of entries with adjusted ``\"target_page\"`` values and
        attached metadata.
    """
    result: list[dict[str, Any]] = []
    override = resolved_override or {}

    for segment in segments:
        offset = segment.offset if segment.offset is not None else 0

        for idx, entry in enumerate(segment.entries):
            new_entry = dict(entry)
            new_entry["_segment_type"] = segment.segment_type
            new_entry["_segment_offset"] = offset

            original_index = segment.start_idx + idx
            if original_index in override and override[original_index] is not None:
                # Directly resolved physical page overrides any offset logic.
                resolved_page = int(override[original_index])  # type: ignore[arg-type]
                if page_count > 0:
                    resolved_page = max(1, min(resolved_page, page_count))
                new_entry["_original_target_page"] = entry.get("target_page")
                new_entry["target_page"] = resolved_page
                new_entry["_resolved_by"] = "direct_location"
                result.append(new_entry)
                continue

            original_tp = entry.get("target_page")
            try:
                page_val = int(original_tp) if original_tp is not None else None
            except (TypeError, ValueError):
                page_val = None

            if segment.segment_type == "preface_unnumbered" and entry.get("_unnumbered"):
                # Unnumbered preface entries: when we know an offset, map them
                # to consecutive PDF pages starting at ``offset + 1``.
                if offset != 0 and page_count != 0:
                    new_entry["_original_target_page"] = None
                    resolved = offset + idx + 1
                    if page_count > 0:
                        resolved = max(1, min(resolved, page_count))
                    new_entry["target_page"] = resolved
                else:
                    # Offset unknown – keep target_page as None and flag for
                    # manual review.
                    new_entry["_original_target_page"] = None
                    new_entry["_needs_manual_offset"] = True
            elif page_val is not None:
                # Regular numbered entry; adjust by segment offset.
                new_entry["_original_target_page"] = page_val
                resolved = page_val + offset
                if page_count > 0:
                    resolved = max(1, min(resolved, page_count))
                new_entry["target_page"] = resolved
            else:
                # Null target_page inside a numbered segment – preserve as-is.
                new_entry["_original_target_page"] = None

            result.append(new_entry)

    return result


@dataclass
class AnchorEntry:
    """Anchor entry used for multi-point offset estimation.

    Parameters
    ----------
    index :
        Index of the entry in the original TOC list.
    content :
        Title/content text for the entry.
    printed_page :
        Printed page number taken from ``target_page``.
    pdf_page :
        Resolved 1-based PDF page index for this entry, when known.
    offset :
        Difference ``pdf_page - printed_page`` once both are known.
    located_by :
        How the anchor was located, for example ``\"text\"`` or ``\"vlm\"``.
    """

    index: int
    content: str
    printed_page: int
    pdf_page: int | None = None
    offset: int | None = None
    located_by: str | None = None


def _is_preface_like_content(content: str) -> bool:
    """Return ``True`` if *content* looks like preface/foreword material."""
    text = str(content).lower()
    # Avoid treating obvious chapter headings as preface even if they
    # contain generic words like "introduction".
    if any(re.search(pat, text, re.IGNORECASE) for pat in BODY_PATTERNS):
        return False
    return any(kw in text for kw in PREFACE_KEYWORDS)


def _appears_before_first_chapter(
    entries: list[dict[str, Any]],
    index: int,
) -> bool:
    """Return ``True`` if entry at *index* is before the first chapter-like item."""
    chapter_patterns = [
        r"第\s*\d+\s*章",
        r"第\s*[一二三四五六七八九十百]+\s*章",
        r"chapter\s*\d+",
        r"^\d+\.\d+",
    ]

    # If a chapter appears before this index, it is not in the preface area.
    for i, entry in enumerate(entries):
        if i >= index:
            break
        text = str(entry.get("content", ""))
        if any(re.search(pat, text, re.IGNORECASE) for pat in chapter_patterns):
            return False

    # If there is a chapter after this entry, treat this as pre-body material.
    for i, entry in enumerate(entries):
        if i <= index:
            continue
        text = str(entry.get("content", ""))
        if any(re.search(pat, text, re.IGNORECASE) for pat in chapter_patterns):
            return True

    return False


def classify_entries(
    entries: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[int]]:
    """Classify TOC entries into unnumbered, suspicious, and normal buckets.

    Parameters
    ----------
    entries :
        TOC entries with ``\"target_page\"`` and ``\"content\"`` fields.

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        Three lists of entry indices ``(unnumbered, suspicious, normal)``.
    """
    unnumbered: list[int] = []
    suspicious: list[int] = []
    normal: list[int] = []

    page_counts: dict[int, int] = {}
    for entry in entries:
        tp = entry.get("target_page")
        if tp is None:
            continue
        try:
            page = int(tp)
        except (TypeError, ValueError):
            continue
        page_counts[page] = page_counts.get(page, 0) + 1

    for i, entry in enumerate(entries):
        tp = entry.get("target_page")
        content = str(entry.get("content", ""))

        if tp is None:
            unnumbered.append(i)
            continue

        try:
            page = int(tp)
        except (TypeError, ValueError):
            unnumbered.append(i)
            continue

        is_suspicious = False

        # Many entries sharing the same small page number that also look
        # preface-like are strong candidates for fabricated page numbers.
        if page_counts.get(page, 0) > 3 and page < 10 and _is_preface_like_content(content):
            is_suspicious = True

        # Preface-style content with very small page numbers that appears
        # before the first chapter is also suspicious.
        if page < 5 and _is_preface_like_content(content):
            if _appears_before_first_chapter(entries, i):
                is_suspicious = True

        if is_suspicious:
            suspicious.append(i)
        else:
            normal.append(i)

    return unnumbered, suspicious, normal


def _calculate_anchor_priority(content: str, printed_page: int) -> int:
    """Return a priority score used when selecting anchor entries."""
    priority = 0

    # Strongly prefer chapter headings.
    if any(re.search(pat, content, re.IGNORECASE) for pat in BODY_PATTERNS[:3]):
        priority += 100

    # Prefer section headings like "1.1 Title".
    if re.match(r"^\d+(?:\.\d+)+\s+", content):
        priority += 50

    # Prefer longer, more distinctive titles.
    priority += min(len(content), 30)

    # Prefer anchors with larger printed page numbers. Very small page
    # numbers are harder to localise because the corresponding content
    # may appear anywhere in the front matter.
    if printed_page > 100:
        priority += 50
    elif printed_page > 50:
        priority += 40
    elif printed_page > 20:
        priority += 30
    elif printed_page > 10:
        priority += 20
    elif printed_page > 5:
        priority += 10
    else:
        priority -= 20

    return priority


def select_anchor_entries(
    entries: list[dict[str, Any]],
    normal_indices: list[int],
    max_anchors: int = 5,
) -> list[AnchorEntry]:
    """Select strong anchor entries for multi-point offset estimation.

    Parameters
    ----------
    entries :
        Full list of TOC entries.
    normal_indices :
        Indices of entries considered normal (non-suspicious) by
        :func:`classify_entries`.
    max_anchors :
        Maximum number of anchors to select.

    Returns
    -------
    list[AnchorEntry]
        Selected anchors sorted by printed page number.
    """
    if not normal_indices:
        return []

    candidates: list[tuple[int, str, int, int]] = []
    for idx in normal_indices:
        entry = entries[idx]
        content = str(entry.get("content", "")).strip()
        tp = entry.get("target_page")
        if not content or tp is None:
            continue
        try:
            printed_page = int(tp)
        except (TypeError, ValueError):
            continue
        if printed_page <= 0:
            continue

        priority = _calculate_anchor_priority(content, printed_page)
        candidates.append((idx, content, printed_page, priority))

    if not candidates:
        return []

    # Sort by priority descending.
    candidates.sort(key=lambda item: item[3], reverse=True)

    selected: list[AnchorEntry] = []
    page_ranges_covered: set[int] = set()

    for idx, content, printed_page, priority in candidates:
        if len(selected) >= max_anchors:
            break
        # Bucket by 50-page ranges to encourage spread.
        bucket = printed_page // 50
        if bucket not in page_ranges_covered or len(selected) < 2:
            selected.append(
                AnchorEntry(
                    index=idx,
                    content=content,
                    printed_page=printed_page,
                )
            )
            page_ranges_covered.add(bucket)

    selected.sort(key=lambda anchor: anchor.printed_page)
    return selected


def _normalize_text_for_search(text: str) -> str:
    """Normalise page text for substring search."""
    result = re.sub(r"\s+", "", text)
    result = result.lower()
    result = re.sub(r'[·・．。，、：；！？"\'【】（）\[\](){}「」『』]', "", result)
    return result.strip()


def _build_search_terms(content: str) -> list[str]:
    """Return a list of search terms derived from *content*."""
    terms: list[str] = []
    raw = content.strip()
    if not raw:
        return terms

    full_norm = _normalize_text_for_search(raw)
    if len(full_norm) >= 2:
        terms.append(full_norm)

    # Try to strip common chapter/section prefixes.
    patterns = [
        r"^第\s*\d+\s*章\s*(.+)$",
        r"^第\s*[一二三四五六七八九十百]+\s*章\s*(.+)$",
        r"^chapter\s*\d+\s*(.+)$",
        r"^\d+(?:\.\d+)+\s+(.+)$",
        r"^section\s*\d+\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            tail = m.group(1).strip()
            if tail:
                tail_norm = _normalize_text_for_search(tail)
                if len(tail_norm) >= 2 and tail_norm not in terms:
                    terms.append(tail_norm)
            break

    # Also search just for the chapter marker when present.
    chapter_match = re.match(
        r"^(第\s*\d+\s*章|第\s*[一二三四五六七八九十百]+\s*章|chapter\s*\d+)",
        raw,
        re.IGNORECASE,
    )
    if chapter_match:
        marker = _normalize_text_for_search(chapter_match.group(1))
        if marker and marker not in terms:
            terms.append(marker)

    return terms


def locate_entry_by_text_search(
    pdf_path: Path,
    content: str,
    search_start: int = 1,
    search_end: int | None = None,
) -> int | None:
    """Locate an entry in the PDF using text search alone.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    content :
        Entry title/content to search for.
    search_start :
        1-based index of the first page to search (inclusive).
    search_end :
        1-based index of the last page to search (inclusive). When omitted,
        searches at most the first 100 pages.

    Returns
    -------
    int | None
        The 1-based PDF page index where a matching term was found, or
        ``None`` when no match exists in the given range.
    """
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return None

    terms = _build_search_terms(content)
    if not terms:
        return None

    resolved = Path(pdf_path).expanduser().resolve()
    try:
        with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
            page_count = doc.page_count
            if page_count == 0:
                return None

            start_idx = max(0, search_start - 1)
            end_page = search_end if search_end is not None else min(page_count, 100)
            end_idx = max(start_idx, min(page_count, end_page))

            for page_idx in range(start_idx, end_idx):
                page = doc.load_page(page_idx)
                text = page.get_text("text")
                if not text:
                    continue
                norm = _normalize_text_for_search(text)
                if any(term and term in norm for term in terms):
                    return page_idx + 1
    except Exception:
        return None

    return None
