"""SiliconFlow Qwen API client for extracting TOC data from PDFs."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .fingerprints import (
    build_page_fingerprint,
    dominant_dimensions,
    build_canonical_map_for_dims,
)
from .utils import coerce_positive_int as _util_coerce_positive_int, download_to_temp as _util_download_to_temp

CHAT_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_BATCH_SIZE = 3
JPEG_QUALITY = 80
# Cache maps (pdf_path, index) -> (payload_entry, fingerprint_dict)
_PAYLOAD_CACHE: Dict[Tuple[str, int], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = {}
_PAGE_NUMBER_CACHE: Dict[Tuple[str, int], Optional[str]] = {}
_CACHE_LOCK = threading.Lock()


class TOCExtractionError(RuntimeError):
    """Raised when the external TOC extraction service fails."""


def fetch_document_json(
    pdf_path: Optional[Path],
    api_key: str,
    poll_interval: int = 5,  # kept for CLI compatibility; unused
    timeout: int = 600,  # kept for CLI compatibility; unused
    page_limit: int = 15,
    remote_url: Optional[str] = None,
    task_id: Optional[str] = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    start_page: int = 1,
) -> Path:
    """Extract TOC data using SiliconFlow's Qwen/Qwen3-VL-32B-Instruct model.

    Returns a temporary JSON file path containing the model output.
    """

    if task_id is not None:
        raise TOCExtractionError("SiliconFlow workflow does not support --task-id")

    source_path, cleanup_required = _resolve_source_pdf(pdf_path, remote_url)
    try:
        page_payloads, fingerprints = _collect_page_payloads(
            source_path, page_limit, start_page=start_page
        )
        aggregated: List[Dict[str, Any]] = []

        effective_batch = max(1, batch_size)
        batches: List[List[Dict[str, Any]]] = list(
            _chunk_iterable(page_payloads, effective_batch)
        )

        if batches:
            max_attempts = 3

            def _fetch_toc_for_batch(
                batch: List[Dict[str, Any]]
            ) -> List[Dict[str, Any]]:
                attempts = 0
                last_error: Optional[Exception] = None
                while attempts < max_attempts:
                    attempts += 1
                    payload = _build_payload(
                        batch, page_limit, start_page=start_page
                    )
                    try:
                        response_body = _call_chat_completion(
                            api_key, payload, request_timeout=timeout
                        )
                        return _parse_response_payload(response_body)
                    except TOCExtractionError as exc:
                        last_error = exc
                        if attempts >= max_attempts or not _is_retryable_error(exc):
                            raise
                        # Simple exponential backoff capped at 5 seconds
                        sleep_seconds = min(2 ** (attempts - 1), 5)
                        time.sleep(sleep_seconds)

                # Should not be reached because we either return or raise above
                if last_error is not None:
                    raise last_error
                return []

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                for toc_json in executor.map(_fetch_toc_for_batch, batches):
                    aggregated.extend(toc_json)

        offset = None
        try:
            offset = _infer_page_offset(
                source_path,
                aggregated,
                api_key,
                timeout,
                fingerprints,
            )
        except TOCExtractionError:
            offset = None

        # Build a canonical page map for this source PDF (based on dominant dims)
        page_map: Dict[int, int] = {}
        dims = dominant_dimensions(fingerprints) if fingerprints else None
        if dims:
            page_map = build_canonical_map_for_dims(fingerprints, dims)

        packaged = {
            "toc": aggregated,
            "page_offset": offset,
            "fingerprints": fingerprints,
            "page_map": page_map,
        }

        return _write_temp_json(packaged)
    finally:
        if cleanup_required:
            _safe_unlink(source_path)
        _purge_cache_for_path(source_path)


def _resolve_source_pdf(
    pdf_path: Optional[Path], remote_url: Optional[str]
) -> Tuple[Path, bool]:
    if remote_url:
        temp_path = _download_remote_pdf(remote_url)
        return temp_path, True
    if not pdf_path:
        raise TOCExtractionError("Provide a local PDF path or --remote-url")
    resolved = Path(pdf_path).expanduser().resolve()
    if not resolved.is_file():
        raise TOCExtractionError(f"File not found: {resolved}")
    return resolved, False


def _download_remote_pdf(url: str) -> Path:
    try:
        return _util_download_to_temp(url, prefix="siliconflow-", suffix=".pdf")
    except requests.RequestException as exc:
        raise TOCExtractionError(f"Failed to download remote PDF: {exc}") from exc


def _collect_page_payloads(
    pdf_path: Path,
    max_pages: int,
    *,
    start_page: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise TOCExtractionError(
            "PyMuPDF (fitz) is required to extract content from PDFs."
        ) from exc

    resolved = pdf_path.resolve()
    with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
        total_pages = doc.page_count

    if total_pages == 0:
        raise TOCExtractionError("PDF contains no pages")

    # Convert 1-based start_page to 0-based index and clamp at 0
    start_index = max(0, start_page - 1)
    if max_pages <= 0:
        end_page = total_pages
    else:
        end_page = min(total_pages, start_index + max_pages)

    # When the requested window lies entirely past the end of the document,
    # treat this as "no pages left to scan" rather than a hard failure.
    if start_index >= end_page:
        return [], []

    payloads: List[Dict[str, Any]] = []
    fingerprints: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_get_or_render_page_payload, resolved, index)
            for index in range(start_index, end_page)
        ]

        for future in futures:
            entry, fingerprint = future.result()
            if entry:
                payloads.append(entry)
            if fingerprint:
                fingerprints.append(fingerprint)

    if not payloads:
        raise TOCExtractionError(
            "Unable to extract text or render images from the supplied PDF"
        )

    return payloads, fingerprints


def _build_payload(
    page_payloads: List[Dict[str, Any]],
    max_pages: int,
    start_page: int = 1,
) -> Dict[str, Any]:
    instructions = (
        "You are an assistant that extracts a book's table of contents from PDF content. "
        "For every TOC line, output an object with keys: "
        "'page' (integer, the source page number indicated by the excerpt header such as [Page 4]), "
        "'target_page' (integer or null, the destination page referenced inside the TOC line), and "
        "'content' (string, the title text without trailing page numbers or dot leaders). "
        "If a TOC line omits a destination page number, set 'target_page' to null. "
        "Respond with a JSON object in the form {\"toc\": [...]} where the array contains only these objects. "
        "Do not include explanations, prose, markdown code fences, or any text outside this JSON object. "
        "If no TOC entries are present, respond with {\"toc\": []}."
    )

    if max_pages <= 0:
        if start_page <= 1:
            range_label = "the entire document"
        else:
            range_label = f"pages {start_page} and later"
    else:
        if start_page <= 1:
            range_label = f"the first {max_pages} pages"
        else:
            end_page = start_page + max_pages - 1
            range_label = f"pages {start_page} to {end_page}"

    intro_text = (
        f"The following content comes from {range_label} of a PDF. Each section begins "
        "with a header like [Page 4]. Analyze the provided text or image for each section, identify "
        "the table-of-contents entries, and extract the title along with any target page number mentioned "
        "in the line. Return JSON exactly as described."
    )

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": intro_text}]

    for payload in page_payloads:
        page_number = payload["page"]
        user_content.append(
            {
                "type": "text",
                "text": f"[Page {page_number}] Analyze the following content:",
            }
        )

        if "text" in payload:
            user_content.append({"type": "text", "text": payload["text"]})
        elif "image_b64" in payload:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        # Rendered as JPEG downstream
                        "url": f"data:image/jpeg;base64,{payload['image_b64']}"
                    },
                }
            )

    user_content.append(
        {
            "type": "text",
            "text": (
                "Example output: {\"toc\": [{\"page\": 4, \"target_page\": 5, \"content\": \"Chapter 1\"}]}. "
                "Return ONLY the JSON object described above. Do not add extra commentary or formatting."
            ),
        }
    )

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content},
    ]

    return {
        "model": MODEL_NAME,
        "temperature": 0.2,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }


def _call_chat_completion(
    api_key: str, payload: Dict[str, Any], request_timeout: int
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            CHAT_ENDPOINT, json=payload, headers=headers, timeout=request_timeout
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = _safe_json(exc.response) if exc.response is not None else str(exc)
        raise TOCExtractionError(f"SiliconFlow error: {detail}") from exc
    except requests.RequestException as exc:
        raise TOCExtractionError(f"Failed to call SiliconFlow: {exc}") from exc

    body = response.json()
    if not isinstance(body, dict):
        raise TOCExtractionError("Unexpected SiliconFlow response format")
    return body


def _is_retryable_error(exc: TOCExtractionError) -> bool:
    cause = exc.__cause__
    if isinstance(cause, requests.HTTPError) and cause.response is not None:
        status = cause.response.status_code
        if status == 429 or 500 <= status < 600:
            return True
    return False


def _parse_response_payload(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = body.get("choices")
    if not choices:
        raise TOCExtractionError(f"SiliconFlow response missing choices: {body}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise TOCExtractionError(f"SiliconFlow response missing text content: {message}")

    json_text = _extract_json_block(content)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        # Best-effort repair for slightly invalid JSON coming back from the model.
        # In practice we sometimes see unquoted or single-quoted keys like
        #   {page: 11, target_page: 67, content: "Title"}
        # even though the response is supposed to be strict JSON. When this happens
        # we attempt to quote the known keys and parse again; if that still fails we
        # surface the original error.
        repaired = _repair_toc_json(json_text)
        if repaired is None:
            raise TOCExtractionError(
                f"Failed to parse SiliconFlow JSON output: {exc}\nRaw: {json_text}"
            ) from exc
        data = repaired

    toc: Any
    if isinstance(data, dict):
        if "toc" in data:
            toc = data["toc"]
        elif "output" in data and isinstance(data["output"], list):
            toc = data["output"]
        # Some SiliconFlow responses occasionally return a single TOC-like
        # object instead of wrapping it in a {\"toc\": [...]} container.
        # When we see a mapping that looks like one TOC entry, treat it as a
        # singleton list rather than failing hard.
        elif "page" in data and "content" in data:
            toc = [data]
        else:
            raise TOCExtractionError(
                f"SiliconFlow response missing 'toc' key: {json_text}"
            )
    elif isinstance(data, list):
        toc = data
    else:
        raise TOCExtractionError(
            f"SiliconFlow response must be a JSON array or object with 'toc'; received {type(data).__name__}."
        )

    if not isinstance(toc, list):
        raise TOCExtractionError("The 'toc' field must be a JSON array.")

    return toc


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        fence_end = stripped.find("```", 3)
        if fence_end != -1:
            stripped = stripped[3:fence_end].strip()
            if stripped.lower().startswith("json\n"):
                stripped = stripped[5:]

    candidate = _find_json_substring(stripped)
    return candidate if candidate is not None else stripped


def _find_json_substring(text: str) -> Optional[str]:
    """Return the first JSON object/array substring found in text.

    This implementation is tolerant of leading/trailing commentary and avoids
    being confused by brackets that appear inside string literals by delegating
    to the JSON decoder's raw_decode at different candidate offsets.
    """
    decoder = json.JSONDecoder()

    # Fast path: the whole string (after trimming) is valid JSON.
    stripped = text.strip()
    try:
        _, end = decoder.raw_decode(stripped)
        return stripped[:end]
    except json.JSONDecodeError:
        pass

    # General path: search for the first plausible JSON opener and let
    # raw_decode determine where the JSON payload ends.
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return text[idx : idx + end]

    return None


def _repair_toc_json(json_text: str) -> Optional[Any]:
    """Best-effort repair for slightly invalid TOC JSON.

    SiliconFlow is instructed (and configured via response_format) to return
    strict JSON, but in practice the model may occasionally emit structures
    with unquoted or single-quoted keys such as:

        {page: 11, target_page: 67, content: "Title"}
        {'page': 11, 'content': "Title"}

    This helper walks the string outside of double-quoted regions and
    rewrites the known TOC-related keys (\"toc\", \"page\", \"target_page\",
    \"content\") into properly double-quoted JSON keys. If parsing still fails
    we give up and let the caller surface the original error.
    """

    keys = ("toc", "page", "target_page", "content")
    result: List[str] = []
    in_string = False
    escape = False
    i = 0
    length = len(json_text)

    while i < length:
        ch = json_text[i]

        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue

        # Handle single-quoted keys like 'page': 1
        if ch == "'":
            matched = False
            for key in keys:
                key_len = len(key)
                start = i + 1
                end = start + key_len
                if end > length:
                    continue
                if json_text[start:end] != key:
                    continue
                # Require a closing single quote immediately after the key
                if end >= length or json_text[end] != "'":
                    continue
                # Require a colon after optional whitespace
                j = end + 1
                while j < length and json_text[j].isspace():
                    j += 1
                if j >= length or json_text[j] != ":":
                    continue

                # Rewrite 'key' as "key"
                result.append('"')
                result.append(key)
                result.append('"')
                i = end + 1
                matched = True
                break

            if matched:
                continue

            # Not a recognized key pattern; keep the single quote as-is.
            result.append(ch)
            i += 1
            continue

        # Handle bare keys like page: 1 or target_page: 2
        if ch.isalpha():
            matched = False
            for key in keys:
                key_len = len(key)
                end = i + key_len
                if end > length or json_text[i:end] != key:
                    continue
                # Ensure we are not in the middle of a longer identifier
                before = json_text[i - 1] if i > 0 else ""
                after = json_text[end] if end < length else ""
                if before.isalnum() or before == "_" or after.isalnum() or after == "_":
                    continue

                # Require a colon after optional whitespace
                j = end
                while j < length and json_text[j].isspace():
                    j += 1
                if j >= length or json_text[j] != ":":
                    continue

                # Rewrite key as "key"
                result.append('"')
                result.append(key)
                result.append('"')
                i = end
                matched = True
                break

            if matched:
                continue

        result.append(ch)
        i += 1

    fixed = "".join(result)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def _chunk_iterable(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _trim_text(text: str, max_lines: int = 120, max_chars: int = 6000) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    trimmed = "\n".join(lines)
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars]
    return trimmed


def _infer_page_offset(
    pdf_path: Path,
    entries: List[Dict[str, Any]],
    api_key: str,
    timeout: int,
    fingerprints: Optional[List[Dict[str, Any]]] = None,
    max_samples: int = 3,
) -> Optional[int]:
    if not entries:
        return None

    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return None

    resolved = pdf_path.resolve()
    with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
        page_count = doc.page_count

    # Build candidate (canonical_index, pdf_index0) pairs for sampling
    cano_pdf_pairs: List[Tuple[int, int]] = []

    # 1) Prefer sampling at canonical 1/3, 1/2, 2/3 positions on dominant-dimension pages
    dominant_dims: Optional[tuple[int, int]] = None
    canonical_map: Dict[int, int] = {}
    if fingerprints:
        dominant_dims = dominant_dimensions(fingerprints)
        if dominant_dims:
            canonical_map = build_canonical_map_for_dims(fingerprints, dominant_dims)
            total_canonical = len(canonical_map)
            for frac in (1/3, 1/2, 2/3):
                if total_canonical == 0:
                    break
                ci = max(1, min(total_canonical, int(round(total_canonical * frac))))
                pdf_page = canonical_map.get(ci)
                if isinstance(pdf_page, int):
                    idx0 = max(0, pdf_page - 1)
                    pair = (ci, idx0)
                    if pair not in cano_pdf_pairs:
                        cano_pdf_pairs.append(pair)
                if len(cano_pdf_pairs) >= max_samples:
                    break

    # 2) Fallback to TOC-derived anchors if needed
    if len(cano_pdf_pairs) < max_samples:
        for entry in entries:
            target = _util_coerce_positive_int(entry.get("target_page"))
            if target is None:
                target = _util_coerce_positive_int(entry.get("page"))
            if target is None:
                continue
            index0 = max(0, target - 1)
            pair = (target, index0)  # assume canonical ~ printed when baseline not present
            if pair not in cano_pdf_pairs:
                cano_pdf_pairs.append(pair)
            if len(cano_pdf_pairs) >= max_samples:
                break

    # 3) And as last resort add mid / end pages
    if len(cano_pdf_pairs) < max_samples and page_count > 1:
        for idx in (page_count // 2, page_count - 1):
            if idx >= 0:
                pair = (idx + 1, idx)  # approximate canonical=index+1
                if pair not in cano_pdf_pairs:
                    cano_pdf_pairs.append(pair)
            if len(cano_pdf_pairs) >= max_samples:
                break

    offsets: List[int] = []
    for canonical_idx, index0 in cano_pdf_pairs:
        if index0 < 0 or index0 >= page_count:
            continue
        page_number = _get_printed_page_number(resolved, index0, api_key, timeout)
        if page_number is None:
            continue
        # offset aligns printed page to canonical index: canonical = printed + offset
        offsets.append(canonical_idx - page_number)
        if len(offsets) >= max_samples:
            break

    if not offsets:
        return None

    offsets.sort()
    median = offsets[len(offsets) // 2]
    return median


def _get_printed_page_number(
    pdf_path: Path,
    index: int,
    api_key: str,
    timeout: int,
) -> Optional[int]:
    cache_key = (str(pdf_path), index)
    with _CACHE_LOCK:
        if cache_key in _PAGE_NUMBER_CACHE:
            cached = _PAGE_NUMBER_CACHE[cache_key]
            return int(cached) if isinstance(cached, int) else cached

    image_b64 = _render_page_image_base64(pdf_path, index)
    if image_b64 is None:
        with _CACHE_LOCK:
            _PAGE_NUMBER_CACHE[cache_key] = None
        return None

    result = _query_page_number(api_key, image_b64, timeout)
    with _CACHE_LOCK:
        _PAGE_NUMBER_CACHE[cache_key] = result
    return result


def _render_page_image_base64(pdf_path: Path, index: int) -> Optional[str]:
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return None

    try:
        with fitz.open(pdf_path) as doc:  # type: ignore[attr-defined]
            if index < 0 or index >= doc.page_count:
                return None
            page = doc.load_page(index)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
            image_bytes = _pixmap_to_jpeg(pix)
    except Exception:
        return None

    return base64.b64encode(image_bytes).decode("ascii") if image_bytes else None


def _query_page_number(api_key: str, image_b64: str, timeout: int) -> Optional[int]:
    instructions = (
        "You are given an image of a book page. Identify the printed page number "
        "visible on the page. Respond with a JSON object {\"page_number\": <number or null>} "
        "without additional commentary. If you cannot determine the page number, use null."
    )

    user_content = [
        {
            "type": "text",
            "text": "Report the printed page number for this page. Use the JSON format described above.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        },
    ]

    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        body = _call_chat_completion(api_key, payload, request_timeout=timeout)
    except TOCExtractionError:
        return None

    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        return None

    json_text = _extract_json_block(content)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    value = data.get("page_number")
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    # Backward-compat shim: route to shared util
    return _util_coerce_positive_int(value)


def _pixmap_to_jpeg(pix: Any) -> bytes:
    try:
        return pix.tobytes("jpeg", quality=JPEG_QUALITY)
    except TypeError:
        return pix.tobytes("jpeg")


def _get_or_render_page_payload(
    pdf_path: Path, index: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    cache_key = (str(pdf_path), index)
    with _CACHE_LOCK:
        cached = _PAYLOAD_CACHE.get(cache_key)
        if cached is not None:
            return cached[0], cached[1]

    payload, fingerprint = _render_page_payload(pdf_path, index)
    if payload is not None:
        with _CACHE_LOCK:
            _PAYLOAD_CACHE[cache_key] = (payload, fingerprint)
    return payload, fingerprint


def _render_page_payload(pdf_path: Path, index: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        raise TOCExtractionError(
            "PyMuPDF (fitz) is required to extract content from PDFs."
        )

    with fitz.open(pdf_path) as doc:  # type: ignore[attr-defined]
        try:
            page = doc.load_page(index)
        except ValueError:
            return None, None

        entry: Dict[str, Any] = {"page": index + 1}

        text = page.get_text("text").strip()
        fingerprint = build_page_fingerprint(page, text)
        if text:
            entry["text"] = _trim_text(text)
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
            image_bytes = _pixmap_to_jpeg(pix)
            if not image_bytes:
                return None, fingerprint
            entry["image_b64"] = base64.b64encode(image_bytes).decode("ascii")
    return entry, fingerprint


def _purge_cache_for_path(pdf_path: Path) -> None:
    with _CACHE_LOCK:
        try:
            target = str(pdf_path.resolve())
        except FileNotFoundError:
            target = str(pdf_path.absolute())
        keys_to_delete = [key for key in _PAYLOAD_CACHE if key[0] == target]
        for key in keys_to_delete:
            del _PAYLOAD_CACHE[key]
        number_keys = [key for key in _PAGE_NUMBER_CACHE if key[0] == target]
        for key in number_keys:
            del _PAGE_NUMBER_CACHE[key]


def _write_temp_json(data: Any) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".json",
        prefix="siliconflow-toc-",
        mode="w",
        encoding="utf-8",
    )
    with tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
    return Path(tmp.name)


def _safe_json(response: Optional[requests.Response]) -> Any:
    if response is None:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        pass
