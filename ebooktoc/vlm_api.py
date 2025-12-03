"""VLM API client for extracting TOC data from PDFs.

This module exposes an OpenAI Chat Completions–style client that defaults to
SiliconFlow's Qwen3‑VL‑32B‑Instruct model but can be pointed at any
OpenAI‑compatible VLM backend via ``api_base`` and ``model`` parameters.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .fingerprints import (
    build_page_fingerprint,
    dominant_dimensions,
    build_canonical_map_for_dims,
)
from .utils import (
    coerce_positive_int as _util_coerce_positive_int,
    download_to_temp as _util_download_to_temp,
)

# Default OpenAI-compatible VLM configuration (SiliconFlow as the default backend)
API_BASE_DEFAULT = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_BATCH_SIZE = 3
JPEG_QUALITY = 80
DEFAULT_CACHE_SIZE = 500
DEFAULT_DPI_SCALE = 1.5
MAX_IMAGE_DIMENSION = 2048


class LRUCache(MutableMapping):
    """Simple size-bounded LRU cache.

    The cache stores items in insertion order and moves entries to the end
    when they are accessed. When ``maxsize`` is exceeded, the least recently
    used entry is evicted.
    """

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def set(self, key: Any, value: Any) -> None:
        """Set *key* to *value* (alias for ``__setitem__``)."""
        self.__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self):
        with self._lock:
            # Iterate over a snapshot to avoid holding the lock during user loops.
            return iter(list(self._data.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            if key in self._data:
                value = self._data[key]
                self._data.move_to_end(key)
                return value
            return default

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        with self._lock:
            items = list(self._data.items())
        return f"{self.__class__.__name__}({items})"


# LRU caches keyed by (pdf_path, index) for rendered payloads and page numbers.
_PAYLOAD_CACHE: LRUCache = LRUCache()
_PAGE_NUMBER_CACHE: LRUCache = LRUCache()

# Thread-local HTTP sessions with retry for VLM calls
_SESSION_LOCAL = threading.local()


class TOCExtractionError(RuntimeError):
    """Raised when the external TOC extraction service fails."""


def _is_retryable_error(exc: Exception) -> bool:
    """Return True when a wrapped exception is safe to retry.

    We inspect ``exc.__cause__`` (as produced by ``raise ... from``) and treat:

    - HTTP 429 / 5xx as retryable (rate limiting / transient server errors)
    - ``requests.Timeout`` and ``requests.ConnectionError`` as retryable
    - Other errors as non-retryable.
    """
    cause = getattr(exc, "__cause__", None)

    if isinstance(cause, requests.Timeout):
        return True
    if isinstance(cause, requests.ConnectionError):
        return True

    if isinstance(cause, requests.HTTPError) and cause.response is not None:
        status = cause.response.status_code
        if status in (429, 500, 502, 503, 504):
            return True
        return False

    return False


def fetch_document_json(
    pdf_path: Path | None,
    api_key: str,
    poll_interval: int = 5,  # kept for CLI compatibility; unused
    timeout: int = 600,  # kept for CLI compatibility; unused
    page_limit: int = 15,
    remote_url: str | None = None,
    task_id: str | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    start_page: int = 1,
    api_base: str | None = None,
    model: str | None = None,
    max_workers: int = 3,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Extract TOC data using an OpenAI-format VLM API (default: SiliconFlow Qwen).

    Parameters
    ----------
    pdf_path :
        Local PDF path when ``remote_url`` is not provided.
    api_key :
        VLM API token (e.g., SiliconFlow or OpenRouter).
    poll_interval, timeout :
        Retained for CLI compatibility; not used by the streaming workflow.
    page_limit :
        Maximum number of pages to scan from the requested window. A value of
        ``0`` means \"from ``start_page`` to the end of the document\".
    remote_url :
        Optional remote PDF URL to download before scanning. When supplied,
        ``pdf_path`` is ignored.
    task_id :
        Not supported by this workflow; passing a value raises
        :class:`TOCExtractionError`.
    batch_size :
        Number of page payloads to include in each chat completion request.
    start_page :
        1-based starting page for the scan window. Callers can advance this
        between invocations to implement incremental or windowed scanning.

    Returns
    -------
    Path
        Path to a temporary JSON file containing the model output. The JSON
        object includes at least ``\"toc\"``, ``\"page_offset\"``,
        ``\"fingerprints\"``, and ``\"page_map\"`` keys.
    """

    if task_id is not None:
        raise TOCExtractionError("VLM workflow does not support --task-id")

    source_path, cleanup_required = _resolve_source_pdf(pdf_path, remote_url)
    try:
        page_payloads, fingerprints = _collect_page_payloads(
            source_path, page_limit, start_page=start_page
        )
        aggregated: list[dict[str, Any]] = []
        effective_model = model or MODEL_NAME

        effective_batch = max(1, batch_size)
        batches: list[list[dict[str, Any]]] = list(
            _chunk_iterable(page_payloads, effective_batch)
        )

        if batches:
            max_attempts = 3

            def _fetch_toc_for_batch(
                batch_index: int,
                batch: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                attempts = 0
                last_error: Exception | None = None
                while attempts < max_attempts:
                    attempts += 1
                    payload = _build_payload(
                        batch, page_limit, start_page=start_page, model=effective_model
                    )
                    try:
                        response_body = _call_chat_completion(
                            api_key, payload, request_timeout=timeout, api_base=api_base
                        )
                        return _parse_response_payload(response_body)
                    except TOCExtractionError as exc:
                        last_error = exc
                        if attempts >= max_attempts or not _is_retryable_error(exc):
                            raise
                        if progress_callback is not None:
                            status = (
                                f"Batch {batch_index + 1}/{len(batches)} retry "
                                f"{attempts}/{max_attempts}..."
                            )
                            try:
                                progress_callback(batch_index, len(batches), status)
                            except Exception:
                                # Progress updates are best-effort only.
                                pass
                        # Simple exponential backoff capped at 5 seconds
                        sleep_seconds = min(2 ** (attempts - 1), 5)
                        time.sleep(sleep_seconds)

                # Should not be reached because we either return or raise above
                if last_error is not None:
                    raise last_error
                raise RuntimeError(
                    "Unexpected code path reached in _fetch_toc_for_batch"
                )

            worker_count = max(1, max_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_index: dict[concurrent.futures.Future[list[dict[str, Any]]], int] = {}
                for idx, batch in enumerate(batches):
                    future = executor.submit(_fetch_toc_for_batch, idx, batch)
                    future_to_index[future] = idx

                completed_batches = 0
                for future in concurrent.futures.as_completed(future_to_index):
                    toc_json = future.result()
                    aggregated.extend(toc_json)
                    completed_batches += 1
                    if progress_callback is not None:
                        status = f"Entries: {len(aggregated)}"
                        try:
                            progress_callback(
                                completed_batches,
                                len(batches),
                                status,
                            )
                        except Exception:
                            # Progress updates are best-effort only.
                            pass

        offset = None
        try:
            offset = _infer_page_offset(
                source_path,
                aggregated,
                api_key,
                timeout,
                fingerprints,
                api_base=api_base,
                model=effective_model,
                progress_callback=progress_callback,
            )
        except TOCExtractionError:
            offset = None

        # Build a canonical page map for this source PDF (based on dominant dims)
        page_map: dict[int, int] = {}
        dims = dominant_dimensions(fingerprints) if fingerprints else None
        if dims:
            raw_map = build_canonical_map_for_dims(fingerprints, dims)
            # When scanning from a non-first page, fingerprints only cover a
            # window of the document. Adjust the pdf_page values so they are
            # expressed in absolute 1-based page numbers rather than
            # window-relative indices.
            if start_page > 1 and raw_map:
                window_offset = start_page - 1
                page_map = {
                    canon: page + window_offset for canon, page in raw_map.items()
                }
            else:
                page_map = raw_map

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
    pdf_path: Path | None, remote_url: str | None
) -> tuple[Path, bool]:
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect payloads and fingerprints for a window of PDF pages.

    The window is defined by a 1-based ``start_page`` and a maximum number of
    pages ``max_pages``. If ``max_pages`` is less than or equal to zero, all
    pages from ``start_page`` to the end of the document are considered. When
    the requested window lies entirely past the end of the document, this
    function returns empty lists instead of raising an error so callers can
    treat it as "no pages left to scan".

    Raises
    ------
    TOCExtractionError
        If the PDF cannot be opened, contains no pages, or there are pages in
        the requested window but none can be extracted (no text or renderable
        images).
    """
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

    payloads: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []

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
    page_payloads: list[dict[str, Any]],
    max_pages: int,
    start_page: int = 1,
    model: str | None = None,
) -> dict[str, Any]:
    """Build the chat-completion payload for a window of pages.

    Parameters
    ----------
    page_payloads :
        List of per-page payload dictionaries produced by
        :func:`_collect_page_payloads` (each containing either ``\"text\"`` or
        ``\"image_b64\"``, plus a 1-based ``\"page\"`` number).
    max_pages :
        Maximum number of pages described by this prompt. When ``max_pages``
        is less than or equal to zero, the prompt is phrased as covering all
        pages from ``start_page`` onward.
    start_page :
        1-based starting page of the logical window being described to the
        model. This is used only for prompt wording (for example, to say
        \"pages 11 to 20\") and should match the window used when collecting
        ``page_payloads``.
    model :
        Optional model name override for API abstraction.

    Returns
    -------
    dict[str, Any]
        The payload dictionary to send to the VLM chat completions API,
        including system instructions and structured user content.
    """
    instructions = (
        "You are an assistant that extracts a book's table of contents from PDF content. "
        "For every TOC line, output an object with keys: "
        "'page' (integer, the source page number indicated by the excerpt header such as [Page 4]), "
        "'target_page' (integer or null, the destination page number explicitly printed at the END of the TOC line), and "
        "'content' (string, the title text without trailing page numbers or dot leaders). "
        "\n\n"
        "CRITICAL RULES FOR target_page:\n"
        "1. ONLY extract page numbers that are EXPLICITLY PRINTED at the END of the TOC line.\n"
        "2. Page numbers typically appear after dot leaders (like '............   15') or at the right margin.\n"
        "3. If a TOC line has NO page number printed at the end, you MUST set 'target_page' to null.\n"
        "4. Do NOT guess, infer, or fabricate page numbers. Only extract what is actually printed.\n"
        "5. Do NOT confuse section/chapter numbers with page numbers:\n"
        "   - Section numbers (1.1, 1.2.3, 第1章) appear at the BEGINNING of the line.\n"
        "   - Page numbers appear at the END of the line, usually right-aligned.\n"
        "\n"
        "EXAMPLES:\n"
        "- Line: '第二版丛书序' (no number at end) → target_page: null\n"
        "- Line: '前言' (no number at end) → target_page: null\n"
        "- Line: '第 1 章  电磁现象的基本规律 ...... ...  1' → target_page: 1\n"
        "- Line: '1.1  场论和张量分析 ............ .... 1' → target_page: 1 (NOT the '1.1' at the start!)\n"
        "- Line: '1.1.2  张量的定义 ..................  4' → target_page: 4\n"
        "\n"
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

    user_content: list[dict[str, Any]] = [{"type": "text", "text": intro_text}]

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
                "REMEMBER: If a TOC entry has no page number at the end, target_page must be null. "
                "Do NOT fabricate page numbers.\n"
                "Example output: {\"toc\": ["
                "{\"page\": 4, \"target_page\": null, \"content\": \"前言\"}, "
                "{\"page\": 4, \"target_page\": 1, \"content\": \"第1章 导论\"}"
                "]}. "
                "Return ONLY the JSON object described above. Do not add extra commentary or formatting."
            ),
        }
    )

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content},
    ]

    return {
        "model": model or MODEL_NAME,
        "temperature": 0.2,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }


def _call_chat_completion(
    api_key: str,
    payload: dict[str, Any],
    request_timeout: int,
    api_base: str | None = None,
) -> dict[str, Any]:
    def _get_session() -> Session:
        """Return a thread-local Session configured with retry.

        Each worker thread gets its own :class:`requests.Session` instance so
        that parallel VLM requests do not share mutable session state while
        still benefiting from connection pooling.
        """

        session = getattr(_SESSION_LOCAL, "session", None)
        if session is not None:
            return session

        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _SESSION_LOCAL.session = session
        return session

    base = (api_base or API_BASE_DEFAULT).rstrip("/")
    endpoint = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = (10, max(1, int(request_timeout)))
    session = _get_session()
    try:
        response = session.post(endpoint, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = _safe_json(exc.response) if exc.response is not None else str(exc)
        raise TOCExtractionError(f"VLM API error: {detail}") from exc
    except requests.RequestException as exc:
        raise TOCExtractionError(f"Failed to call VLM API: {exc}") from exc

    body = response.json()
    if not isinstance(body, dict):
        raise TOCExtractionError("Unexpected VLM API response format")
    return body


def _parse_response_payload(body: dict[str, Any]) -> list[dict[str, Any]]:
    choices = body.get("choices")
    if not choices:
        raise TOCExtractionError(f"VLM API response missing choices: {body}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise TOCExtractionError(f"VLM API response missing text content: {message}")

    json_text = _extract_json_block(content)
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as exc:
        # Try a best-effort repair pass before giving up.
        repaired = _repair_toc_json(json_text)
        if repaired is None:
            raise TOCExtractionError(
                f"Failed to parse VLM JSON output: {exc}\nRaw: {json_text}"
            ) from exc
        data = repaired

    toc: Any
    if isinstance(data, dict):
        if "toc" in data:
            toc = data["toc"]
        elif "output" in data and isinstance(data["output"], list):
            toc = data["output"]
        # If the model returns a single entry object (with page/content), accept it
        # instead of wrapping it in a {\"toc\": [...]} container.
        # When we see a mapping that looks like one TOC entry, treat it as a
        # singleton list rather than failing hard.
        elif "page" in data and "content" in data:
            toc = [data]
        else:
            raise TOCExtractionError(
                f"VLM API response missing 'toc' key: {json_text}"
            )
    elif isinstance(data, list):
        toc = data
    else:
        raise TOCExtractionError(
            f"VLM API response must be a JSON array or object with 'toc'; received {type(data).__name__}."
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


def _find_json_substring(text: str) -> str | None:
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
    except (json.JSONDecodeError, ValueError):
        pass

    # General path: search for the first plausible JSON opener and let
    # raw_decode determine where the JSON payload ends.
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
        except (json.JSONDecodeError, ValueError):
            continue
        return text[idx : idx + end]

    return None


def _repair_toc_json(json_text: str) -> Any | None:
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
    result: list[str] = []
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
    except (json.JSONDecodeError, ValueError):
        return None


def _chunk_iterable(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
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
    entries: list[dict[str, Any]],
    api_key: str,
    timeout: int,
    fingerprints: list[dict[str, Any]] | None = None,
    max_samples: int = 5,
    api_base: str | None = None,
    model: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> int | None:
    """Infer the offset between printed page numbers and PDF page indices.

    The offset satisfies: ``pdf_page_1based = printed_page + offset``. For
    example, if printed page 1 is on PDF page 11, then ``offset == 10``.

    This implementation uses a heuristic search plus verification strategy:

    1. Select reliable anchor entries from the TOC (prefer larger
       ``target_page`` values).
    2. For each anchor, search a candidate offset range ``[0, 50]``.
    3. Verify the best-found offset using other TOC entries.
    4. Fall back to a sampling-based method if the heuristic search fails.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    entries :
        List of TOC entry dictionaries with ``\"target_page\"`` and/or
        ``\"page\"`` fields.
    api_key :
        VLM API key for page-number recognition.
    timeout :
        Request timeout in seconds.
    fingerprints :
        Optional page fingerprints for compatibility with callers. Currently
        unused by the heuristic search but may be leveraged by future
        refinements.
    max_samples :
        Maximum number of samples to use when falling back to the sampling
        method.
    api_base :
        Optional API base URL override.
    model :
        Optional model name override.

    Returns
    -------
    int | None
        The inferred offset, or ``None`` if inference fails.
    """
    if not entries:
        return None

    # When multiple page-numbering segments are present (for example, preface
    # and body both starting from printed page 1), try to infer a per-segment
    # offset first and return the dominant body's offset for backward
    # compatibility.
    try:
        from .toc_parser import (
            detect_page_segments,
            classify_entries,
            select_anchor_entries,
            locate_entry_by_text_search,
        )
    except Exception:  # pragma: no cover - defensive import
        detect_page_segments = None  # type: ignore[assignment]
        classify_entries = None  # type: ignore[assignment]
        select_anchor_entries = None  # type: ignore[assignment]
        locate_entry_by_text_search = None  # type: ignore[assignment]

    if detect_page_segments is not None:
        try:
            segments = detect_page_segments(entries)
        except Exception:
            segments = []
        else:
            if len(segments) > 1:
                _infer_offsets_for_segments(
                    pdf_path,
                    segments,
                    api_key,
                    timeout,
                    api_base=api_base,
                    model=model,
                )
                for seg in segments:
                    if getattr(seg, "segment_type", None) == "body" and getattr(
                        seg, "offset", None
                    ) is not None:
                        return int(seg.offset)  # type: ignore[return-value]
                for seg in segments:
                    if getattr(seg, "offset", None) is not None:
                        return int(seg.offset)  # type: ignore[return-value]

    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return None

    resolved = pdf_path.resolve()
    with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
        page_count = doc.page_count

    if page_count == 0:
        return None

    # Prefer multi-anchor positioning when classification helpers are
    # available; fall back to legacy heuristics otherwise.
    if classify_entries is not None and select_anchor_entries is not None and locate_entry_by_text_search is not None:
        try:
            _unnumbered, _suspicious, normal = classify_entries(entries)
        except Exception:
            normal = list(range(len(entries)))

        anchors = select_anchor_entries(entries, normal, max_anchors=max_samples)
        if anchors:
            total_anchors = len(anchors)
            for idx, anchor in enumerate(anchors):
                # Try zero-cost text search first.
                pdf_page = locate_entry_by_text_search(
                    resolved,
                    anchor.content,
                    search_start=1,
                    search_end=min(page_count, anchor.printed_page + 100),
                )
                if pdf_page is not None:
                    anchor.pdf_page = pdf_page
                    anchor.located_by = "text"
                    anchor.offset = pdf_page - anchor.printed_page
                    continue

                # Fallback to VLM-based printed-page probing when available.
                pdf_page = locate_anchor_by_vlm(
                    resolved,
                    anchor.content,
                    anchor.printed_page,
                    api_key,
                    timeout,
                    page_count,
                    api_base=api_base,
                    model=model,
                )
                if pdf_page is not None:
                    anchor.pdf_page = pdf_page
                    anchor.located_by = "vlm"
                    anchor.offset = pdf_page - anchor.printed_page

                if progress_callback is not None and total_anchors:
                    try:
                        progress_callback(
                            idx + 1,
                            total_anchors,
                            f"Inferring printed-page offset (VLM)... anchors {idx + 1}/{total_anchors}",
                        )
                    except Exception:
                        pass

            valid_anchors = [(a, a.offset) for a in anchors if a.offset is not None]
            if valid_anchors:
                if len(valid_anchors) == 1:
                    # Only one anchor succeeded; accept its offset but keep
                    # legacy fallbacks available if it later proves wrong.
                    return int(valid_anchors[0][1])  # type: ignore[arg-type]

                offsets = [int(o) for _, o in valid_anchors]  # type: ignore[arg-type]
                offset_range = max(offsets) - min(offsets)

                offsets.sort()
                if offset_range <= 5:
                    # Very consistent offsets – pick the median.
                    return offsets[len(offsets) // 2]
                if offset_range <= 20:
                    # Some variation but still plausible – median is robust.
                    return offsets[len(offsets) // 2]

                # Large disagreement between anchors; prefer the most common
                # offset rather than trusting a single outlier.
                counts = Counter(offsets)
                most_common, _ = counts.most_common(1)[0]
                return most_common

    # Legacy heuristic path: single-anchor style probing with verification and
    # sampling fallback.
    offset = _heuristic_offset_search(
        resolved,
        entries,
        api_key,
        timeout,
        page_count,
        api_base=api_base,
        model=model,
    )

    if offset is not None:
        if _verify_offset(
            resolved,
            entries,
            offset,
            api_key,
            timeout,
            page_count,
            max_verify=3,
            api_base=api_base,
            model=model,
        ):
            return offset

    return _fallback_offset_sampling(
        resolved,
        entries,
        api_key,
        timeout,
        page_count,
        fingerprints,
        max_samples,
        api_base=api_base,
        model=model,
    )


def _infer_offsets_for_segments(
    pdf_path: Path,
    segments: list[Any],
    api_key: str,
    timeout: int,
    api_base: str | None = None,
    model: str | None = None,
) -> None:
    """Infer offsets for each PageSegment in-place using heuristics.

    The strategy mirrors :func:`_infer_page_offset` but runs independently
    for each segment so that books with multiple page-numbering systems
    (for example, preface and body) receive distinct offsets.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    segments :
        List of :class:`ebooktoc.toc_parser.PageSegment` instances.
    api_key :
        VLM API key.
    timeout :
        Request timeout in seconds.
    api_base :
        Optional API base URL override.
    model :
        Optional model name override.
    """
    if not segments:
        return

    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return

    resolved = pdf_path.resolve()
    with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
        page_count = doc.page_count

    if page_count == 0:
        return

    for segment in segments:
        segment_entries = getattr(segment, "entries", None) or []
        if not segment_entries:
            continue

        # Use a conservative offset search range for all segments to keep API
        # calls bounded. Most books have front matter + body offsets well
        # below 80 pages.
        offset = _heuristic_offset_search(
            resolved,
            segment_entries,
            api_key,
            timeout,
            page_count,
            api_base=api_base,
            model=model,
            offset_range=(0, 80),
            step=5,
        )
        if offset is None:
            continue

        if _verify_offset(
            resolved,
            segment_entries,
            offset,
            api_key,
            timeout,
            page_count,
            max_verify=3,
            api_base=api_base,
            model=model,
        ):
            try:
                # PageSegment has an ``offset`` attribute; fall back silently
                # when it does not.
                segment.offset = int(offset)  # type: ignore[attr-defined]
            except Exception:
                continue


def _heuristic_offset_search(
    pdf_path: Path,
    entries: list[dict[str, Any]],
    api_key: str,
    timeout: int,
    page_count: int,
    api_base: str | None = None,
    model: str | None = None,
    offset_range: tuple[int, int] = (0, 50),
    step: int = 5,
) -> int | None:
    """Search for a plausible offset using TOC anchors.

    This function selects TOC entries with relatively large printed page
    numbers (for example, ``target_page > 30``) under the assumption that
    later chapters are more likely to have visible printed page numbers. For
    each such anchor, it searches a candidate offset range and compares the
    printed page number detected by the VLM with the TOC value.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    entries :
        List of TOC entry dictionaries.
    api_key :
        VLM API key.
    timeout :
        Request timeout in seconds.
    page_count :
        Total number of pages in the PDF.
    api_base :
        Optional API base URL override.
    model :
        Optional model name override.
    offset_range :
        Inclusive ``(min_offset, max_offset)`` range to search.
    step :
        Step size for the coarse search before fine-tuning.

    Returns
    -------
    int | None
        The first offset that yields a consistent match for an anchor, or
        ``None`` if no candidate is found.
    """
    # Collect reliable anchors: entries with target_page > 30.
    anchors: list[tuple[int, str]] = []
    for entry in entries:
        target = _coerce_positive_int(entry.get("target_page"))
        if target is None:
            target = _coerce_positive_int(entry.get("page"))
        if target is not None and target > 30:
            title = (entry.get("content") or "")[:50]
            anchors.append((target, title))

    # Sort by target_page descending (larger pages are more reliable).
    anchors.sort(key=lambda x: x[0], reverse=True)

    # Also add some medium-range anchors for diversity.
    medium_anchors = [(t, title) for t, title in anchors if 50 <= t <= 150]
    anchors = anchors[:3] + medium_anchors[:2]

    if not anchors:
        # No reliable anchors found; fall back to smaller pages (>10).
        for entry in entries:
            target = _coerce_positive_int(entry.get("target_page"))
            if target is not None and target > 10:
                anchors.append((target, (entry.get("content") or "")[:50]))
                if len(anchors) >= 3:
                    break

    if not anchors:
        return None

    min_off, max_off = offset_range

    for target_printed, _title in anchors[:3]:
        # Coarse search with a configurable step.
        for candidate_offset in range(min_off, max_off + 1, step):
            pdf_page_1based = target_printed + candidate_offset
            pdf_idx = pdf_page_1based - 1

            if pdf_idx < 0 or pdf_idx >= page_count:
                continue

            printed = _get_printed_page_number(
                pdf_path,
                pdf_idx,
                api_key,
                timeout,
                api_base=api_base,
                model=model,
            )

            if printed is None:
                continue

            if printed == target_printed:
                # Found an exact match.
                return candidate_offset

            # If close, perform a fine-grained search around this offset.
            if abs(printed - target_printed) <= step:
                fine_offset = _fine_search_offset(
                    pdf_path,
                    target_printed,
                    candidate_offset,
                    api_key,
                    timeout,
                    page_count,
                    api_base=api_base,
                    model=model,
                    search_radius=step,
                )
                if fine_offset is not None:
                    return fine_offset

    return None


def _fine_search_offset(
    pdf_path: Path,
    target_printed: int,
    center_offset: int,
    api_key: str,
    timeout: int,
    page_count: int,
    api_base: str | None = None,
    model: str | None = None,
    search_radius: int = 5,
) -> int | None:
    """Search offsets in a small window around ``center_offset``.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    target_printed :
        Printed page number from the TOC for the anchor entry.
    center_offset :
        Offset around which to search.
    api_key :
        VLM API key.
    timeout :
        Request timeout in seconds.
    page_count :
        Total number of pages in the PDF.
    api_base :
        Optional API base URL override.
    model :
        Optional model name override.
    search_radius :
        Symmetric radius (in offset units) around ``center_offset``.

    Returns
    -------
    int | None
        The first offset in the window that yields an exact match, or
        ``None`` if no candidate matches.
    """
    for delta in range(-search_radius, search_radius + 1):
        if delta == 0:
            # The centre offset will already have been checked by the caller.
            continue

        candidate_offset = center_offset + delta
        pdf_idx = target_printed + candidate_offset - 1

        if pdf_idx < 0 or pdf_idx >= page_count:
            continue

        printed = _get_printed_page_number(
            pdf_path,
            pdf_idx,
            api_key,
            timeout,
            api_base=api_base,
            model=model,
        )

        if printed == target_printed:
            return candidate_offset

    return None


def _verify_offset(
    pdf_path: Path,
    entries: list[dict[str, Any]],
    offset: int,
    api_key: str,
    timeout: int,
    page_count: int,
    max_verify: int = 3,
    api_base: str | None = None,
    model: str | None = None,
    tolerance: int = 1,
) -> bool:
    """Verify a candidate offset by sampling TOC entries.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    entries :
        TOC entries to use for verification.
    offset :
        Candidate offset to verify.
    api_key :
        VLM API key.
    timeout :
        Request timeout in seconds.
    page_count :
        Total number of pages in the PDF.
    max_verify :
        Maximum number of entries to verify.
    api_base :
        Optional API base URL override.
    model :
        Optional model name override.
    tolerance :
        Allowed absolute difference between detected and expected printed
        page numbers for a verification to be considered a success.

    Returns
    -------
    bool
        ``True`` if the majority of sampled entries agree with the offset
        within the specified tolerance, otherwise ``False``.
    """
    verify_targets: list[int] = []
    for entry in entries:
        target = _coerce_positive_int(entry.get("target_page"))
        if target is None:
            target = _coerce_positive_int(entry.get("page"))
        if target is not None and 20 < target < 200:
            verify_targets.append(target)

    # Deduplicate to avoid querying the same page multiple times.
    verify_targets = list(set(verify_targets))
    if len(verify_targets) > max_verify and max_verify > 0:
        # Sample roughly evenly spaced entries from the list.
        step = max(1, len(verify_targets) // max_verify)
        verify_targets = verify_targets[::step][:max_verify]

    if not verify_targets:
        # No verification candidates; treat the offset as acceptable.
        return True

    success_count = 0
    total_checked = 0

    for target_printed in verify_targets:
        pdf_idx = target_printed + offset - 1
        if pdf_idx < 0 or pdf_idx >= page_count:
            continue

        printed = _get_printed_page_number(
            pdf_path,
            pdf_idx,
            api_key,
            timeout,
            api_base=api_base,
            model=model,
        )

        if printed is None:
            continue

        total_checked += 1
        if abs(printed - target_printed) <= tolerance:
            success_count += 1

    if total_checked == 0:
        # If we could not read any printed numbers, assume success so the
        # caller can decide how to proceed.
        return True

    # Require a simple majority of successful checks.
    return success_count >= (total_checked + 1) // 2


def _fallback_offset_sampling(
    pdf_path: Path,
    entries: list[dict[str, Any]],
    api_key: str,
    timeout: int,
    page_count: int,
    fingerprints: list[dict[str, Any]] | None,
    max_samples: int,
    api_base: str | None = None,
    model: str | None = None,
) -> int | None:
    """Fallback offset estimation using sampled printed page numbers.

    This method samples a small number of pages from the latter half of the
    document (where printed page numbers are more likely to be stable) and
    derives an offset from the relationship between printed and PDF indices.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    entries :
        TOC entries (currently unused but accepted for future extensions).
    api_key :
        VLM API key.
    timeout :
        Request timeout in seconds.
    page_count :
        Total number of pages in the PDF.
    fingerprints :
        Optional page fingerprints (currently unused).
    max_samples :
        Maximum number of samples to use.
    api_base :
        Optional API base URL override.
    model :
        Optional model name override.

    Returns
    -------
    int | None
        Median offset computed from the sampled pages, or ``None`` if no
        reliable samples are available.
    """
    if page_count <= 0 or max_samples <= 0:
        return None

    sample_indices: list[int] = []

    # Prefer pages at regular intervals in the latter portion of the PDF.
    for frac in (0.4, 0.5, 0.6, 0.7, 0.8):
        idx = int(page_count * frac)
        if idx >= page_count:
            idx = page_count - 1
        if idx < 0:
            continue
        if idx not in sample_indices:
            sample_indices.append(idx)
        if len(sample_indices) >= max_samples:
            break

    if not sample_indices:
        return None

    offsets: list[int] = []

    for pdf_idx in sample_indices:
        if pdf_idx < 0 or pdf_idx >= page_count:
            continue

        printed = _get_printed_page_number(
            pdf_path,
            pdf_idx,
            api_key,
            timeout,
            api_base=api_base,
            model=model,
        )

        if printed is None or printed <= 0:
            continue

        # offset = pdf_page_1based - printed_page
        pdf_page_1based = pdf_idx + 1
        offset = pdf_page_1based - printed
        offsets.append(offset)

    if not offsets:
        return None

    offsets.sort()
    return offsets[len(offsets) // 2]


def _get_printed_page_numbers_concurrent(
    pdf_path: Path,
    page_indices: list[int],
    api_key: str,
    timeout: int,
    api_base: str | None = None,
    model: str | None = None,
    max_workers: int = 4,
) -> dict[int, int | None]:
    """Return printed page numbers for multiple PDF pages concurrently.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    page_indices :
        0-based PDF page indices to query.
    api_key :
        VLM API key.
    timeout :
        Request timeout per call in seconds.
    api_base :
        Optional API base URL override.
    model :
        Optional model name override.
    max_workers :
        Maximum number of concurrent worker threads.

    Returns
    -------
    dict[int, int | None]
        Mapping from page index to detected printed page number, or ``None``
        when the model could not determine a number.
    """
    results: dict[int, int | None] = {}
    if not page_indices:
        return results

    def _query(idx: int) -> tuple[int, int | None]:
        printed = _get_printed_page_number(
            pdf_path,
            idx,
            api_key,
            timeout,
            api_base=api_base,
            model=model,
        )
        return idx, printed

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_query, idx) for idx in page_indices]
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, printed = future.result()
            except Exception:
                continue
            results[idx] = printed

    return results


def locate_anchor_by_vlm(
    pdf_path: Path,
    anchor_content: str,
    anchor_printed_page: int,
    api_key: str,
    timeout: int,
    page_count: int,
    api_base: str | None = None,
    model: str | None = None,
) -> int | None:
    """Locate an anchor entry's PDF page using VLM.

    Strategy
    --------
    * For small printed pages (1–10), search the early portion of the PDF
      with a fine step so that we do not skip the true location.
    * For larger printed pages, search around the estimated region based
      on typical offsets.
    * Perform a coarse scan first, then refine around the closest match.
    """
    if page_count <= 0:
        return None

    # Phase 1: coarse sampling at a handful of strategic positions to
    # estimate the offset between printed and PDF pages.
    sample_indices = _get_sample_positions(page_count, anchor_printed_page)
    if not sample_indices:
        return None

    sample_results = _get_printed_page_numbers_concurrent(
        pdf_path,
        sample_indices,
        api_key,
        timeout,
        api_base=api_base,
        model=model,
        max_workers=4,
    )

    # Exact matches in the coarse samples can be used immediately.
    for idx0, printed in sample_results.items():
        if printed == anchor_printed_page:
            return idx0 + 1

    estimated_offset = _estimate_offset_from_samples(
        sample_results,
        anchor_printed_page,
    )
    if estimated_offset is None:
        return None

    # Phase 2: fine search around the estimated location in a narrow window.
    estimated_pdf_page = anchor_printed_page + estimated_offset
    fine_indices: list[int] = []
    for delta in range(-5, 6):
        candidate = estimated_pdf_page + delta - 1  # 0-based index
        if candidate < 0 or candidate >= page_count:
            continue
        if candidate in sample_results:
            continue
        fine_indices.append(candidate)

    if not fine_indices:
        return None

    fine_results = _get_printed_page_numbers_concurrent(
        pdf_path,
        fine_indices,
        api_key,
        timeout,
        api_base=api_base,
        model=model,
        max_workers=4,
    )
    for idx0, printed in fine_results.items():
        if printed == anchor_printed_page:
            return idx0 + 1

    return None


def _get_sample_positions(page_count: int, anchor_printed_page: int) -> list[int]:
    """Return 0-based page indices to probe for coarse offset sampling."""
    positions: list[int] = []
    if page_count <= 0:
        return positions

    if anchor_printed_page <= 10:
        # When printed page numbers are very small, scatter samples in the
        # front of the document.
        candidates = [5, 10, 15, 20, 30, 40, 50]
    else:
        base = anchor_printed_page
        candidates = [base + 5, base + 15, base + 25, base + 40]

    for pos in candidates:
        idx = pos - 1
        if 0 <= idx < page_count and idx not in positions:
            positions.append(idx)

    return positions[:5]


def _estimate_offset_from_samples(
    sample_results: dict[int, int | None],
    target_printed: int,
) -> int | None:
    """Estimate a PDF offset from coarse printed-page samples."""
    valid: list[tuple[int, int]] = [
        (idx0, printed)
        for idx0, printed in sample_results.items()
        if printed is not None and printed > 0
    ]
    if not valid:
        return None

    # Pick the sampled page whose printed page is closest to the target.
    closest_idx, closest_printed = min(
        valid,
        key=lambda item: abs(item[1] - target_printed),
    )
    # offset = pdf_page_1based - printed_page
    estimated_offset = (closest_idx + 1) - closest_printed
    return estimated_offset


def _fine_search_for_printed_page(
    pdf_path: Path,
    target_printed: int,
    reference_pdf_page: int,
    reference_printed: int,
    api_key: str,
    timeout: int,
    page_count: int,
    api_base: str | None = None,
    model: str | None = None,
) -> int | None:
    """Search nearby pages for an exact printed page number match."""
    estimated_offset = reference_pdf_page - reference_printed
    estimated_pdf_page = target_printed + estimated_offset

    start = max(1, estimated_pdf_page - 5)
    end = min(page_count, estimated_pdf_page + 5)

    for test_page in range(start, end + 1):
        if test_page == reference_pdf_page:
            continue
        printed = _get_printed_page_number(
            pdf_path,
            test_page - 1,
            api_key,
            timeout,
            api_base=api_base,
            model=model,
        )
        if printed == target_printed:
            return test_page

    return None


def locate_heading_by_vlm(
    pdf_path: Path,
    heading_text: str,
    search_start: int,
    search_end: int,
    api_key: str,
    timeout: int,
    api_base: str | None = None,
    model: str | None = None,
) -> int | None:
    """Locate a heading/title within a PDF page range using the VLM.

    This helper is used for entries that do not have reliable printed
    page numbers (for example, unnumbered preface items). It scans a
    small range of pages and asks the VLM whether the heading appears on
    each page, stopping at the first positive hit.
    """
    for page_num in range(max(1, search_start), max(search_start, search_end) + 1):
        if _check_page_for_heading(
            pdf_path,
            page_num,
            heading_text,
            api_key,
            timeout,
            api_base=api_base,
            model=model,
        ):
            return page_num
    return None


def _check_page_for_heading(
    pdf_path: Path,
    page_num: int,
    heading_text: str,
    api_key: str,
    timeout: int,
    api_base: str | None = None,
    model: str | None = None,
) -> bool:
    """Return ``True`` if VLM judges that *heading_text* appears on *page_num*."""
    image_b64 = _render_page_image_base64(pdf_path, page_num - 1)
    if image_b64 is None:
        return False

    instructions = (
        "You are given a page from a book. Determine whether the page contains "
        f"the heading or title: \"{heading_text}\".\n"
        "Headings may appear at the top of the page, as chapter titles, or as "
        "section headings. Respond with a JSON object {\"found\": true} if the "
        "heading is present, or {\"found\": false} otherwise. Only respond true "
        "for exact or very close matches (minor punctuation or spacing "
        "differences are acceptable)."
    )

    user_content = [
        {
            "type": "text",
            "text": f"Does this page contain the heading \"{heading_text}\"?",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        },
    ]

    payload = {
        "model": model or MODEL_NAME,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        body = _call_chat_completion(
            api_key,
            payload,
            request_timeout=timeout,
            api_base=api_base,
        )
    except TOCExtractionError:
        return False

    choices = body.get("choices") or []
    if not choices:
        return False

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        return False

    json_text = _extract_json_block(content)
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return False

    found = data.get("found")
    try:
        return bool(found)
    except Exception:
        return False


def resolve_unnumbered_entries(
    pdf_path: Path,
    entries: list[dict[str, Any]],
    unnumbered_indices: list[int],
    suspicious_indices: list[int],
    first_body_pdf_page: int | None,
    api_key: str | None,
    timeout: int,
    api_base: str | None = None,
    model: str | None = None,
) -> dict[int, int | None]:
    """Resolve PDF pages for unnumbered and suspicious TOC entries.

    The resolver prefers free text search and only falls back to VLM
    when necessary. It is intended for front-matter items such as
    prefaces and forewords that lack printed page numbers.

    Parameters
    ----------
    pdf_path :
        Path to the PDF file.
    entries :
        Full list of TOC entries.
    unnumbered_indices :
        Indices of entries whose ``target_page`` is ``None``.
    suspicious_indices :
        Indices of entries whose ``target_page`` looks fabricated.
    first_body_pdf_page :
        Approximate PDF page where the main body starts. When provided,
        search for unnumbered entries will be restricted to pages before
        this page; otherwise a conservative front range is used.
    api_key :
        Optional VLM API key. When omitted, only text search is used.
    timeout :
        Request timeout in seconds for VLM calls.
    api_base :
        Optional API base URL override.
    model :
        Optional model name override.

    Returns
    -------
    dict[int, int | None]
        Mapping from entry index to resolved 1-based PDF page number.
        Values may be ``None`` when resolution fails.
    """
    indices: list[int] = list(dict.fromkeys(unnumbered_indices + suspicious_indices))
    if not indices:
        return {}

    try:
        from .toc_parser import locate_entry_by_text_search
    except Exception:  # pragma: no cover - defensive import
        locate_entry_by_text_search = None  # type: ignore[assignment]

    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return {}

    resolved = pdf_path.resolve()
    try:
        with fitz.open(resolved) as doc:  # type: ignore[attr-defined]
            page_count = doc.page_count
    except Exception:
        return {}

    if page_count <= 0:
        return {}

    if first_body_pdf_page and first_body_pdf_page > 1:
        search_end = max(1, min(page_count, first_body_pdf_page - 1))
    else:
        search_end = min(page_count, 50)

    results: dict[int, int | None] = {}

    for idx in indices:
        entry = entries[idx]
        content = str(entry.get("content", "")).strip()
        if not content:
            results[idx] = None
            continue

        page_num: int | None = None

        # Stage 1: text search in front-matter range.
        if locate_entry_by_text_search is not None:
            try:
                page_num = locate_entry_by_text_search(
                    resolved,
                    content,
                    search_start=1,
                    search_end=search_end,
                )
            except Exception:
                page_num = None

        # Stage 2: optional VLM heading search when text search fails.
        if page_num is None and api_key:
            page_num = locate_heading_by_vlm(
                resolved,
                content,
                search_start=1,
                search_end=search_end,
                api_key=api_key,
                timeout=timeout,
                api_base=api_base,
                model=model,
            )

        results[idx] = page_num

    return results


def _get_printed_page_number(
    pdf_path: Path,
    index: int,
    api_key: str,
    timeout: int,
    api_base: str | None = None,
    model: str | None = None,
) -> int | None:
    cache_key = (str(pdf_path), index)
    if cache_key in _PAGE_NUMBER_CACHE:
        cached = _PAGE_NUMBER_CACHE[cache_key]
        return int(cached) if isinstance(cached, int) else cached

    image_b64 = _render_page_image_base64(pdf_path, index)
    if image_b64 is None:
        _PAGE_NUMBER_CACHE[cache_key] = None
        return None

    result = _query_page_number(
        api_key,
        image_b64,
        timeout,
        api_base=api_base,
        model=model,
    )
    _PAGE_NUMBER_CACHE[cache_key] = result
    return result


def _render_page_image_base64(pdf_path: Path, index: int) -> str | None:
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return None

    try:
        with fitz.open(pdf_path) as doc:  # type: ignore[attr-defined]
            if index < 0 or index >= doc.page_count:
                return None
            page = doc.load_page(index)
            scale = DEFAULT_DPI_SCALE
            try:
                rect = page.rect
                max_dim = max(rect.width, rect.height)
                if max_dim > 0:
                    projected = max_dim * scale
                    if projected > MAX_IMAGE_DIMENSION:
                        scale = MAX_IMAGE_DIMENSION / max_dim
            except Exception:
                # Fall back to default scale when geometry inspection fails.
                pass
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix)
            image_bytes = _pixmap_to_jpeg(pix)
    except Exception:
        return None

    return base64.b64encode(image_bytes).decode("ascii") if image_bytes else None


def _query_page_number(
    api_key: str,
    image_b64: str,
    timeout: int,
    api_base: str | None = None,
    model: str | None = None,
) -> int | None:
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
        "model": model or MODEL_NAME,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        body = _call_chat_completion(
            api_key,
            payload,
            request_timeout=timeout,
            api_base=api_base,
        )
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
    except (json.JSONDecodeError, ValueError):
        return None

    value = data.get("page_number")
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any) -> int | None:
    # Backward-compat shim: route to shared util
    return _util_coerce_positive_int(value)


def _pixmap_to_jpeg(pix: Any) -> bytes:
    try:
        return pix.tobytes("jpeg", quality=JPEG_QUALITY)
    except TypeError:
        return pix.tobytes("jpeg")


def _get_or_render_page_payload(
    pdf_path: Path, index: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    cache_key = (str(pdf_path), index)
    cached = _PAYLOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1]

    payload, fingerprint = _render_page_payload(pdf_path, index)
    if payload is not None:
        _PAYLOAD_CACHE[cache_key] = (payload, fingerprint)
    return payload, fingerprint


def _render_page_payload(
    pdf_path: Path, index: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
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

        entry: dict[str, Any] = {"page": index + 1}

        text = page.get_text("text").strip()
        fingerprint = build_page_fingerprint(page, text)
        if text:
            entry["text"] = _trim_text(text)
        else:
            scale = DEFAULT_DPI_SCALE
            try:
                rect = page.rect
                max_dim = max(rect.width, rect.height)
                if max_dim > 0:
                    projected = max_dim * scale
                    if projected > MAX_IMAGE_DIMENSION:
                        scale = MAX_IMAGE_DIMENSION / max_dim
            except Exception:
                # Fall back to default scale when geometry inspection fails.
                pass
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix)
            image_bytes = _pixmap_to_jpeg(pix)
            if not image_bytes:
                return None, fingerprint
            entry["image_b64"] = base64.b64encode(image_bytes).decode("ascii")
    return entry, fingerprint


def _purge_cache_for_path(pdf_path: Path) -> None:
    try:
        target = str(pdf_path.resolve())
    except FileNotFoundError:
        target = str(pdf_path.absolute())

    # Take a snapshot of matching keys to delete to avoid mutating during iteration.
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


def _safe_json(response: requests.Response | None) -> Any:
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
