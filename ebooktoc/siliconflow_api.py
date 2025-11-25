"""SiliconFlow Qwen API client for extracting TOC data from PDFs."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

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
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> Path:
    """Extract TOC data using an OpenAI-format VLM API (default: SiliconFlow Qwen).

    Returns a temporary JSON file path containing the model output.
    """

    if task_id is not None:
        raise TOCExtractionError("VLM workflow does not support --task-id")

    source_path, cleanup_required = _resolve_source_pdf(pdf_path, remote_url)
    try:
        page_payloads, fingerprints = _collect_page_payloads(source_path, page_limit)
        aggregated: List[Dict[str, Any]] = []
        effective_model = model or MODEL_NAME

        effective_batch = max(1, batch_size)
        for batch_index, batch in enumerate(
            _chunk_iterable(page_payloads, effective_batch), start=1
        ):
            payload = _build_payload(batch, page_limit, model=effective_model)
            response_body = _call_chat_completion(
                api_key,
                payload,
                request_timeout=timeout,
                api_base=api_base,
            )
            toc_json = _parse_response_payload(response_body)
            aggregated.extend(toc_json)
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


def _collect_page_payloads(pdf_path: Path, max_pages: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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

    end_page = total_pages if max_pages <= 0 else min(max_pages, total_pages)
    payloads: List[Dict[str, Any]] = []
    fingerprints: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_get_or_render_page_payload, resolved, index)
            for index in range(end_page)
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
    model: Optional[str] = None,
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

    range_label = max_pages if max_pages > 0 else "all"
    intro_text = (
        f"The following content comes from the first {range_label} pages of a PDF. Each section begins "
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
        "model": model or MODEL_NAME,
        "temperature": 0.2,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }


def _call_chat_completion(
    api_key: str,
    payload: Dict[str, Any],
    request_timeout: int,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    base = (api_base or API_BASE_DEFAULT).rstrip("/")
    endpoint = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            endpoint, json=payload, headers=headers, timeout=request_timeout
        )
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


def _parse_response_payload(body: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    except json.JSONDecodeError as exc:
        raise TOCExtractionError(
            f"Failed to parse VLM JSON output: {exc}\nRaw: {json_text}"
        ) from exc

    toc: Any
    if isinstance(data, dict):
        if "toc" in data:
            toc = data["toc"]
        elif "output" in data and isinstance(data["output"], list):
            toc = data["output"]
        else:
            raise TOCExtractionError(
                f"VLM API response missing 'toc' key: {json_text}"
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
    decoder = json.JSONDecoder()
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = text[start : end + 1]
        try:
            decoder.decode(snippet)
        except json.JSONDecodeError:
            continue
        return snippet
    try:
        decoder.decode(text)
        return text
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
    api_base: Optional[str] = None,
    model: Optional[str] = None,
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
        page_number = _get_printed_page_number(
            resolved,
            index0,
            api_key,
            timeout,
            api_base=api_base,
            model=model,
        )
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
    api_base: Optional[str] = None,
    model: Optional[str] = None,
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

    result = _query_page_number(
        api_key,
        image_b64,
        timeout,
        api_base=api_base,
        model=model,
    )
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


def _query_page_number(
    api_key: str,
    image_b64: str,
    timeout: int,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[int]:
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
