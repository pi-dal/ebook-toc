from __future__ import annotations

import json

import requests

import ebooktoc.siliconflow_api as api
from ebooktoc.siliconflow_api import (
    _extract_json_block,
    _find_json_substring,
    _parse_response_payload,
    _is_retryable_error,
    _chunk_iterable,
    _trim_text,
)


def test_extract_json_block_with_fence_and_inline():
    fenced = """```
json
{\"toc\": []}
```"""
    # When given a fenced JSON object, the helper should return the inner
    # JSON object string so that the caller can decide how to interpret it.
    assert _extract_json_block(fenced) == '{"toc": []}'

    inline = "prefix {\"toc\": [1]} suffix"
    assert _extract_json_block(inline) == '{"toc": [1]}'


def test_find_json_substring_handles_arrays_and_objects():
    text = "noise [1,2,3] more"
    assert _find_json_substring(text) == "[1,2,3]"

    text2 = "noise {\"a\":1} tail"
    assert _find_json_substring(text2) == '{"a":1}'


def test_chunk_iterable_and_trim_text():
    items = list(_chunk_iterable(range(5), 2))
    assert items == [[0, 1], [2, 3], [4]]

    long_text = "\n".join(str(i) for i in range(200))
    t = _trim_text(long_text, max_lines=10, max_chars=15)
    assert len(t.splitlines()) <= 10
    assert len(t) <= 15


def test_parse_response_payload_repairs_unquoted_keys():
    # Simulate a SiliconFlow response where the model emitted bare keys
    # inside the toc array, which would normally be invalid JSON.
    body = {
        "choices": [
            {
                "message": {
                    "content": '{"toc": [{page: 11, target_page: 67, content: "Chapter 2"}, '
                    '{page: 12, target_page: 70, content: "Chapter 3"}]}'
                }
            }
        ]
    }

    toc = _parse_response_payload(body)
    assert isinstance(toc, list)
    assert len(toc) == 2
    assert toc[0]["page"] == 11
    assert toc[0]["target_page"] == 67
    assert toc[0]["content"] == "Chapter 2"


def test_parse_response_payload_handles_brackets_in_content():
    # Ensure that brackets inside content strings do not confuse JSON extraction.
    body = {
        "choices": [
            {
                "message": {
                    "content": '{"toc": [{\"page\": 4, \"target_page\": 5, \"content\": \"[Page 4] 引言\"}]}'
                }
            }
        ]
    }

    toc = _parse_response_payload(body)
    assert isinstance(toc, list)
    assert len(toc) == 1
    assert toc[0]["page"] == 4
    assert toc[0]["target_page"] == 5
    assert toc[0]["content"] == "[Page 4] 引言"


def test_parse_response_payload_accepts_single_entry_object():
    # If the model returns a bare TOC entry object instead of {\"toc\": [...]},
    # the parser should coerce it into a singleton list.
    body = {
        "choices": [
            {
                "message": {
                    "content": '{"page": 11, "target_page": 67, "content": "2.5.2 小带电体在外电场中的静电能"}'
                }
            }
        ]
    }

    toc = _parse_response_payload(body)
    assert isinstance(toc, list)
    assert len(toc) == 1
    entry = toc[0]
    assert entry["page"] == 11
    assert entry["target_page"] == 67
    assert entry["content"].startswith("2.5.2 小带电体在外电场中的静电能")


def test_fetch_document_json_page_map_respects_start_page(monkeypatch, tmp_path):
    # Ensure that when fetch_document_json is called with a non-1 start_page,
    # the emitted page_map expresses absolute PDF page numbers rather than
    # window-relative indices.
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("dummy", encoding="utf-8")

    def fake_collect(pdf_path, max_pages, *, start_page=1):
        assert pdf_path == pdf
        assert max_pages == 2
        assert start_page == 3
        payloads = [
            {"page": start_page, "text": "p3"},
            {"page": start_page + 1, "text": "p4"},
        ]
        fps = [
            {"width": 100, "height": 200},
            {"width": 100, "height": 200},
        ]
        return payloads, fps

    def fake_call(api_key, payload, request_timeout):
        # Minimal SiliconFlow-style body; content is strict JSON.
        return {"choices": [{"message": {"content": '{"toc": []}'}}]}

    def fake_infer_offset(pdf_path, entries, api_key, timeout, fingerprints, max_samples=3):
        return None

    monkeypatch.setattr(api, "_collect_page_payloads", fake_collect)
    monkeypatch.setattr(api, "_call_chat_completion", fake_call)
    monkeypatch.setattr(api, "_infer_page_offset", fake_infer_offset)

    json_path = api.fetch_document_json(
        pdf_path=pdf,
        api_key="test-key",
        page_limit=2,
        start_page=3,
        batch_size=1,
    )

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    page_map = data.get("page_map")
    # Canonical indices start at 1; JSON serialisation turns keys into strings.
    # The underlying window covers pages 3 and 4, so values should be 3 and 4.
    assert page_map == {"1": 3, "2": 4}


def test_is_retryable_error_handles_http_and_network_errors():
    # HTTP 500 and 429 should be considered retryable.
    for status in (429, 500):
        response = requests.Response()
        response.status_code = status
        cause = requests.HTTPError(response=response)
        exc = RuntimeError("wrapper")
        # Manually attach cause to mimic \"raise ... from\" semantics
        exc.__cause__ = cause  # type: ignore[attr-defined]
        assert _is_retryable_error(exc) is True

    # HTTP 400 should not be retryable.
    response = requests.Response()
    response.status_code = 400
    cause = requests.HTTPError(response=response)
    exc = RuntimeError("wrapper")
    exc.__cause__ = cause  # type: ignore[attr-defined]
    assert _is_retryable_error(exc) is False

    # Transient network errors like Timeout / ConnectionError should be retryable.
    for err in (requests.Timeout(), requests.ConnectionError()):
        exc = RuntimeError("wrapper")
        exc.__cause__ = err  # type: ignore[attr-defined]
        assert _is_retryable_error(exc) is True
