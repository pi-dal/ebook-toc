from __future__ import annotations

from ebooktoc.siliconflow_api import (
    _extract_json_block,
    _find_json_substring,
    _parse_response_payload,
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
