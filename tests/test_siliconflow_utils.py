from __future__ import annotations

from ebooktoc.siliconflow_api import (
    _extract_json_block,
    _find_json_substring,
    _chunk_iterable,
    _trim_text,
)


def test_extract_json_block_with_fence_and_inline():
    fenced = """```
json
{\"toc\": []}
```"""
    # Function currently prefers array substring when present
    assert _extract_json_block(fenced) == '[]'

    inline = "prefix {\"toc\": [1]} suffix"
    assert _extract_json_block(inline) == '[1]'


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
