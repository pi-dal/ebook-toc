from __future__ import annotations

from ebooktoc.toc_parser import (
    extract_toc_entries,
    deduplicate_entries,
    filter_entries,
)


def test_extract_from_list_and_normalization():
    raw = [
        {"page": 1, "content": " Intro ", "target_page": None},
        {"page": "2", "content": "Chapter 1", "target_page": "10"},
        {"page": "x", "content": "Bad"},  # invalid page
        {"page": 3, "content": "   "},  # empty content
    ]
    items = extract_toc_entries(raw)
    assert items == [
        {"page": 1, "content": "Intro"},
        {"page": 2, "content": "Chapter 1", "target_page": 10},
    ]


def test_extract_from_dict_with_pages_blocks():
    doc = {
        "pages": [
            {
                "page": 5,
                "blocks": [
                    {"block_type": "title", "text": "目录"},
                    {"type": "text", "content": "not a toc line"},
                ],
            },
            {
                "page_number": 6,
                "blocks": [
                    {"type": "text", "content": "Table of Contents"},
                ],
            },
        ]
    }
    items = extract_toc_entries(doc)
    assert {item["page"] for item in items} == {5, 6}


def test_deduplicate_entries_by_content_and_target():
    raw = [
        {"page": 1, "content": "Chapter A", "target_page": 10},
        {"page": 2, "content": "Chapter A", "target_page": 10},  # dup
        {"page": 3, "content": "chapter a", "target_page": 10},  # dup by case
        {"page": 4, "content": "Chapter A", "target_page": 11},  # different target keeps
    ]
    items = deduplicate_entries(raw)
    assert len(items) == 2
    # Content preserved, one for target 10, one for 11
    assert sorted({i.get("target_page") for i in items}) == [10, 11]


def test_deduplicate_entries_handles_chapter_number_variants():
    raw = [
        {"page": 1, "content": "第一章 绪论 ...... 1", "target_page": 1},
        {"page": 2, "content": "第1章 绪论 1", "target_page": 1},
    ]
    items = deduplicate_entries(raw)
    assert len(items) == 1


def test_deduplicate_entries_fuzzy_match_recent_entries():
    raw = [
        {"page": 1, "content": "第1章 绪论", "target_page": 1},
        {"page": 2, "content": "第一章 绪论与背景", "target_page": 1},
    ]
    # With fuzzy matching, these should collapse into one entry.
    items = deduplicate_entries(raw, fuzzy_threshold=0.85)
    assert len(items) == 1

    # Disabling fuzzy matching should keep both (normalisation alone differs).
    items_no_fuzzy = deduplicate_entries(raw, fuzzy_threshold=None)
    assert len(items_no_fuzzy) == 2


def test_filter_entries_contains_and_regex():
    raw = [
        {"page": 1, "content": "Introduction"},
        {"page": 2, "content": "Methods and Materials"},
        {"page": 3, "content": "Results"},
    ]
    # substring filter
    items = filter_entries(raw, contains="methods")
    assert [i["page"] for i in items] == [2]
    # regex filter
    import re

    items = filter_entries(raw, pattern=re.compile(r"^re", re.I))
    assert [i["page"] for i in items] == [3]
