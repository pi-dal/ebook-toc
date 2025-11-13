from __future__ import annotations

from ebooktoc.toc_parser import infer_missing_targets


def test_infer_missing_targets_dot_leaders_and_trailing_digits():
    raw = [
        {"page": 1, "content": "绪论 ...... 12"},
        {"page": 2, "content": "Chapter 1···45"},
        {"page": 3, "content": "Methods   78"},
        {"page": 4, "content": "1.2.3 Not capture"},
        {"page": 5, "content": "No number"},
    ]
    items = infer_missing_targets(raw)
    pages = [e.get("target_page") for e in items]
    # Should capture 12, 45, 78; others remain None
    assert pages[:3] == [12, 45, 78]
    assert pages[3:] == [None, None]

