from __future__ import annotations

from pathlib import Path

import ebooktoc.cli as cli


def test_apply_page_mapping_prefers_canonical_then_offset_then_clamp():
    entries = [
        {"content": "A", "target_page": 5},     # 5 + 5 -> 10 -> mapping[10]=100
        {"content": "B", "target_page": 7},     # direct mapping 7 -> 70
        {"content": "C", "target_page": 200},   # clamp to page_count
        {"content": "D", "page": 3},            # fallback to 'page' field
    ]
    mapping = {10: 100, 7: 70}
    resolved = cli._apply_page_mapping(entries, mapping, page_offset=5, page_count=120)
    pages = [e.get("target_page") for e in resolved]
    # For the last item, mapping miss falls back to offset-adjusted numeric (3+5=8)
    assert pages == [100, 70, 120, 8]


def test_refine_offset_with_mapping_selects_best_offset():
    # entries that would align with offset=10 (targets 1,2,3 -> mapping keys 11,12,13)
    entries = [
        {"content": "a", "target_page": 1},
        {"content": "b", "target_page": 2},
        {"content": "c", "target_page": 3},
    ]
    mapping = {11: 101, 12: 102, 13: 103}
    best = cli._refine_offset_with_mapping(entries, mapping, initial_offset=None, window=15)
    assert best == 10


def test_derive_output_stem_local_and_remote(tmp_path):
    pdf = tmp_path / "foo.pdf"
    pdf.write_text("x", encoding="utf-8")
    assert cli._derive_output_stem(pdf, None) == "foo"
    assert cli._derive_output_stem(None, "https://example.com/bar.pdf?x=1") == "bar"
    assert cli._derive_output_stem(None, None) == "toc"


def test_detect_goodnotes_indices_from_fps():
    dims = (100, 200)
    fps = [
        {"width": 100, "height": 200},  # keep 1
        {"width": 120, "height": 240},  # remove 2
        {"width": 100, "height": 200},  # keep 3
    ]
    keep, removed = cli._detect_goodnotes_indices_from_fps(fps, dims)
    assert keep == [1, 3]
    assert removed == [2]
