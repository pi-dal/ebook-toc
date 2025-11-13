from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json

import ebooktoc.cli as cli


def test_apply_with_goodnotes_clean_uses_clean_to_original_mapping(tmp_path, monkeypatch):
    # Prepare a dummy PDF and JSON
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%...mock...")

    data = {
        "toc": [
            {"page": 1, "target_page": 2, "content": "Chapter 1"},
            {"page": 1, "target_page": 3, "content": "Chapter 2"},
        ],
        "page_offset": 0,
        "fingerprints": [],
    }
    json_path = tmp_path / "book_toc.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    # Simulate original fps with one GoodNotes page (index 2) of different size
    original_fps = [
        {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
        {"width": 200, "height": 200, "text_len": 0, "text_hash": None, "image_count": 0},  # GN
        {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
        {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
    ]

    # Clean PDF would contain pages [1,3,4] -> clean indices [1,2,3]
    clean_path = tmp_path / "clean.pdf"
    clean_to_original = {1: 1, 2: 3, 3: 4}
    clean_fps = [
        {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
        {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
        {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
    ]

    def fake_compute_pdf_fingerprints(path):
        if Path(path) == clean_path:
            return (clean_fps, 3)
        return (original_fps, 4)

    def fake_build_clean_pdf(path, keep_indices):
        # Ensure keep_indices matches [1,3,4]
        assert keep_indices == [1, 3, 4]
        clean_path.write_bytes(b"%PDF-1.4\n%...mock clean...")
        return (clean_path, clean_to_original)

    captured = {}

    def fake_write_pdf_toc(pdf, entries, output_path, page_offset=None):
        captured["pages"] = [e.get("target_page") for e in entries]
        return SimpleNamespace(added=len(entries), skipped=[], output_path=Path(output_path))

    monkeypatch.setattr(cli, "compute_pdf_fingerprints", fake_compute_pdf_fingerprints)
    monkeypatch.setattr(cli, "_build_clean_pdf", fake_build_clean_pdf)
    monkeypatch.setattr(cli, "write_pdf_toc", fake_write_pdf_toc)

    args = SimpleNamespace(
        pdf=pdf_path,
        json=json_path,
        output=tmp_path / "book_with_toc.pdf",
        override_offset=None,
        api_key=None,
        timeout=30,
        goodnotes_clean=True,
    )

    cli._run_apply(args)

    # Both TOC items should resolve to original pages [3,4]
    assert captured.get("pages") == [3, 4]

