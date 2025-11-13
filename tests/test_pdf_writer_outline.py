from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import sys


def test_pdf_writer_fallback_outline(monkeypatch, tmp_path):
    # Prepare a dummy PDF file path
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%...mock...")
    out_path = tmp_path / "out.pdf"

    # Fake fitz module and document
    class FakeDoc:
        def __init__(self):
            self.page_count = 50
            self.outlines = []

        def set_toc(self, rows):
            # Force fallback path
            raise RuntimeError("set_toc failed")

        def add_outline(self, title, page_index):
            self.outlines.append((title, page_index))
            return True

        def save(self, dest, deflate=True):
            # simulate save without writing
            return None

        def close(self):
            return None

    class FakeFitz:
        def open(self, path):  # type: ignore[no-redef]
            return FakeDoc()

    # Inject fake module under name 'fitz'
    sys.modules['fitz'] = FakeFitz()

    from ebooktoc.pdf_writer import write_pdf_toc

    entries = [
        {"content": "A", "target_page": 2},
        {"content": "B", "page": 3},
        {"content": "C", "target_page": 0},  # invalid -> skip
    ]

    result = write_pdf_toc(pdf_path, entries, out_path)
    # Two valid bookmarks added via add_outline fallback
    assert result.added == 2
    assert result.output_path == out_path.resolve()

