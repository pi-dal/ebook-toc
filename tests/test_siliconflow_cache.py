from __future__ import annotations

from pathlib import Path

import ebooktoc.vlm_api as api


def test_payload_cache_returns_fingerprint_not_page_number(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("dummy", encoding="utf-8")

    payload = {"page": 1, "text": "stub"}
    fingerprint = {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0}

    def fake_render(pdf_path, index):
        # Ensure our stub is used rather than attempting real rendering
        assert Path(pdf_path) == pdf
        assert index == 0
        return payload, fingerprint

    # Clear caches and deliberately poison page-number cache with an int
    api._PAYLOAD_CACHE.clear()
    api._PAGE_NUMBER_CACHE.clear()
    api._PAGE_NUMBER_CACHE[(str(pdf), 0)] = 42

    monkeypatch.setattr(api, "_render_page_payload", fake_render)

    p1, f1 = api._get_or_render_page_payload(pdf, 0)
    assert p1 == payload
    assert f1 == fingerprint

    # Second call should come from _PAYLOAD_CACHE and still yield fingerprint dict
    p2, f2 = api._get_or_render_page_payload(pdf, 0)
    assert p2 == payload
    assert f2 == fingerprint
