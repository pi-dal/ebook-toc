from __future__ import annotations

from pathlib import Path
import threading

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


def test_lru_cache_evicts_least_recently_used():
    cache = api.LRUCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    # Access "a" so that "b" becomes the least recently used entry.
    _ = cache["a"]
    cache["c"] = 3

    assert "a" in cache
    assert "c" in cache
    assert "b" not in cache


def test_lru_cache_set_alias_and_basic_get():
    cache = api.LRUCache(maxsize=2)
    cache.set("x", 1)
    assert cache["x"] == 1
    # get() should mark the key as recently used without raising.
    assert cache.get("x") == 1


def test_lru_cache_thread_safety_under_concurrent_access():
    cache = api.LRUCache(maxsize=10)

    def worker(offset: int) -> None:
        for i in range(100):
            key = f"k{(i + offset) % 20}"
            cache[key] = i
            _ = cache.get(key)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Cache should respect its max size and not raise during concurrent access.
    assert len(cache) <= 10
