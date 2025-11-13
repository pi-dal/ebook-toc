from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import builtins

from ebooktoc import utils


def test_coerce_positive_int_behaviour():
    c = utils.coerce_positive_int
    assert c(1) == 1
    assert c("2") == 2
    assert c(0) is None
    assert c(-3) is None
    assert c(None) is None
    assert c("x") is None


def test_download_to_temp(monkeypatch, tmp_path):
    content = b"hello"

    class FakeResponse:
        def __init__(self, data: bytes):
            self.content = data

        def raise_for_status(self):
            return None

    def fake_get(url, timeout=60):
        assert url == "https://example.com/file"
        assert timeout == 60
        return FakeResponse(content)

    import requests

    monkeypatch.setattr(requests, "get", fake_get)

    path = utils.download_to_temp("https://example.com/file", prefix="t-", suffix=".bin")
    try:
        assert Path(path).is_file()
        assert Path(path).read_bytes() == content
    finally:
        Path(path).unlink(missing_ok=True)

