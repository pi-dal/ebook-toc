"""Tests for page offset inference logic."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import ebooktoc.cli as cli

from ebooktoc.vlm_api import (
    _heuristic_offset_search,
    _infer_page_offset,
    _verify_offset,
)


def _install_fake_fitz(page_count: int) -> ModuleType:
    """Return a fake ``fitz`` module with configurable ``page_count``."""
    fake_mod = ModuleType("fitz")

    class _FakeDoc:
        def __init__(self, count: int) -> None:
            self.page_count = count

        def __enter__(self) -> "._FakeDoc":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    def _open(_path: Path) -> _FakeDoc:
        return _FakeDoc(page_count)

    fake_mod.open = _open  # type: ignore[attr-defined]
    return fake_mod


class TestHeuristicOffsetSearch:
    """Tests for ``_heuristic_offset_search``."""

    def test_finds_correct_offset_when_printed_matches_target(self) -> None:
        """Find offset when VLM returns matching printed page."""
        entries = [
            {"target_page": 50, "content": "Chapter 5"},
            {"target_page": 100, "content": "Chapter 10"},
        ]

        # Mock: PDF page 60 (0-indexed: 59) has printed page 50
        # and PDF page 110 (0-indexed: 109) has printed page 100.
        # So offset = 60 - 50 = 10.
        def mock_get_printed(pdf_path: Path, idx: int, api_key: str, timeout: int, **_: Any) -> int | None:
            if idx == 59:
                return 50
            if idx == 109:
                return 100
            return None

        with patch("ebooktoc.vlm_api._get_printed_page_number", mock_get_printed):
            offset = _heuristic_offset_search(
                Path("/fake.pdf"),
                entries,
                "fake_key",
                60,
                page_count=200,
            )

        assert offset == 10

    def test_returns_none_when_no_match_found(self) -> None:
        """Return None when no matching page is found."""
        entries = [{"target_page": 50, "content": "Chapter 5"}]

        def mock_get_printed(*_: Any, **__: Any) -> int | None:
            return None

        with patch("ebooktoc.vlm_api._get_printed_page_number", mock_get_printed):
            offset = _heuristic_offset_search(
                Path("/fake.pdf"),
                entries,
                "fake_key",
                60,
                page_count=200,
            )

        assert offset is None


class TestVerifyOffset:
    """Tests for ``_verify_offset``."""

    def test_returns_true_when_majority_matches(self) -> None:
        """Return True when majority of verifications succeed."""
        entries = [
            {"target_page": 30, "content": "A"},
            {"target_page": 60, "content": "B"},
            {"target_page": 90, "content": "C"},
        ]

        # With offset=10, PDF pages 40, 70, 100 (1-based) correspond to
        # indices 39, 69, 99 and should have printed pages 30, 60, 90.
        def mock_get_printed(pdf_path: Path, idx: int, api_key: str, timeout: int, **_: Any) -> int | None:
            mapping = {39: 30, 69: 60, 99: 90}
            return mapping.get(idx)

        with patch("ebooktoc.vlm_api._get_printed_page_number", mock_get_printed):
            result = _verify_offset(
                Path("/fake.pdf"),
                entries,
                offset=10,
                api_key="fake",
                timeout=60,
                page_count=200,
            )

        assert result is True

    def test_returns_false_when_majority_fails(self) -> None:
        """Return False when majority of verifications fail."""
        entries = [
            {"target_page": 30, "content": "A"},
            {"target_page": 60, "content": "B"},
            {"target_page": 90, "content": "C"},
        ]

        def mock_get_printed(pdf_path: Path, idx: int, api_key: str, timeout: int, **_: Any) -> int | None:
            return 999

        with patch("ebooktoc.vlm_api._get_printed_page_number", mock_get_printed):
            result = _verify_offset(
                Path("/fake.pdf"),
                entries,
                offset=10,
                api_key="fake",
                timeout=60,
                page_count=200,
            )

        assert result is False


class TestInferPageOffset:
    """Integration tests for ``_infer_page_offset``."""

    def test_uses_heuristic_when_successful(self) -> None:
        """Use heuristic method when it can infer offset."""
        entries = [{"target_page": 50, "content": "Chapter"}]

        call_count = {"heuristic": 0}

        def mock_get_printed(pdf_path: Path, idx: int, api_key: str, timeout: int, **_: Any) -> int | None:
            # Match at PDF index 59 (page 60) for target_page 50 -> offset 10.
            if idx == 59:
                call_count["heuristic"] += 1
                return 50
            return None

        fake_fitz = _install_fake_fitz(page_count=200)

        with patch.dict("sys.modules", {"fitz": fake_fitz}):
            with patch("ebooktoc.vlm_api._get_printed_page_number", mock_get_printed):
                offset = _infer_page_offset(
                    Path("/fake.pdf"),
                    entries,
                    "fake_key",
                    60,
                )

        assert offset == 10
        assert call_count["heuristic"] > 0


class TestOffsetWithGoodNotes:
    """Tests for offset inference with GoodNotes-inserted pages."""

    def test_infer_offset_uses_clean_pdf_with_goodnotes_and_api_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Offset inference should run on a clean PDF when GoodNotes pages are present."""
        pdf_path = tmp_path / "book.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%mock")

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

        # Simulate original fps with one GoodNotes page (index 2) of different size.
        original_fps = [
            {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
            {"width": 200, "height": 200, "text_len": 0, "text_hash": None, "image_count": 0},
            {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
            {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
        ]

        clean_path = tmp_path / "clean.pdf"
        clean_to_original = {1: 1, 2: 3, 3: 4}
        clean_fps = [
            {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
            {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
            {"width": 100, "height": 100, "text_len": 0, "text_hash": None, "image_count": 0},
        ]

        def fake_compute_pdf_fingerprints(path: Path):
            if Path(path) == clean_path:
                return clean_fps, 3
            return original_fps, 4

        def fake_build_clean_pdf(path: Path, keep_indices: list[int]):
            assert keep_indices == [1, 3, 4]
            clean_path.write_bytes(b"%PDF-1.4\n%mock clean")
            return clean_path, clean_to_original

        captured: dict[str, Any] = {}

        def fake_infer_page_offset(
            pdf: Path,
            entries: list[dict[str, Any]],
            api_key: str,
            timeout: int,
            fingerprints: list[dict[str, Any]] | None = None,
            api_base: str | None = None,
            model: str | None = None,
        ) -> int | None:
            captured["infer_pdf"] = pdf
            captured["infer_fps"] = fingerprints
            return 7

        def fake_write_pdf_toc(pdf: Path, entries: list[dict[str, Any]], output_path: Path, page_offset: int | None = None):
            captured["written_pages"] = [e.get("target_page") for e in entries]
            captured["page_offset"] = page_offset
            return SimpleNamespace(added=len(entries), skipped=[], output_path=Path(output_path))

        monkeypatch.setattr(cli, "compute_pdf_fingerprints", fake_compute_pdf_fingerprints)
        monkeypatch.setattr(cli, "_build_clean_pdf", fake_build_clean_pdf)
        monkeypatch.setattr(cli, "_infer_page_offset", fake_infer_page_offset)
        monkeypatch.setattr(cli, "write_pdf_toc", fake_write_pdf_toc)

        args = SimpleNamespace(
            pdf=pdf_path,
            json=json_path,
            output=tmp_path / "book_with_toc.pdf",
            override_offset=None,
            api_key="fake-key",
            api_base=None,
            model=None,
            timeout=30,
            goodnotes_clean=True,
            verify_printed=False,
        )

        cli._run_apply(args)

        # Ensure VLM-based offset inference ran against the clean PDF and did not reuse original fingerprints.
        assert captured.get("infer_pdf") == clean_path
        assert captured.get("infer_fps") is None
