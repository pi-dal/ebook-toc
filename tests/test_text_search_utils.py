"""Tests for text-search helpers used in entry location."""

from __future__ import annotations

from ebooktoc.toc_parser import _build_search_terms, _normalize_text_for_search


class TestNormalizeTextForSearch:
    """Tests for _normalize_text_for_search."""

    def test_removes_whitespace_and_lowercases(self) -> None:
        assert _normalize_text_for_search(" Hello   World ") == "helloworld"

    def test_removes_punctuation(self) -> None:
        result = _normalize_text_for_search('Hello, "World"!')
        # Focus on quoting characters being removed.
        assert '"' not in result


class TestBuildSearchTerms:
    """Tests for _build_search_terms."""

    def test_chapter_heading_generates_multiple_terms(self) -> None:
        terms = _build_search_terms("Chapter 1 Introduction to Physics")
        assert "chapter1introductiontophysics" in terms
        assert "introductiontophysics" in terms

    def test_chinese_chapter_heading(self) -> None:
        terms = _build_search_terms("第1章 电磁学基础")
        assert any("电磁学基础" in t for t in terms) or any("第1章电磁学基础" in t for t in terms)

    def test_section_number_heading(self) -> None:
        terms = _build_search_terms("1.1 Background and Motivation")
        assert "backgroundandmotivation" in terms
