"""Tests for TOC entry classification and anchor selection."""

from __future__ import annotations

from ebooktoc.toc_parser import (
    classify_entries,
    select_anchor_entries,
    _is_preface_like_content,
    _calculate_anchor_priority,
)


class TestClassifyEntries:
    """Tests for classify_entries."""

    def test_identifies_unnumbered_entries(self) -> None:
        entries = [
            {"content": "Preface", "target_page": None},
            {"content": "Chapter 1", "target_page": 1},
        ]
        unnumbered, suspicious, normal = classify_entries(entries)
        assert unnumbered == [0]
        assert suspicious == []
        assert normal == [1]

    def test_identifies_suspicious_entries(self) -> None:
        entries = [
            {"content": "Preface to 2nd Edition", "target_page": 1},
            {"content": "Preface to 1st Edition", "target_page": 1},
            {"content": "Foreword", "target_page": 1},
            {"content": "Acknowledgments", "target_page": 1},
            {"content": "Chapter 1 Introduction", "target_page": 1},
            {"content": "1.1 Background", "target_page": 1},
            {"content": "1.2 Methods", "target_page": 5},
        ]
        unnumbered, suspicious, normal = classify_entries(entries)
        # Preface-like entries with page 1 are suspicious.
        assert 0 in suspicious
        assert 1 in suspicious
        assert 2 in suspicious or 3 in suspicious

    def test_all_normal_entries(self) -> None:
        entries = [
            {"content": "Chapter 1", "target_page": 1},
            {"content": "Chapter 2", "target_page": 25},
            {"content": "Chapter 3", "target_page": 50},
        ]
        unnumbered, suspicious, normal = classify_entries(entries)
        assert unnumbered == []
        assert suspicious == []
        assert normal == [0, 1, 2]


class TestSelectAnchorEntries:
    """Tests for select_anchor_entries."""

    def test_prefers_chapter_headings(self) -> None:
        entries = [
            {"content": "1.1 Some Section", "target_page": 5},
            {"content": "Chapter 1 Introduction", "target_page": 1},
            {"content": "1.2 Another Section", "target_page": 10},
        ]
        anchors = select_anchor_entries(entries, [0, 1, 2], max_anchors=2)
        assert any(a.content == "Chapter 1 Introduction" for a in anchors)

    def test_spreads_across_page_ranges(self) -> None:
        entries = [
            {"content": "Chapter 1", "target_page": 1},
            {"content": "Chapter 2", "target_page": 10},
            {"content": "Chapter 3", "target_page": 20},
            {"content": "Chapter 10", "target_page": 150},
            {"content": "Chapter 15", "target_page": 250},
        ]
        anchors = select_anchor_entries(entries, [0, 1, 2, 3, 4], max_anchors=3)
        pages = [a.printed_page for a in anchors]
        assert any(p < 50 for p in pages)
        assert any(p >= 100 for p in pages)


class TestPrefaceHeuristics:
    """Tests for preface-like content helpers."""

    def test_is_preface_like_content(self) -> None:
        assert _is_preface_like_content("前言")
        assert _is_preface_like_content("第二版丛书序")
        assert _is_preface_like_content("Preface")
        assert not _is_preface_like_content("Chapter 1 Introduction")

    def test_calculate_anchor_priority(self) -> None:
        # Chapter-like content should receive a higher score than a plain title.
        score_chapter = _calculate_anchor_priority("第1章 概述", 10)
        score_plain = _calculate_anchor_priority("概述", 10)
        assert score_chapter > score_plain
