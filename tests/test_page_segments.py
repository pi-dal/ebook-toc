"""Tests for multi-segment page numbering detection and merging."""

from __future__ import annotations

from typing import Any

from ebooktoc.toc_parser import (
    PageSegment,
    detect_page_segments,
    _classify_segment,
    merge_segments_with_offsets,
)


class TestDetectPageSegments:
    """Tests for detect_page_segments."""

    def test_single_segment_no_reset(self) -> None:
        """TOC with continuous page numbers should yield one segment."""
        entries = [
            {"content": "Chapter 1", "target_page": 1},
            {"content": "Chapter 2", "target_page": 15},
            {"content": "Chapter 3", "target_page": 30},
        ]
        segments = detect_page_segments(entries)
        assert len(segments) == 1
        assert segments[0].start_idx == 0
        assert segments[0].end_idx == len(entries)
        assert segments[0].entries == entries

    def test_two_segments_preface_and_body(self) -> None:
        """TOC with preface and body should have two segments."""
        entries = [
            {"content": "序言", "target_page": 1},
            {"content": "前言", "target_page": 3},
            {"content": "目录", "target_page": 5},
            {"content": "第1章 导论", "target_page": 1},  # reset
            {"content": "1.1 背景", "target_page": 5},
            {"content": "第2章", "target_page": 20},
        ]
        segments = detect_page_segments(entries)
        assert len(segments) == 2
        assert [len(s.entries) for s in segments] == [3, 3]

    def test_detects_multiple_resets(self) -> None:
        """Should detect multiple page resets."""
        entries = [
            {"content": "Preface", "target_page": 1},
            {"content": "Chapter 1", "target_page": 1},  # reset 1
            {"content": "Chapter 2", "target_page": 50},
            {"content": "Appendix A", "target_page": 1},  # reset 2
        ]
        segments = detect_page_segments(entries)
        assert len(segments) == 3

    def test_empty_entries(self) -> None:
        """Empty entries should produce no segments."""
        assert detect_page_segments([]) == []

    def test_entries_without_target_page(self) -> None:
        """Entries without target_page should not break detection."""
        entries = [
            {"content": "A", "target_page": 1},
            {"content": "B"},  # no target_page
            {"content": "C", "target_page": 5},
        ]
        segments = detect_page_segments(entries)
        assert len(segments) == 1


class TestClassifySegment:
    """Tests for _classify_segment."""

    def test_preface_keywords(self) -> None:
        """Segments with preface-related keywords classify as preface."""
        entries = [
            {"content": "序言"},
            {"content": "前言"},
        ]
        result = _classify_segment(entries, segment_index=0, total_segments=2)
        assert result == "preface"

    def test_body_with_chapters(self) -> None:
        """Segments with chapter-style content classify as body."""
        entries = [
            {"content": "第1章 概述"},
            {"content": "1.1 简介"},
            {"content": "1.2 方法"},
        ]
        result = _classify_segment(entries, segment_index=1, total_segments=2)
        assert result == "body"

    def test_appendix_keywords(self) -> None:
        """Segments with appendix keywords classify as appendix."""
        entries = [
            {"content": "附录A"},
            {"content": "参考文献"},
            {"content": "索引"},
        ]
        result = _classify_segment(entries, segment_index=2, total_segments=3)
        assert result == "appendix"


class TestMergeSegmentsWithOffsets:
    """Tests for merge_segments_with_offsets."""

    def test_applies_different_offsets(self) -> None:
        """Different segments should receive different offsets."""
        segments = [
            PageSegment(
                start_idx=0,
                end_idx=2,
                entries=[
                    {"content": "序言", "target_page": 1},
                    {"content": "前言", "target_page": 3},
                ],
                segment_type="preface",
                offset=9,  # PDF page 10 = preface page 1
            ),
            PageSegment(
                start_idx=2,
                end_idx=4,
                entries=[
                    {"content": "第1章", "target_page": 1},
                    {"content": "第2章", "target_page": 20},
                ],
                segment_type="body",
                offset=16,  # PDF page 17 = body page 1
            ),
        ]

        result = merge_segments_with_offsets(segments)

        assert len(result) == 4
        # Preface: 1 + 9 = 10, 3 + 9 = 12
        assert result[0]["target_page"] == 10
        assert result[1]["target_page"] == 12
        # Body: 1 + 16 = 17, 20 + 16 = 36
        assert result[2]["target_page"] == 17
        assert result[3]["target_page"] == 36

        # Metadata
        assert result[0]["_segment_type"] == "preface"
        assert result[2]["_segment_type"] == "body"
        assert result[0]["_original_target_page"] == 1
        assert result[2]["_original_target_page"] == 1

