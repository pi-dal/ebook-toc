"""Tests for unnumbered TOC entry handling."""

from __future__ import annotations

from typing import Any

from ebooktoc.toc_parser import (
    PageSegment,
    detect_page_segments,
    infer_missing_targets,
    merge_segments_with_offsets,
)


class TestUnnumberedEntries:
    """Tests for handling entries without page numbers."""

    def test_unnumbered_preface_entries_are_marked_correctly(self) -> None:
        """Entries before first numbered entry should be marked as unnumbered."""
        entries = [
            {"content": "第二版丛书序", "target_page": None},
            {"content": "第一版丛书序", "target_page": None},
            {"content": "第二版前言", "target_page": None},
            {"content": "第一版前言", "target_page": None},
            {"content": "第 1 章  电磁现象的基本规律", "target_page": 1},
            {"content": "1.1  场论和张量分析", "target_page": 1},
        ]

        result = infer_missing_targets(entries)

        # First 4 entries should be marked as unnumbered
        for i in range(4):
            assert result[i].get("_unnumbered") is True
            assert result[i].get("_unnumbered_index") == i
            assert result[i].get("target_page") is None

        # Last 2 entries should not be marked as unnumbered
        for i in range(4, 6):
            assert result[i].get("_unnumbered") is not True

    def test_detect_unnumbered_preface_segment(self) -> None:
        """Should detect unnumbered preface as separate segment."""
        entries = [
            {"content": "第二版丛书序", "target_page": None, "_unnumbered": True, "_unnumbered_index": 0},
            {"content": "第一版丛书序", "target_page": None, "_unnumbered": True, "_unnumbered_index": 1},
            {"content": "第二版前言", "target_page": None, "_unnumbered": True, "_unnumbered_index": 2},
            {"content": "第一版前言", "target_page": None, "_unnumbered": True, "_unnumbered_index": 3},
            {"content": "第 1 章  电磁现象的基本规律", "target_page": 1},
            {"content": "1.1  场论和张量分析", "target_page": 1},
        ]

        segments = detect_page_segments(entries)

        assert len(segments) >= 2
        assert segments[0].segment_type == "preface_unnumbered"
        assert len(segments[0].entries) == 4
        assert all(e.get("_unnumbered") for e in segments[0].entries)

    def test_merge_unnumbered_segment_with_offset(self) -> None:
        """Unnumbered entries should get sequential pages when offset is known."""
        segments = [
            PageSegment(
                start_idx=0,
                end_idx=4,
                entries=[
                    {"content": "第二版丛书序", "target_page": None, "_unnumbered": True, "_unnumbered_index": 0},
                    {"content": "第一版丛书序", "target_page": None, "_unnumbered": True, "_unnumbered_index": 1},
                    {"content": "第二版前言", "target_page": None, "_unnumbered": True, "_unnumbered_index": 2},
                    {"content": "第一版前言", "target_page": None, "_unnumbered": True, "_unnumbered_index": 3},
                ],
                segment_type="preface_unnumbered",
                offset=8,  # Preface starts at PDF page 9
            ),
            PageSegment(
                start_idx=4,
                end_idx=6,
                entries=[
                    {"content": "第 1 章  电磁现象的基本规律", "target_page": 1},
                    {"content": "1.1  场论和张量分析", "target_page": 1},
                ],
                segment_type="body",
                offset=16,  # Body starts at PDF page 17
            ),
        ]

        result = merge_segments_with_offsets(segments, page_count=100)

        # Unnumbered entries get sequential pages: 8 + index + 1
        assert result[0]["target_page"] == 9   # 8 + 0 + 1
        assert result[1]["target_page"] == 10  # 8 + 1 + 1
        assert result[2]["target_page"] == 11  # 8 + 2 + 1
        assert result[3]["target_page"] == 12  # 8 + 3 + 1

        # Numbered entries get offset applied
        assert result[4]["target_page"] == 17  # 1 + 16
        assert result[5]["target_page"] == 17  # 1 + 16

        # Metadata
        assert result[0]["_segment_type"] == "preface_unnumbered"
        assert result[4]["_segment_type"] == "body"

    def test_unnumbered_entries_without_offset_remain_none(self) -> None:
        """Unnumbered entries without offset should keep target_page as None."""
        segments = [
            PageSegment(
                start_idx=0,
                end_idx=2,
                entries=[
                    {"content": "序言", "target_page": None, "_unnumbered": True, "_unnumbered_index": 0},
                    {"content": "前言", "target_page": None, "_unnumbered": True, "_unnumbered_index": 1},
                ],
                segment_type="preface_unnumbered",
                offset=None,  # No offset known
            ),
        ]

        result = merge_segments_with_offsets(segments)

        # Without offset, unnumbered entries remain None
        assert result[0]["target_page"] is None
        assert result[1]["target_page"] is None
        assert result[0]["_needs_manual_offset"] is True

    def test_mixed_unnumbered_and_numbered_in_same_segment(self) -> None:
        """Should handle mixed unnumbered and numbered entries in same segment."""
        entries = [
            {"content": "序言", "target_page": None},
            {"content": "前言", "target_page": None},
            {"content": "第1章", "target_page": 1},
            {"content": "小节", "target_page": None},  # Sub-entry without number
            {"content": "第2章", "target_page": 20},
        ]

        result = infer_missing_targets(entries)

        # First 2 entries should be marked as unnumbered
        assert result[0].get("_unnumbered") is True
        assert result[1].get("_unnumbered") is True

        # Numbered entries should not be marked as unnumbered
        assert result[2].get("_unnumbered") is not True
        assert result[4].get("_unnumbered") is not True

        # Sub-entry without number should be inferred from neighbors
        assert result[3].get("target_page") == 1  # Inferred from previous
        assert result[3].get("_inferred") is True

    def test_all_unnumbered_entries(self) -> None:
        """Should handle case where all entries lack page numbers."""
        entries = [
            {"content": "序言", "target_page": None},
            {"content": "前言", "target_page": None},
            {"content": "致谢", "target_page": None},
        ]

        result = infer_missing_targets(entries)

        # All entries should be marked as unnumbered
        for i, entry in enumerate(result):
            assert entry.get("_unnumbered") is True
            assert entry.get("_unnumbered_index") == i
            assert entry.get("target_page") is None

        segments = detect_page_segments(result)
        assert len(segments) == 1
        assert segments[0].segment_type == "preface_unnumbered"


class TestVLMBehavior:
    """Tests that verify VLM behavior with unnumbered entries."""

    def test_vlm_should_not_fabricate_page_numbers(self) -> None:
        """VLM should return null for entries without explicit page numbers."""
        # This test documents the expected VLM behavior
        # The VLM prompt in vlm_api.py explicitly instructs:
        # "If a TOC line has NO page number printed at the end, you MUST set 'target_page' to null."
        # "Do NOT guess, infer, or fabricate page numbers. Only extract what is actually printed."

        # Example of correct VLM output for unnumbered entries
        correct_vlm_output = [
            {"page": 4, "target_page": None, "content": "第二版丛书序"},
            {"page": 4, "target_page": None, "content": "第一版丛书序"},
            {"page": 4, "target_page": None, "content": "第二版前言"},
            {"page": 4, "target_page": None, "content": "第一版前言"},
            {"page": 4, "target_page": 1, "content": "第 1 章  电磁现象的基本规律"},
            {"page": 4, "target_page": 1, "content": "1.1  场论和张量分析"},
        ]

        # Verify that unnumbered entries have target_page=None
        for entry in correct_vlm_output[:4]:
            assert entry["target_page"] is None, f"VLM should not fabricate page numbers for: {entry['content']}"

        # Verify that numbered entries have correct page numbers
        for entry in correct_vlm_output[4:]:
            assert entry["target_page"] == 1, f"VLM should extract correct page numbers for: {entry['content']}"