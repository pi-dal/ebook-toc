"""Command line interface for the ebook-toc tool."""

from __future__ import annotations

import argparse
import re
import sys
import time
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from rich.console import Console
from .progress import (
    ProgressReporter,
    TimingReport,
    create_progress,
    is_interactive,
    print_step_complete,
    timed_progress,
    timed_step,
)
from . import __version__

from .fingerprints import (
    compute_pdf_fingerprints,
    dominant_dimensions,
    build_canonical_map_for_dims,
)
from .pdf_writer import write_pdf_toc
from .vlm_api import (
    TOCExtractionError,
    fetch_document_json,
    _infer_page_offset,
    _infer_offsets_for_segments,
)
from .toc_parser import (
    deduplicate_entries,
    extract_toc_entries,
    filter_entries,
    infer_missing_targets,
    detect_page_segments,
    merge_segments_with_offsets,
    classify_entries,
)
from .utils import (
    dump_json,
    ensure_file,
    ensure_output_path,
    load_json,
    coerce_positive_int as _util_coerce_positive_int,
    download_to_temp as _util_download_to_temp,
)

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ebook-toc",
        description=(
            "Scan a PDF with an OpenAI-format VLM (default: SiliconFlow Qwen) "
            "and extract table-of-contents entries."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        # Use distribution version to avoid drift with pyproject.toml
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    parser._subparsers_action = subparsers  # type: ignore[attr-defined]

    help_parser = subparsers.add_parser(
        "help",
        help="Show help for the CLI or a specific command.",
    )
    help_parser.add_argument(
        "topic",
        type=str,
        nargs="?",
        default=None,
        help="Command name to show help for (e.g. 'scan').",
    )
    help_parser.set_defaults(func=_run_help)

    scan_parser = subparsers.add_parser(
        "scan",
        help=(
            "Upload or download a PDF, call an OpenAI-format VLM "
            "(default: SiliconFlow Qwen), and export TOC JSON."
        ),
    )
    scan_parser.add_argument(
        "pdf",
        type=Path,
        metavar="PDF",
        nargs="?",
        help="Path to the source PDF file.",
    )
    scan_parser.add_argument(
        "--api-key",
        "-k",
        required=True,
        help="VLM API token (OpenAI-format; default backend is SiliconFlow).",
    )
    scan_parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible API base URL (e.g. https://api.siliconflow.cn/v1 "
            "or https://api.openai.com/v1). Defaults to SiliconFlow."
        ),
    )
    scan_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "VLM model name in OpenAI format "
            "(default: Qwen/Qwen3-VL-32B-Instruct)."
        ),
    )
    scan_parser.add_argument(
        "--remote-url",
        type=str,
        default=None,
        help="Remote PDF URL to download before calling the VLM.",
    )
    scan_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to the TOC JSON file (defaults to output/json/<name>_toc.json).",
    )
    scan_parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Reserved for compatibility; ignored by SiliconFlow workflow.",
    )
    scan_parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="VLM request timeout in seconds (default: 600).",
    )
    scan_parser.add_argument(
        "--pages",
        type=int,
        default=10,
        help="Number of leading pages to scan (0 means scan entire document; default: 10).",
    )
    scan_parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Upper bound on pages to analyze when auto-expanding (default: 50).",
    )
    scan_parser.add_argument(
        "--step-pages",
        type=int,
        default=10,
        help="Page increment when auto-expanding the scan window (default: 10).",
    )
    scan_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of pages to send per VLM request (default: 10).",
    )
    scan_parser.add_argument(
        "--fuzzy-dedup",
        type=float,
        default=0.85,
        help=(
            "Fuzzy deduplication threshold in [0.0,1.0] "
            "(0.0 disables fuzzy matching; default: 0.85)."
        ),
    )
    scan_parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of concurrent VLM requests (default: 3).",
    )
    scan_parser.add_argument(
        "--save-json",
        dest="save_json",
        action="store_const",
        const=True,
        default=None,
        help="Save the detected TOC to a JSON file without prompting.",
    )
    scan_parser.add_argument(
        "--apply-toc",
        dest="apply_toc",
        action="store_const",
        const=True,
        default=None,
        help="Embed the detected TOC into the PDF without prompting.",
    )
    scan_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview TOC entries without writing JSON or modifying the PDF.",
    )
    scan_parser.add_argument(
        "--no-auto-expand",
        action="store_true",
        help="Disable automatic page expansion when no TOC entries are found.",
    )
    scan_parser.add_argument(
        "--filter-contains",
        type=str,
        default=None,
        help="Keep only entries whose content contains this substring (case-insensitive).",
    )
    scan_parser.add_argument(
        "--filter-regex",
        type=str,
        default=None,
        help="Keep only entries whose content matches this regular expression (case-insensitive).",
    )
    scan_parser.add_argument(
        "--goodnotes-clean",
        dest="goodnotes_clean",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Detect and remove non-dominant-size pages (e.g., GoodNotes insertions) before scanning; "
            "if omitted in interactive mode, the CLI will ask (default: No)."
        ),
    )
    scan_parser.set_defaults(func=_run_scan)

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply a saved TOC JSON to a PDF and embed bookmarks.",
    )
    apply_parser.add_argument(
        "pdf",
        type=Path,
        metavar="PDF",
        help="Path to the source PDF file.",
    )
    apply_parser.add_argument(
        "json",
        type=Path,
        metavar="JSON",
        help="Path to a TOC JSON file (as produced by the scan command).",
    )
    apply_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Destination PDF path (defaults to output/pdf/<name>_with_toc.pdf).",
    )
    apply_parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=None,
        help="VLM API token (optional; improves offset inference on apply).",
    )
    apply_parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible API base URL used for offset refinement "
            "(defaults to SiliconFlow when omitted)."
        ),
    )
    apply_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "VLM model name used for offset/verification "
            "(default: Qwen/Qwen3-VL-32B-Instruct)."
        ),
    )
    apply_parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="VLM request timeout in seconds (default: 600).",
    )
    apply_parser.add_argument(
        "--override-offset",
        type=int,
        default=None,
        help="Manually override printed-page offset (canonical = printed + offset).",
    )
    apply_parser.add_argument(
        "--verify-printed",
        action="store_true",
        help=(
            "After mapping, verify bookmarks by reading printed page numbers via VLM within a small window "
            "(requires --api-key). Adjust entries when mismatches are detected."
        ),
    )
    apply_parser.add_argument(
        "--verify-window",
        type=int,
        default=6,
        help="Window size for printed-page verification around predicted page (default: 6).",
    )
    apply_parser.add_argument(
        "--verify-max",
        type=int,
        default=80,
        help="Maximum number of entries to verify via VLM to limit API calls (default: 80).",
    )
    apply_parser.add_argument(
        "--goodnotes-clean",
        dest="goodnotes_clean",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Detect and remove non-dominant-size pages (e.g., GoodNotes insertions) to build a clean PDF; "
            "if omitted in interactive mode, the CLI will ask (default: No)."
        ),
    )
    apply_parser.set_defaults(func=_run_apply)

    return parser


def _run_scan(args: argparse.Namespace) -> None:
    report = TimingReport()
    sources_selected = sum(1 for option in (args.remote_url, args.pdf) if option)
    if sources_selected == 0:
        console.print("[red]Provide a PDF path or --remote-url.[/]")
        raise SystemExit(1)
    if sources_selected > 1:
        console.print("[red]Choose only one of PDF path or --remote-url.[/]")
        raise SystemExit(1)

    if args.remote_url:
        remote_url = args.remote_url
        pdf_path = None
    else:
        remote_url = None
        try:
            pdf_path = ensure_file(args.pdf)
        except FileNotFoundError as err:
            console.print(f"[red]{err}[/]")
            raise SystemExit(1) from err

    pattern = None
    if args.filter_regex:
        try:
            pattern = re.compile(args.filter_regex, re.IGNORECASE)
        except re.error as err:
            console.print(f"[red]Invalid regular expression for --filter-regex: {err}[/]")
            raise SystemExit(5) from err

    save_json = args.save_json
    apply_toc = args.apply_toc
    goodnotes_clean = args.goodnotes_clean

    if args.dry_run:
        save_json = False
        apply_toc = False
        if goodnotes_clean is None:
            goodnotes_clean = False
    else:
        if save_json is None:
            if sys.stdin.isatty():
                save_json = _prompt_yes_no("Save TOC as JSON?", default=True)
            else:
                save_json = True
        if apply_toc is None:
            if sys.stdin.isatty():
                apply_toc = _prompt_yes_no("Embed TOC into the PDF?", default=False)
            else:
                apply_toc = False
        if goodnotes_clean is None:
            if sys.stdin.isatty():
                goodnotes_clean = _prompt_yes_no(
                    "Clean GoodNotes insertions before scanning?", default=False
                )
            else:
                goodnotes_clean = False

    try:
        # Optional GoodNotes cleaning for scan: strip non-dominant-size pages before calling VLM
        original_pdf_for_output: Path | None = pdf_path
        temp_download: Path | None = None
        clean_pdf_path: Path | None = None
        clean_map: dict[int, int] = {}

        if goodnotes_clean:
            # Ensure we have a local copy to clean when remote URL is used
            if remote_url and not pdf_path:
                try:
                    temp_download = _download_to_temp(remote_url)
                    original_pdf_for_output = temp_download
                except requests.RequestException as exc:
                    console.print(f"[red]Failed to download remote PDF: {exc}[/]")
                    raise SystemExit(3) from exc
            else:
                original_pdf_for_output = pdf_path

            if original_pdf_for_output is None:
                console.print("[red]Unable to resolve source PDF for GoodNotes cleaning.[/]")
                raise SystemExit(1)

            try:
                if is_interactive():
                    with create_progress("Computing fingerprints for GoodNotes...", total=None) as (
                        progress,
                        task_id,
                    ):
                        reporter = ProgressReporter(progress, task_id)

                        def _fp_cb(done: int, total: int) -> None:
                            reporter(
                                done,
                                total,
                                f"Computing fingerprints... {done}/{total} pages",
                            )

                        fps0, _ = compute_pdf_fingerprints(
                            original_pdf_for_output,
                            progress_callback=_fp_cb,
                        )
                else:
                    fps0, _ = compute_pdf_fingerprints(original_pdf_for_output)
            except Exception:
                fps0 = []
            dims0 = dominant_dimensions(fps0) if fps0 else None
            if dims0:
                keep_indices, removed_indices = _detect_goodnotes_indices_from_fps(fps0, dims0)
            else:
                keep_indices, removed_indices = [], []

            if removed_indices:
                console.print(
                    f"[cyan]Detected {len(removed_indices)} non-dominant pages; scanning a clean copy...[/]"
                )
                clean_pdf_path, clean_map = _build_clean_pdf(original_pdf_for_output, keep_indices)
                scan_pdf = clean_pdf_path
                scan_remote = None
            else:
                scan_pdf = original_pdf_for_output
                scan_remote = None
        else:
            scan_pdf = pdf_path
            scan_remote = remote_url

        if scan_remote:
            console.print(
                f"🌐 Downloading remote PDF and extracting TOC (batch size {args.batch_size})..."
            )
        elif args.pages > 0:
            console.print(
                f"📖 Scanning first {args.pages} pages (batch size {args.batch_size}) to detect TOC..."
            )
        else:
            console.print(
                f"📖 Scanning entire document to detect TOC (batch size {args.batch_size})..."
            )

        scan_start = time.perf_counter()
        final_entries, used_limit, page_offset, fingerprints = _scan_with_adaptive_pages(
            api_key=args.api_key,
            pdf_path=scan_pdf,
            remote_url=scan_remote,
            initial_limit=args.pages,
            max_pages=args.max_pages,
            step=args.step_pages,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            auto_expand=not args.no_auto_expand,
            contains=args.filter_contains,
            pattern=pattern,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            api_base=args.api_base,
            model=args.model,
            fuzzy_threshold=args.fuzzy_dedup if args.fuzzy_dedup and args.fuzzy_dedup > 0 else None,
        )
        scan_elapsed = time.perf_counter() - scan_start
        scanned_pages = "all" if used_limit == 0 else str(used_limit)
        report.add(
            "VLM scanning",
            scan_elapsed,
            f"{scanned_pages} pages, {len(final_entries)} entries",
        )
    except TOCExtractionError as err:
        console.print(f"[red]{err}[/]")
        raise SystemExit(2) from err
    except requests.RequestException as err:
        console.print(f"[red]Network error while calling the VLM API: {err}[/]")
        raise SystemExit(3) from err
    finally:
        # Clean up temporary clean PDF and downloaded file if created
        try:
            if 'clean_pdf_path' in locals() and clean_pdf_path:
                Path(clean_pdf_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if 'temp_download' in locals() and temp_download:
                Path(temp_download).unlink(missing_ok=True)
        except Exception:
            pass

    pages_label = "全部" if used_limit == 0 else used_limit
    console.print(
        f"扫描完成，共 {len(final_entries)} 条目录 (扫描页数: {pages_label})"
    )
    # Additional English timing summary for power users
    print_step_complete(
        "Scan completed",
        scan_elapsed,
        f"scanned {pages_label} pages, found {len(final_entries)} entries",
    )
    if page_offset is not None:
        console.print(
            f"[cyan]Detected printed-page offset: {page_offset:+d} (PDF page 1 corresponds to printed page {1 - page_offset}).[/]"
        )

    # Prefer naming from the original source (not the clean copy)
    try:
        base_stem_source = original_pdf_for_output if 'original_pdf_for_output' in locals() and original_pdf_for_output else pdf_path
    except NameError:
        base_stem_source = pdf_path
    base_stem = _derive_output_stem(base_stem_source, remote_url)
    json_output_default = Path("output/json") / f"{base_stem}_toc.json"
    pdf_output_default = Path("output/pdf") / f"{base_stem}_with_toc.pdf"
    json_output_path = (
        args.output if args.output is not None else json_output_default
    )

    if args.dry_run:
        _print_toc_preview(final_entries, page_offset)
        console.print("[yellow]Dry-run: 未写入任何文件。[/]")
        return

    if save_json:
        # Detect and persist page-numbering segments (for multi-offset books).
        segments = detect_page_segments(final_entries)
        segments_data: list[dict[str, Any]] = []
        for seg in segments:
            segments_data.append(
                {
                    "start_idx": seg.start_idx,
                    "end_idx": seg.end_idx,
                    "segment_type": seg.segment_type,
                    # Offset is typically inferred at apply-time; keep here
                    # for forward-compatibility when it is known.
                    "offset": seg.offset,
                }
            )

        # Save comprehensive fingerprints for the original source when possible
        src_for_fps = base_stem_source
        if src_for_fps is not None:
            try:
                full_fps, _ = compute_pdf_fingerprints(src_for_fps)
            except Exception:
                full_fps = fingerprints
        else:
            full_fps = fingerprints

        dims0 = dominant_dimensions(full_fps) if full_fps else None
        page_map0 = build_canonical_map_for_dims(full_fps, dims0) if dims0 else {}

        payload = {
            "toc": final_entries,
            "page_offset": page_offset,
            "segments": segments_data,
            "fingerprints": full_fps,
            "page_map": page_map0,
        }
        if goodnotes_clean:
            try:
                if clean_map:
                    payload["clean_map"] = clean_map
            except NameError:
                pass

        json_path = ensure_output_path(json_output_path)
        dump_json(payload, json_path)
        console.print(f"[green]JSON 已保存至[/] {json_path}")
    else:
        console.print("[yellow]未保存 JSON，以下为输出结果：[/]")
        console.print_json(data={"toc": final_entries, "page_offset": page_offset})

    if not final_entries:
        if apply_toc:
            console.print("[yellow]No TOC entries detected; skipping PDF update.[/]")
        apply_toc = False

    if apply_toc:
        if pdf_path is None:
            console.print("[yellow]远程 PDF 未下载，无法写入目录。[/]")
        else:
            # Build canonical page map for current PDF based on dominant dimensions
            try:
                if is_interactive():
                    with timed_progress(
                        "Computing fingerprints for apply...",
                        total=None,
                        report=report,
                        step_name="Computing fingerprints (apply)",
                    ) as (progress, task_id):
                        reporter = ProgressReporter(progress, task_id)

                        def _fp_cb(done: int, total: int) -> None:
                            reporter(
                                done,
                                total,
                                f"Computing fingerprints... {done}/{total} pages",
                            )

                        current_fps, page_count = compute_pdf_fingerprints(
                            pdf_path,
                            progress_callback=_fp_cb,
                        )
                else:
                    with timed_step("Computing fingerprints (apply)", report):
                        current_fps, page_count = compute_pdf_fingerprints(pdf_path)
            except Exception:
                current_fps, page_count = [], _get_pdf_page_count(pdf_path)

            dims = dominant_dimensions(current_fps) if current_fps else None
            canonical_map = (
                build_canonical_map_for_dims(current_fps, dims) if dims else {}
            )

            with timed_step("Mapping TOC entries (scan/apply)", report):
                refined_offset = _refine_offset_with_mapping(
                    final_entries, canonical_map, page_offset
                )
                if refined_offset != page_offset:
                    console.print(
                        f"[cyan]Refined printed-page offset: {refined_offset:+d} (was {page_offset}).[/]"
                    )
                resolved_entries = _apply_page_mapping(
                    final_entries, canonical_map, refined_offset, page_count
                )

            pdf_output_path = pdf_output_default
            try:
                if is_interactive():
                    total_entries = len(resolved_entries)
                    with timed_progress(
                        "Writing PDF bookmarks...",
                        total=total_entries or None,
                        report=report,
                        step_name="Writing PDF bookmarks",
                    ) as (progress, task_id):
                        reporter = ProgressReporter(progress, task_id)

                        def _write_progress(done: int, total: int) -> None:
                            reporter(
                                done,
                                total,
                                f"Writing PDF bookmarks... {done}/{total} entries",
                            )

                        result = write_pdf_toc(
                            pdf_path,
                            resolved_entries,
                            pdf_output_path,
                            page_offset=None,
                            progress_callback=_write_progress,
                        )
                else:
                    with timed_step("Writing PDF bookmarks", report):
                        result = write_pdf_toc(
                            pdf_path,
                            resolved_entries,
                            pdf_output_path,
                            page_offset=None,
                        )
            except Exception as err:
                console.print(f"[red]写入 PDF 目录失败: {err}[/]")
                raise SystemExit(6) from err

            console.print(
                f"[green]PDF 目录已写入 {result.added} 条[/] -> {result.output_path}"
            )
            if result.skipped:
                console.print("[yellow]以下条目因异常被跳过：[/]")
                for reason in result.skipped:
                    console.print(f"  - {reason}")
            # Print timing summary for scan+apply when apply_toc is enabled.
            report.print_summary()
    else:
        console.print("[cyan]未写入 PDF 目录。[/]")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(0)

    args.func(args)


def _derive_output_stem(pdf_path: Path | None, remote_url: str | None) -> str:
    if pdf_path:
        return pdf_path.stem
    if remote_url:
        parsed = urlparse(remote_url)
        candidate = Path(parsed.path).stem
        return candidate or "remote_document"
    return "toc"


def _download_to_temp(url: str) -> Path:
    # Delegate to shared util to avoid duplication
    return _util_download_to_temp(url, suffix=".pdf", prefix="scan-")


def _print_toc_preview(entries: list[dict[str, Any]], page_offset: int | None) -> None:
    if not entries:
        console.print("[yellow]未检测到目录条目。[/]")
        return

    console.print("[b]TOC Preview[/]")
    current_segment: str | None = None
    for entry in entries:
        # Optional segment metadata (present after multi-segment mapping).
        segment_type = entry.get("_segment_type")
        if segment_type and segment_type != current_segment:
            current_segment = segment_type
            segment_offset = entry.get("_segment_offset", "?")
            console.print(
                f"\n[dim]── {segment_type} (offset: {segment_offset}) ──[/]"
            )

        page = entry.get("page")
        target = entry.get("target_page")
        original_target = entry.get("_original_target_page", target)
        title = (entry.get("content") or "").strip()

        display_target = original_target if original_target is not None else "-"
        resolved = None
        if target is not None and original_target is not None and target != original_target:
            resolved = target
        elif page_offset is not None and original_target is not None:
            # Fallback preview when per-entry offsets are not yet applied.
            resolved = original_target + page_offset

        detail = f" → PDF {resolved}" if resolved is not None else ""
        console.print(f"  - p. {display_target}{detail} : {title}")


def _prompt_yes_no(question: str, default: bool) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    default_label = "Yes" if default else "No"
    prompt = f"{question}{suffix}(default: {default_label}) "
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        console.print("[yellow]Please respond with 'y' or 'n'.[/]")


def _coerce_positive_int(value: Any) -> int | None:
    return _util_coerce_positive_int(value)


def _apply_page_mapping(
    entries: list[dict[str, Any]],
    mapping: dict[int, int],
    page_offset: int | None,
    page_count: int,
    resolved_override: dict[int, int | None] | None = None,
) -> list[dict[str, Any]]:
    resolved_override = resolved_override or {}
    resolved_entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if idx in resolved_override and resolved_override[idx] is not None:
            resolved_page = int(resolved_override[idx])  # type: ignore[arg-type]
            if page_count:
                resolved_page = max(1, min(resolved_page, page_count))
            adjusted = dict(entry)
            adjusted["target_page"] = resolved_page
            adjusted["_resolved_by"] = "direct_location"
            resolved_entries.append(adjusted)
            continue

        base_page = _coerce_positive_int(entry.get("target_page"))
        if base_page is None:
            base_page = _coerce_positive_int(entry.get("page"))

        resolved_page = None
        # Prefer pure canonical mapping first (robust against wrong offset),
        # then try offset-adjusted canonical index, then numeric fallback
        if base_page is not None:
            if mapping:
                resolved_page = mapping.get(base_page)
            if resolved_page is None and page_offset is not None:
                canonical_idx = base_page + page_offset
                if mapping:
                    resolved_page = mapping.get(canonical_idx)
                if resolved_page is None:
                    resolved_page = canonical_idx
        if resolved_page is None:
            resolved_page = base_page

        if resolved_page is not None:
            resolved_page = max(1, min(resolved_page, page_count)) if page_count else resolved_page
            adjusted = dict(entry)
            adjusted["target_page"] = resolved_page
            resolved_entries.append(adjusted)
        else:
            resolved_entries.append(entry)

    return resolved_entries


def _get_pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        return 0

    try:
        with fitz.open(pdf_path) as doc:  # type: ignore[attr-defined]
            return doc.page_count
    except Exception:
        return 0


def _refine_offset_with_mapping(
    entries: list[dict[str, Any]],
    mapping: dict[int, int],
    initial_offset: int | None,
    window: int = 40,
) -> int | None:
    """Return an offset that maximizes mapping hits for target_page + offset.

    This is robust when VLM-estimated offset is off; we search a neighborhood
    around the initial estimate (or a default range) and choose the offset that
    yields the highest number of canonical-index hits in mapping.
    """
    targets: list[int] = []
    for e in entries:
        t = _coerce_positive_int(e.get("target_page"))
        if t is None:
            t = _coerce_positive_int(e.get("page"))
        if t is not None:
            targets.append(t)
    if not targets or not mapping:
        return initial_offset

    if initial_offset is None:
        center = 0
    else:
        center = int(initial_offset)

    best_off = initial_offset
    best_score = -1
    # Search window around center, also try a few far guesses if center is bad
    candidates = list(range(center - window, center + window + 1))
    if 0 not in candidates:
        candidates.append(0)
    for off in candidates:
        hits = 0
        for t in targets:
            if mapping.get(t + off) is not None:
                hits += 1
        if hits > best_score:
            best_score = hits
            best_off = off
    return best_off


def _run_apply(args: argparse.Namespace) -> None:
    report = TimingReport()
    try:
        pdf_path = ensure_file(args.pdf)
    except FileNotFoundError as err:
        console.print(f"[red]{err}[/]")
        raise SystemExit(1) from err

    try:
        json_path = ensure_file(args.json)
    except FileNotFoundError as err:
        console.print(f"[red]{err}[/]")
        raise SystemExit(1) from err

    # Phase 1: load JSON
    try:
        with timed_step("Loading JSON", report):
            raw_data = load_json(json_path)
    except OSError as err:
        console.print(f"[red]Unable to read JSON: {err}[/]")
        raise SystemExit(4) from err

    fingerprints: list[dict[str, Any]] = []
    saved_clean_map: dict[int, int] = {}
    if isinstance(raw_data, dict) and "toc" in raw_data:
        entries = extract_toc_entries(raw_data.get("toc"))
        entries = infer_missing_targets(entries)
        page_offset = _coerce_positive_int(raw_data.get("page_offset"))
        stored_fps = raw_data.get("fingerprints")
        if isinstance(stored_fps, list):
            fingerprints = stored_fps
        raw_clean_map = raw_data.get("clean_map")
        if isinstance(raw_clean_map, dict):
            tmp: dict[int, int] = {}
            for k, v in raw_clean_map.items():
                try:
                    ck = int(k)
                    cv = int(v)
                except (TypeError, ValueError):
                    continue
                if ck > 0 and cv > 0:
                    tmp[ck] = cv
            saved_clean_map = tmp
    else:
        entries = extract_toc_entries(raw_data)
        entries = infer_missing_targets(entries)
        page_offset = None

    if not entries:
        console.print("[yellow]No TOC entries found in JSON. Nothing to apply.[/]")
        return

    # Classify entries up-front so we can optionally resolve unnumbered
    # and suspicious ones directly in the PDF.
    try:
        unnumbered_idx, suspicious_idx, normal_idx = classify_entries(entries)
    except Exception:
        unnumbered_idx, suspicious_idx, normal_idx = [], [], list(range(len(entries)))

    output_path = (
        args.output
        if args.output is not None
        else Path("output/pdf") / f"{pdf_path.stem}_with_toc.pdf"
    )

    page_count = 0
    # Interactive prompt for GoodNotes clean (default No) if not provided
    if args.goodnotes_clean is None:
        if sys.stdin.isatty():
            args.goodnotes_clean = _prompt_yes_no(
                "Clean GoodNotes insertions before applying bookmarks?", default=False
            )
        else:
            args.goodnotes_clean = False

    # Phase 2: compute fingerprints for current PDF
    try:
        if args.goodnotes_clean:
            # Preserve behaviour for tests that monkeypatch cli.compute_pdf_fingerprints
            with timed_step("Computing fingerprints", report):
                current_fps, page_count = compute_pdf_fingerprints(pdf_path)
        else:
            if is_interactive():
                with timed_progress(
                    "Computing fingerprints...",
                    total=None,
                    report=report,
                    step_name="Computing fingerprints",
                ) as (progress, task_id):
                    reporter = ProgressReporter(progress, task_id)

                    def _fp_cb(done: int, total: int) -> None:
                        reporter(
                            done,
                            total,
                            f"Computing fingerprints... {done}/{total} pages",
                        )

                    current_fps, page_count = compute_pdf_fingerprints(
                        pdf_path,
                        progress_callback=_fp_cb,
                    )
            else:
                with timed_step("Computing fingerprints", report):
                    current_fps, page_count = compute_pdf_fingerprints(pdf_path)
    except Exception:
        current_fps, page_count = [], _get_pdf_page_count(pdf_path)

    # Determine dominant page dimensions from stored fingerprints; fallback to current
    dims = dominant_dimensions(fingerprints) if fingerprints else None
    if dims is None and current_fps:
        dims = dominant_dimensions(current_fps)

    # Start with override if provided, else JSON value
    refined_offset = args.override_offset if hasattr(args, 'override_offset') and args.override_offset is not None else page_offset

    # Optional: VLM-based refinement if API key supplied
    if getattr(args, "api_key", None):
        with timed_step("Refining page offset (VLM)", report):
            try:
                infer_pdf_path = pdf_path
                temp_clean_for_infer: Path | None = None

                if args.goodnotes_clean and dims and current_fps:
                    keep_indices, removed_indices = _detect_goodnotes_indices_from_fps(
                        current_fps,
                        dims,
                    )
                    if removed_indices and keep_indices:
                        temp_clean_for_infer, _ = _build_clean_pdf(pdf_path, keep_indices)
                        infer_pdf_path = temp_clean_for_infer

                try:
                    vlm_offset = _infer_page_offset(
                        infer_pdf_path,
                        entries,
                        args.api_key,
                        args.timeout,
                        current_fps if infer_pdf_path == pdf_path else None,
                        api_base=args.api_base,
                        model=args.model,
                    )
                    if vlm_offset is not None:
                        refined_offset = vlm_offset
                        console.print(
                            f"[cyan]Refined printed-page offset (VLM): {refined_offset:+d} (was {page_offset}).[/]"
                        )
                finally:
                    if temp_clean_for_infer is not None:
                        try:
                            temp_clean_for_infer.unlink(missing_ok=True)
                        except Exception:
                            pass
            except TOCExtractionError:
                # If VLM-based offset refinement fails, keep the previous offset.
                pass

    # If GoodNotes cleaning requested, build clean PDF and resolve via clean->original mapping
    if args.goodnotes_clean:
        # Prefer using saved clean_map from JSON for maximum stability
        if saved_clean_map and refined_offset is not None:
            console.print("[cyan]Using saved clean_map from JSON for GoodNotes alignment.[/]")
            resolved_entries = []
            preview_rows: list[str] = []
            for e in entries:
                base_page = _coerce_positive_int(e.get("target_page"))
                if base_page is None:
                    base_page = _coerce_positive_int(e.get("page"))
                if base_page is None:
                    resolved_entries.append(e)
                    continue
                canonical_idx = base_page + refined_offset
                resolved_orig = saved_clean_map.get(canonical_idx)
                if resolved_orig is None:
                    # Fallback: clamp numeric adjustment
                    candidate = canonical_idx
                    if page_count:
                        candidate = max(1, min(candidate, page_count))
                    resolved_orig = candidate
                adjusted = dict(e)
                adjusted["target_page"] = resolved_orig
                resolved_entries.append(adjusted)
                if len(preview_rows) < 12:
                    title = str(e.get("content") or "").strip()
                    preview_rows.append(
                        f"tp={base_page} canon={canonical_idx} -> orig={resolved_orig} | {title[:40]}"
                    )
            if preview_rows:
                console.print("[dim]Preview mapping (first 12):\n" + "\n".join("  " + r for r in preview_rows))
        else:
            resolved_entries = _apply_with_goodnotes_clean(
                pdf_path,
                entries,
                refined_offset,
                dims,
            )
        # Optional VLM-based printed-number verification pass
        if getattr(args, 'verify_printed', False):
            try:
                resolved_entries = _adjust_entries_by_printed(
                    pdf_path,
                    resolved_entries,
                    getattr(args, 'api_key', None),
                    getattr(args, 'timeout', 600),
                    api_base=args.api_base,
                    model=args.model,
                    window=max(1, int(getattr(args, 'verify_window', 6))),
                    max_checks=max(10, int(getattr(args, 'verify_max', 80))),
                )
            except Exception:
                pass
    else:
        # Non-GoodNotes path: support multi-segment page numbering.
        segments = detect_page_segments(entries)

        # Attempt to resolve unnumbered/suspicious entries directly when possible.
        resolved_unnumbered: dict[int, int | None] = {}
        if (unnumbered_idx or suspicious_idx) and pdf_path is not None:
            from .vlm_api import resolve_unnumbered_entries

            try:
                with timed_step("Locating unnumbered/suspicious entries", report):
                    resolved_unnumbered = resolve_unnumbered_entries(
                        pdf_path,
                        entries,
                        unnumbered_idx,
                        suspicious_idx,
                        first_body_pdf_page=None,
                        api_key=getattr(args, "api_key", None),
                        timeout=args.timeout,
                        api_base=getattr(args, "api_base", None),
                        model=getattr(args, "model", None),
                    )
                located = sum(1 for v in resolved_unnumbered.values() if v is not None)
                total_special = len(resolved_unnumbered)
                if total_special:
                    console.print(
                        f"[cyan]Directly located {located}/{total_special} unnumbered/suspicious entries.[/]"
                    )
            except Exception:
                resolved_unnumbered = {}

        if len(segments) <= 1:
            # Single segment: retain existing canonical-map refinement logic.
            canonical_map: dict[int, int] = {}
            if dims and current_fps:
                canonical_map = build_canonical_map_for_dims(current_fps, dims)
            refined_offset = _refine_offset_with_mapping(
                entries, canonical_map, refined_offset
            )
            if refined_offset != page_offset and not getattr(args, "override_offset", None):
                console.print(
                    f"[cyan]Refined printed-page offset: {refined_offset:+d} (was {page_offset}).[/]"
                )
            with timed_step("Mapping TOC entries", report):
                resolved_entries = _apply_page_mapping(
                    entries,
                    canonical_map,
                    refined_offset,
                    page_count,
                    resolved_override=resolved_unnumbered,
                )
        else:
            console.print(f"[cyan]Detected {len(segments)} page-numbering segments.[/]")

            if getattr(args, "api_key", None):
                # Use VLM to infer per-segment offsets when an API key is available.
                with timed_step("Inferring segment offsets (VLM)", report):
                    _infer_offsets_for_segments(
                        pdf_path,
                        segments,
                        args.api_key,
                        args.timeout,
                        api_base=args.api_base,
                        model=args.model,
                    )
                    for seg in segments:
                        off = getattr(seg, "offset", None)
                        offset_str = f"{off:+d}" if isinstance(off, int) else "unknown"
                        console.print(
                            f"  - {seg.segment_type}: {len(seg.entries)} entries, offset={offset_str}"
                        )
            else:
                # Without an API key, prefer any segment offsets stored in the
                # JSON payload; otherwise fall back to a single refined offset.
                saved_segments = (
                    raw_data.get("segments", []) if isinstance(raw_data, dict) else []
                )
                if saved_segments:
                    for idx, seg in enumerate(segments):
                        if idx < len(saved_segments):
                            stored = saved_segments[idx]
                            try:
                                stored_off = int(stored.get("offset"))  # type: ignore[arg-type]
                            except (TypeError, ValueError):
                                stored_off = None
                            if stored_off is not None:
                                seg.offset = stored_off
                else:
                    for seg in segments:
                        seg.offset = refined_offset

            with timed_step("Mapping TOC entries (multi-segment)", report):
                resolved_entries = merge_segments_with_offsets(
                    segments,
                    page_count,
                    resolved_override=resolved_unnumbered,
                )
                # Clamp resolved target_page values into the valid page range.
                if page_count > 0:
                    for entry in resolved_entries:
                        tp = entry.get("target_page")
                        try:
                            if tp is not None:
                                entry["target_page"] = max(1, min(int(tp), page_count))
                        except (TypeError, ValueError):
                            continue

    try:
        if is_interactive():
            total_entries = len(resolved_entries)
            with timed_progress(
                "Writing PDF bookmarks...",
                total=total_entries or None,
                report=report,
                step_name="Writing PDF bookmarks",
            ) as (progress, task_id):
                reporter = ProgressReporter(progress, task_id)

                def _write_progress(done: int, total: int) -> None:
                    reporter(
                        done,
                        total,
                        f"Writing PDF bookmarks... {done}/{total} entries",
                    )

                result = write_pdf_toc(
                    pdf_path,
                    resolved_entries,
                    output_path,
                    page_offset=None,
                    progress_callback=_write_progress,
                )
        else:
            with timed_step("Writing PDF bookmarks", report):
                result = write_pdf_toc(
                    pdf_path,
                    resolved_entries,
                    output_path,
                    page_offset=None,
                )
    except Exception as err:
        console.print(f"[red]Failed to write PDF bookmarks: {err}[/]")
        raise SystemExit(6) from err

    console.print(
        f"[green]PDF bookmarks written: {result.added} entries[/] -> {result.output_path}"
    )
    if refined_offset is not None:
        console.print(
            f"[cyan]Applied printed-page offset: {refined_offset:+d} (PDF page 1 corresponds to printed page {1 - refined_offset}).[/]"
        )
    if result.skipped:
        console.print("[yellow]Skipped entries:[/]")
        for reason in result.skipped:
            console.print(f"  - {reason}")

    # Print per-phase timing summary for apply
    report.print_summary()


def _apply_with_goodnotes_clean(
    pdf_path: Path,
    entries: list[dict[str, Any]],
    initial_offset: int | None,
    baseline_dims: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    """Resolve TOC entries by removing non-dominant-size pages (GoodNotes insertions).

    Steps:
    - Detect keep (dominant-size) pages and removed pages.
    - Build a temporary clean PDF with only keep pages, record clean->original mapping.
    - Build canonical mapping on the clean PDF and refine offset using mapping hits.
    - Map resolved pages from clean back to original via the recorded mapping.
    - Clean up temporary PDF; return resolved entries for the original PDF.
    """
    try:
        current_fps, original_page_count = compute_pdf_fingerprints(pdf_path)
    except Exception:
        # Fallback: resolve using numeric pages and provided offset only
        page_count = _get_pdf_page_count(pdf_path)
        return _apply_page_mapping(entries, {}, initial_offset, page_count)

    dims = baseline_dims if baseline_dims else dominant_dimensions(current_fps)
    if not dims:
        # Unable to establish dominant size; fallback using numeric mapping with optional refinement
        dims2 = dominant_dimensions(current_fps) if current_fps else None
        if not dims2:
            return _apply_page_mapping(entries, {}, initial_offset, original_page_count)
        canonical_map = build_canonical_map_for_dims(current_fps, dims2)
        refined = _refine_offset_with_mapping(entries, canonical_map, initial_offset)
        return _apply_page_mapping(entries, canonical_map, refined, original_page_count)

    keep_indices, removed_indices = _detect_goodnotes_indices_from_fps(current_fps, dims)
    if not removed_indices:
        # Nothing to clean; prefer JSON/override offset without refinement
        canonical_map = build_canonical_map_for_dims(current_fps, dims)
        refined = initial_offset
        return _apply_page_mapping(entries, canonical_map, refined, original_page_count)

    console.print(
        f"[cyan]Detected {len(removed_indices)} non-dominant pages; building clean PDF for mapping...[/]"
    )

    clean_pdf_path, clean_to_original = _build_clean_pdf(pdf_path, keep_indices)
    try:
        try:
            clean_fps, _ = compute_pdf_fingerprints(clean_pdf_path)
        except Exception:
            clean_fps, _ = [], 0

        clean_dims = dims if dims else dominant_dimensions(clean_fps)
        canonical_map_clean: dict[int, int] = {}
        if clean_dims and clean_fps:
            canonical_map_clean = build_canonical_map_for_dims(clean_fps, clean_dims)

        # In GoodNotes-clean mode, prefer JSON/override offset without additional refinement,
        # as the clean PDF already excludes inserts and canonical_map_clean is dense.
        refined = initial_offset

        # Resolve via clean -> original
        resolved_entries: list[dict[str, Any]] = []
        for entry in entries:
            base_page = _coerce_positive_int(entry.get("target_page"))
            if base_page is None:
                base_page = _coerce_positive_int(entry.get("page"))

            resolved_clean = None
            if base_page is not None:
                if canonical_map_clean:
                    if refined is not None:
                        canonical_idx = base_page + refined
                        resolved_clean = canonical_map_clean.get(canonical_idx)
                    if resolved_clean is None:
                        resolved_clean = canonical_map_clean.get(base_page)
                if resolved_clean is None and refined is not None:
                    resolved_clean = base_page + refined
                if resolved_clean is None:
                    resolved_clean = base_page

            if resolved_clean is not None:
                resolved_orig = clean_to_original.get(resolved_clean, resolved_clean)
                resolved_orig = max(1, min(resolved_orig, original_page_count)) if original_page_count else resolved_orig
                adjusted = dict(entry)
                adjusted["target_page"] = resolved_orig
                resolved_entries.append(adjusted)
            else:
                resolved_entries.append(entry)

        return resolved_entries
    finally:
        try:
            Path(clean_pdf_path).unlink(missing_ok=True)
        except Exception:
            pass


def _detect_goodnotes_indices_from_fps(
    fps: list[dict[str, Any]], dims: tuple[int, int]
) -> tuple[list[int], list[int]]:
    keep: list[int] = []
    removed: list[int] = []
    w0, h0 = int(dims[0]), int(dims[1])
    for idx, fp in enumerate(fps, start=1):
        w = int(fp.get("width") or 0)
        h = int(fp.get("height") or 0)
        if (w, h) == (w0, h0):
            keep.append(idx)
        else:
            removed.append(idx)
    return keep, removed


def _build_clean_pdf(
    pdf_path: Path, keep_indices: list[int]
) -> tuple[Path, dict[int, int]]:
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF (fitz) is required for GoodNotes cleaning") from exc

    src = fitz.open(str(pdf_path))  # type: ignore[attr-defined]
    try:
        clean = fitz.open()  # type: ignore[attr-defined]
        clean_to_original: dict[int, int] = {}
        total = len(keep_indices)

        if not keep_indices:
            # Nothing to keep; return an empty PDF.
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="clean-")
            tmp.close()
            clean.save(tmp.name)
            clean.close()
            return Path(tmp.name), clean_to_original

        # Merge consecutive indices into ranges so we can insert in batches.
        sorted_indices = sorted(keep_indices)
        ranges: list[tuple[int, int]] = []
        start = end = sorted_indices[0]
        for idx in sorted_indices[1:]:
            if idx == end + 1:
                end = idx
            else:
                ranges.append((start, end))
                start = end = idx
        ranges.append((start, end))

        clean_idx = 0

        # Use a progress bar in both interactive and non-interactive modes.
        # When stdout is not a TTY, create_progress internally disables output.
        with create_progress(
            "Building clean PDF...", total=total or None
        ) as (progress, task_id):
            processed = 0
            for range_start, range_end in ranges:
                try:
                    # Batch insert a contiguous range of pages.
                    clean.insert_pdf(
                        src,
                        from_page=range_start - 1,
                        to_page=range_end - 1,
                    )
                    for orig_idx in range(range_start, range_end + 1):
                        clean_idx += 1
                        clean_to_original[clean_idx] = orig_idx
                        processed += 1
                        progress.update(
                            task_id,
                            completed=processed,
                            total=total or processed,
                            description=(
                                f"Building clean PDF... {processed}/{total} pages"
                                if total
                                else "Building clean PDF..."
                            ),
                        )
                except Exception:
                    # Fallback to per-page insertion when bulk insert fails.
                    for orig_idx in range(range_start, range_end + 1):
                        try:
                            clean.insert_pdf(
                                src,
                                from_page=orig_idx - 1,
                                to_page=orig_idx - 1,
                            )
                            clean_idx += 1
                            clean_to_original[clean_idx] = orig_idx
                            processed += 1
                            progress.update(
                                task_id,
                                completed=processed,
                                total=total or processed,
                                description=(
                                    f"Building clean PDF... {processed}/{total} pages"
                                    if total
                                    else "Building clean PDF..."
                                ),
                            )
                        except Exception:
                            continue

        # Save to a temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="clean-")
        tmp.close()
        clean.save(tmp.name)
        clean.close()
        return Path(tmp.name), clean_to_original
    finally:
        src.close()


## Removed unused text-based refinement helper to reduce maintenance.


def _adjust_entries_by_printed(
    pdf_path: Path,
    entries: list[dict[str, Any]],
    api_key: str | None,
    timeout: int,
    api_base: str | None = None,
    model: str | None = None,
    window: int = 6,
    max_checks: int = 80,
) -> list[dict[str, Any]]:
    if not api_key:
        return entries
    try:
        from .vlm_api import _get_printed_page_number
    except Exception:
        return entries

    # Prefer prominent headings to verify (reduce API calls)
    keywords = ("章", "节", "§", "绪论", "引言", "习题", "复习")
    candidates: list[int] = []
    others: list[int] = []
    for idx, e in enumerate(entries):
        title = str(e.get("content") or "")
        (candidates if any(k in title for k in keywords) else others).append(idx)
    order = candidates + others

    checks = 0
    changed = 0
    adjusted = entries[:]

    if is_interactive():
        with create_progress(
            "Verifying printed pages...", total=max_checks
        ) as (progress, task_id):
            for idx in order:
                if checks >= max_checks:
                    break
                e = adjusted[idx]
                guess = _coerce_positive_int(e.get("target_page"))
                if not guess:
                    continue
                # Conservative: only adjust on exact printed-page match, nearest to guess
                exact_matches: list[int] = []
                for p in range(max(1, guess - window), guess + window + 1):
                    pn = _get_printed_page_number(
                        pdf_path,
                        p - 1,
                        api_key,
                        timeout,
                        api_base=api_base,
                        model=model,
                    )
                    if pn is None:
                        continue
                    if int(pn) == int(e.get("target_page") or 0):
                        exact_matches.append(p)
                checks += 1
                if exact_matches:
                    # choose the closest exact match to current guess
                    best_p = min(exact_matches, key=lambda p: abs(p - guess))
                    if best_p != guess:
                        new_e = dict(e)
                        new_e["target_page"] = best_p
                        adjusted[idx] = new_e
                        changed += 1
                progress.update(
                    task_id,
                    completed=checks,
                    total=max_checks,
                    description=(
                        f"Verifying printed pages... {checks}/{max_checks} | "
                        f"Adjusted: {changed}"
                    ),
                )
    else:
        for idx in order:
            if checks >= max_checks:
                break
            e = adjusted[idx]
            guess = _coerce_positive_int(e.get("target_page"))
            if not guess:
                continue
            exact_matches = []
            for p in range(max(1, guess - window), guess + window + 1):
                pn = _get_printed_page_number(
                    pdf_path,
                    p - 1,
                    api_key,
                    timeout,
                    api_base=api_base,
                    model=model,
                )
                if pn is None:
                    continue
                if int(pn) == int(e.get("target_page") or 0):
                    exact_matches.append(p)
            checks += 1
            if exact_matches:
                best_p = min(exact_matches, key=lambda p: abs(p - guess))
                if best_p != guess:
                    new_e = dict(e)
                    new_e["target_page"] = best_p
                    adjusted[idx] = new_e
                    changed += 1

    if changed:
        console.print(
            f"[cyan]Printed-page verification adjusted {changed} entries "
            f"(checked {checks}; exact matches only)."
        )
    return adjusted

def _scan_with_adaptive_pages(
    *,
    api_key: str,
    pdf_path: Path | None,
    remote_url: str | None,
    initial_limit: int,
    max_pages: int,
    step: int,
    timeout: int,
    poll_interval: int,
    auto_expand: bool,
    contains: str | None,
    pattern: re.Pattern[str] | None,
    fuzzy_threshold: float | None,
    batch_size: int,
    max_workers: int,
    api_base: str | None,
    model: str | None,
) -> tuple[list[dict[str, Any]], int, int | None, list[dict[str, Any]]]:
    current_limit = max(initial_limit, 0)
    effective_step = max(step, 1)
    upper_bound = max_pages if max_pages > 0 else None
    should_expand = auto_expand and current_limit > 0

    # Track cumulative results across incremental batches so that when a TOC
    # appears in a later window we still return earlier candidates.
    cumulative_entries: list[dict[str, Any]] = []
    cumulative_fingerprints: list[dict[str, Any]] = []
    page_offset: int | None = None

    previous_limit = 0

    while True:
        # When current_limit is 0, we scan the entire document in a single call,
        # preserving the existing "scan all pages" behaviour.
        if current_limit == 0:
            start_page = 1
            window_size = 0
        else:
            start_page = previous_limit + 1
            window_size = current_limit - previous_limit
            if window_size <= 0:
                # Nothing new to scan; avoid an infinite loop.
                return (
                    infer_missing_targets(
                        filter_entries(
                            deduplicate_entries(
                                cumulative_entries,
                                fuzzy_threshold=fuzzy_threshold,
                            ),
                            contains=contains,
                            pattern=pattern,
                        )
                    ),
                    current_limit,
                    page_offset,
                    cumulative_fingerprints,
                )

        if is_interactive():
            if window_size <= 0:
                range_label = "entire document"
            else:
                end_page = start_page + window_size - 1
                range_label = f"pages {start_page}-{end_page}"

            # Show current scan window and cumulative entries while delegating
            # per-batch updates to the VLM progress callback.
            with create_progress(
                f"Scanning {range_label}... | Found: {len(cumulative_entries)} entries",
                total=None,
            ) as (progress, task_id):
                reporter = ProgressReporter(progress, task_id)

                def _progress_callback(
                    completed_batches: int,
                    total_batches: int,
                    status: str,
                ) -> None:
                    desc = (
                        f"Calling VLM API... {completed_batches}/{total_batches} "
                        f"batches | {status}"
                    )
                    reporter(completed_batches, total_batches or 1, desc)

                json_path = fetch_document_json(
                    pdf_path,
                    api_key,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    page_limit=window_size,
                    remote_url=remote_url,
                    batch_size=batch_size,
                    start_page=start_page,
                    api_base=api_base,
                    model=model,
                    max_workers=max_workers,
                    progress_callback=_progress_callback,
                )
        else:
            json_path = fetch_document_json(
                pdf_path,
                api_key,
                poll_interval=poll_interval,
                timeout=timeout,
                page_limit=window_size,
                remote_url=remote_url,
                batch_size=batch_size,
                start_page=start_page,
                api_base=api_base,
                model=model,
                max_workers=max_workers,
            )

        try:
            raw_data = load_json(json_path)
        finally:
            Path(json_path).unlink(missing_ok=True)

        batch_fingerprints: list[dict[str, Any]] = []
        if isinstance(raw_data, dict) and "toc" in raw_data:
            entries_data = raw_data.get("toc")
            raw_offset = raw_data.get("page_offset")
            try:
                batch_offset = int(raw_offset)
            except (TypeError, ValueError, OverflowError):
                batch_offset = None
            if batch_offset is not None:
                page_offset = batch_offset
            fps = raw_data.get("fingerprints")
            if isinstance(fps, list):
                batch_fingerprints = fps
        else:
            entries_data = raw_data

        # Accumulate entries and fingerprints from this batch into the running
        # collections so we can deduplicate and filter across all scanned pages.
        batch_entries = extract_toc_entries(entries_data)
        cumulative_entries.extend(batch_entries)
        cumulative_fingerprints.extend(batch_fingerprints)

        processed_entries = deduplicate_entries(
            cumulative_entries,
            fuzzy_threshold=fuzzy_threshold,
        )
        processed_entries = filter_entries(
            processed_entries,
            contains=contains,
            pattern=pattern,
        )
        processed_entries = infer_missing_targets(processed_entries)

        if processed_entries:
            return processed_entries, current_limit, page_offset, cumulative_fingerprints

        if not should_expand:
            return processed_entries, current_limit, page_offset, cumulative_fingerprints

        next_limit = current_limit + effective_step
        if upper_bound is not None:
            next_limit = min(next_limit, upper_bound)

        if next_limit == current_limit:
            return processed_entries, current_limit, page_offset, cumulative_fingerprints

        if is_interactive():
            console.print(
                f"[cyan]Expanding to pages {previous_limit + 1}-{next_limit}...[/]"
            )
        else:
            console.print(
                f"[yellow]未找到目录，扩展扫描页数到 {next_limit} 页 (批量 {batch_size})...[/]"
            )
        previous_limit = current_limit
        current_limit = next_limit


def _run_help(args: argparse.Namespace) -> None:
    parser = build_parser()
    subparsers = getattr(parser, "_subparsers_action", None)
    topic = args.topic

    if topic and isinstance(subparsers, argparse._SubParsersAction):
        subparser = subparsers.choices.get(topic)
        if subparser:
            subparser.print_help()
            return
        console.print(f"[yellow]Unknown command '{topic}'. Showing available commands.[/]")

    parser.print_help()
    if topic is None and isinstance(subparsers, argparse._SubParsersAction):
        scan_parser = subparsers.choices.get("scan")
        if scan_parser:
            console.print("\n[b]scan command options:[/]")
            console.print(scan_parser.format_help(), markup=False, highlight=False)


if __name__ == "__main__":
    main()
