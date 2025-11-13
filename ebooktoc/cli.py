"""Command line interface for the ebook-toc tool."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from rich.console import Console

from .fingerprints import (
    compute_pdf_fingerprints,
    dominant_dimensions,
    build_canonical_map_for_dims,
)
from .pdf_writer import write_pdf_toc
from .siliconflow_api import TOCExtractionError, fetch_document_json, _infer_page_offset
from .toc_parser import deduplicate_entries, extract_toc_entries, filter_entries
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
        description="Scan a PDF with SiliconFlow Qwen and extract table-of-contents entries.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="ebook-toc %(prog)s version 0.1.0",
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
        help="Upload or download a PDF, call SiliconFlow Qwen, and export TOC JSON.",
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
        help="SiliconFlow API token.",
    )
    scan_parser.add_argument(
        "--remote-url",
        type=str,
        default=None,
        help="Remote PDF URL to download before calling SiliconFlow.",
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
        help="SiliconFlow request timeout in seconds (default: 600).",
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
        default=3,
        help="Number of pages to send per SiliconFlow request (default: 3).",
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
        help="SiliconFlow API token (optional; improves offset inference on apply).",
    )
    apply_parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="SiliconFlow request timeout in seconds (default: 600).",
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
        original_pdf_for_output: Optional[Path] = pdf_path
        temp_download: Optional[Path] = None
        clean_pdf_path: Optional[Path] = None
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
        )
    except TOCExtractionError as err:
        console.print(f"[red]{err}[/]")
        raise SystemExit(2) from err
    except requests.RequestException as err:
        console.print(f"[red]Network error while calling SiliconFlow: {err}[/]")
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
                current_fps, page_count = compute_pdf_fingerprints(pdf_path)
            except Exception:
                current_fps, page_count = [], _get_pdf_page_count(pdf_path)

            dims = dominant_dimensions(current_fps) if current_fps else None
            canonical_map = (
                build_canonical_map_for_dims(current_fps, dims) if dims else {}
            )

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
    else:
        console.print("[cyan]未写入 PDF 目录。[/]")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(0)

    args.func(args)


def _derive_output_stem(pdf_path: Optional[Path], remote_url: Optional[str]) -> str:
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


def _print_toc_preview(entries: list[dict[str, Any]], page_offset: Optional[int]) -> None:
    if not entries:
        console.print("[yellow]未检测到目录条目。[/]")
        return

    console.print("[b]TOC Preview[/]")
    for entry in entries:
        page = entry.get("page")
        target = entry.get("target_page")
        title = entry.get("content", "").strip()
        display_target = target if target is not None else "-"
        resolved = None
        source_value = target if target is not None else entry.get("page")
        base_page = _coerce_positive_int(source_value)
        if page_offset is not None and base_page is not None:
            resolved = base_page + page_offset
        detail = (
            f" (PDF page {resolved})" if resolved is not None else ""
        )
        console.print(
            f"- page {page} -> {display_target} : {title}{detail}"
        )


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


def _coerce_positive_int(value: Any) -> Optional[int]:
    return _util_coerce_positive_int(value)


def _apply_page_mapping(
    entries: list[dict[str, Any]],
    mapping: Dict[int, int],
    page_offset: Optional[int],
    page_count: int,
) -> list[dict[str, Any]]:
    resolved_entries: list[dict[str, Any]] = []
    for entry in entries:
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
    mapping: Dict[int, int],
    initial_offset: Optional[int],
    window: int = 40,
) -> Optional[int]:
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

    try:
        raw_data = load_json(json_path)
    except OSError as err:
        console.print(f"[red]Unable to read JSON: {err}[/]")
        raise SystemExit(4) from err

    fingerprints = []
    saved_clean_map: Dict[int, int] = {}
    if isinstance(raw_data, dict) and "toc" in raw_data:
        entries = extract_toc_entries(raw_data.get("toc"))
        from .toc_parser import infer_missing_targets
        entries = infer_missing_targets(entries)
        page_offset = _coerce_positive_int(raw_data.get("page_offset"))
        stored_fps = raw_data.get("fingerprints")
        if isinstance(stored_fps, list):
            fingerprints = stored_fps
        raw_clean_map = raw_data.get("clean_map")
        if isinstance(raw_clean_map, dict):
            tmp: Dict[int, int] = {}
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
        from .toc_parser import infer_missing_targets
        entries = infer_missing_targets(entries)
        page_offset = None

    if not entries:
        console.print("[yellow]No TOC entries found in JSON. Nothing to apply.[/]")
        return

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

    try:
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
    if getattr(args, 'api_key', None):
        try:
            vlm_offset = _infer_page_offset(
                pdf_path,
                entries,
                args.api_key,
                args.timeout,
                current_fps,
            )
            if vlm_offset is not None:
                refined_offset = vlm_offset
                console.print(
                    f"[cyan]Refined printed-page offset (VLM): {refined_offset:+d} (was {page_offset}).[/]"
                )
        except TOCExtractionError:
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
                    window=max(1, int(getattr(args, 'verify_window', 6))),
                    max_checks=max(10, int(getattr(args, 'verify_max', 80))),
                )
            except Exception:
                pass
    else:
        canonical_map: Dict[int, int] = {}
        if dims and current_fps:
            canonical_map = build_canonical_map_for_dims(current_fps, dims)
        refined_offset = _refine_offset_with_mapping(entries, canonical_map, refined_offset)
        if refined_offset != page_offset and not getattr(args, 'override_offset', None):
            console.print(
                f"[cyan]Refined printed-page offset: {refined_offset:+d} (was {page_offset}).[/]"
            )
        resolved_entries = _apply_page_mapping(
            entries, canonical_map, refined_offset, page_count
        )

    try:
        result = write_pdf_toc(pdf_path, resolved_entries, output_path, page_offset=None)
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


def _apply_with_goodnotes_clean(
    pdf_path: Path,
    entries: list[dict[str, Any]],
    initial_offset: Optional[int],
    baseline_dims: Optional[tuple[int, int]],
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
        canonical_map_clean: Dict[int, int] = {}
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
        for clean_idx, orig_idx in enumerate(keep_indices, start=1):
            try:
                # Append a copy of the original page to the clean document
                clean.insert_pdf(src, from_page=orig_idx - 1, to_page=orig_idx - 1)
                clean_to_original[clean_idx] = orig_idx
            except Exception:
                continue

        # Save to a temporary file
        import tempfile

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
    api_key: Optional[str],
    timeout: int,
    window: int = 6,
    max_checks: int = 80,
) -> list[dict[str, Any]]:
    if not api_key:
        return entries
    try:
        from .siliconflow_api import _get_printed_page_number
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
            pn = _get_printed_page_number(pdf_path, p - 1, api_key, timeout)
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
    if changed:
        console.print(f"[cyan]Printed-page verification adjusted {changed} entries (checked {checks}; exact matches only).")
    return adjusted

def _scan_with_adaptive_pages(
    *,
    api_key: str,
    pdf_path: Optional[Path],
    remote_url: Optional[str],
    initial_limit: int,
    max_pages: int,
    step: int,
    timeout: int,
    poll_interval: int,
    auto_expand: bool,
    contains: Optional[str],
    pattern: Optional[re.Pattern[str]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], int, Optional[int], list[dict[str, Any]]]:
    current_limit = max(initial_limit, 0)
    effective_step = max(step, 1)
    upper_bound = max_pages if max_pages > 0 else None
    should_expand = auto_expand and current_limit > 0

    page_offset: Optional[int] = None

    while True:
        json_path = fetch_document_json(
            pdf_path,
            api_key,
            poll_interval=poll_interval,
            timeout=timeout,
            page_limit=current_limit,
            remote_url=remote_url,
            batch_size=batch_size,
        )

        try:
            raw_data = load_json(json_path)
        finally:
            Path(json_path).unlink(missing_ok=True)

        fingerprints: list[dict[str, Any]] = []
        if isinstance(raw_data, dict) and "toc" in raw_data:
            entries_data = raw_data.get("toc")
            raw_offset = raw_data.get("page_offset")
            try:
                page_offset = int(raw_offset)
            except (TypeError, ValueError, OverflowError):
                page_offset = None
            fps = raw_data.get("fingerprints")
            if isinstance(fps, list):
                fingerprints = fps
        else:
            entries_data = raw_data

        entries = extract_toc_entries(entries_data)
        entries = deduplicate_entries(entries)
        entries = filter_entries(entries, contains=contains, pattern=pattern)
        # Try to infer missing target pages from content when VLM omitted them
        from .toc_parser import infer_missing_targets
        entries = infer_missing_targets(entries)

        if entries:
            return entries, current_limit, page_offset, fingerprints

        if not should_expand:
            return entries, current_limit, page_offset, fingerprints

        next_limit = current_limit + effective_step
        if upper_bound is not None:
            next_limit = min(next_limit, upper_bound)

        if next_limit == current_limit:
            return entries, current_limit, page_offset, fingerprints

        console.print(
            f"[yellow]未找到目录，扩展扫描页数到 {next_limit} 页 (批量 {batch_size})...[/]"
        )
        current_limit = next_limit

    return [], current_limit, page_offset, []


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
