<p align="center">
  <img src="ebook-toc.png" alt="ebook-toc icon" width="160"/>
</p>

# ebook-toc

[![CI](https://github.com/pi-dal/ebook-toc/actions/workflows/ci.yml/badge.svg)](https://github.com/pi-dal/ebook-toc/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/pi-dal/ebook-toc/graph/badge.svg)](https://codecov.io/gh/pi-dal/ebook-toc)

ebook-toc is a Python CLI that extracts a book’s Table of Contents (TOC) from PDFs using a Vision-Language Model (VLM), then optionally embeds the TOC back into the PDF as bookmarks. The current implementation integrates SiliconFlow’s Qwen3‑VL‑32B‑Instruct for TOC detection and printed‑page offset estimation. It supports scanned PDFs by falling back to page images when text is unavailable.

This project is currently in Alpha and intentionally prioritizes a quick‑and‑dirty end‑to‑end path so it can be exercised and validated early. The public API and on‑disk JSON format may change before v1.0.

Note: Additional VLMs will be supported in future releases; the current integration with SiliconFlow is for evaluation and prototyping.

## Prerequisites

- Python 3.10+
- [PDM](https://pdm.fming.dev)

## Installation

```bash
pdm install
```

## Usage

```bash
pdm run ebook-toc scan input.pdf --api-key sk-xxx --output toc.json --pages 20

# or process a remote PDF directly
pdm run ebook-toc scan --remote-url https://example.com/sample.pdf --api-key sk-xxx --output toc.json

# scan with GoodNotes-clean workflow (strip non-dominant-size insertions before scanning)
pdm run ebook-toc scan input.pdf --goodnotes-clean --api-key sk-xxx --output toc.json

# show CLI help
pdm run python -m ebooktoc.cli help scan

# apply an existing TOC JSON to a PDF
pdm run ebook-toc apply input.pdf output/json/input_toc.json --output output/pdf/input_with_toc.pdf

# apply with GoodNotes-clean workflow (remove non-dominant-size inserts before resolving)
pdm run ebook-toc apply input.pdf output/json/input_toc.json --goodnotes-clean --output output/pdf/input_with_toc.pdf
```

- `input.pdf`: path to the source PDF.
- `--api-key`: SiliconFlow API token.
- `--output`: path to the output JSON file (defaults to `toc.json`).
- `--pages`: number of leading pages to analyze (default `10`, use `0` for the full document).
- `--remote-url`: optional PDF URL; when provided the local `input.pdf` argument can be omitted.
- `--timeout`: SiliconFlow request timeout in seconds (default `600`).
- `--max-pages`: upper bound for automatic page expansion when no TOC is detected (default `50`).
- `--step-pages`: increase in pages per expansion step (default `10`).
- `--no-auto-expand`: disable automatic expansion and use only the initial `--pages` value.
- `--batch-size`: number of pages sent to SiliconFlow per request (default `3`).
- `--save-json`: skip the prompt and persist the TOC JSON to disk.
- `--apply-toc`: skip the prompt and write the TOC into the PDF as bookmarks.
- `scan --goodnotes-clean`: detect and strip non-dominant-size pages (e.g., GoodNotes inserts) before scanning,
  to improve printed-page offset inference and TOC stability.
- `apply --goodnotes-clean`: detect and strip non-dominant-size pages (e.g., GoodNotes insertions),
  resolve bookmarks against the clean PDF, then map them back to the original PDF for writing.
- `--dry-run`: preview detected TOC entries without creating files.
- `--filter-contains`: keep only entries whose content includes the given substring (case-insensitive).
- `--filter-regex`: keep only entries whose content matches the given regular expression (case-insensitive).

## Output

The CLI writes a JSON list containing the detected table-of-contents items:

```json
[
  {"page": 4, "target_page": 5, "content": "Chapter 1: Introduction"},
  {"page": 4, "target_page": 15, "content": "Chapter 2: Methods"}
]
```

`page` records where the TOC text was found, whereas `target_page` (if present) captures the destination page referenced in the entry. The CLI prints a status message before scanning and reports the number of entries upon completion. Future extensions can reuse this output to create PDF bookmarks or other metadata.

If no entries are detected within the initial page window, the tool automatically expands the range by `--step-pages` (unless `--no-auto-expand` is set) until it reaches `--max-pages`. Each batch submitted to SiliconFlow (default 3 pages; configurable with `--batch-size`) is deduplicated, and the results can be further narrowed via `--filter-contains` / `--filter-regex`.

During scanning the CLI also samples a few PDF pages with the VLM to infer the offset between the PDF index and the printed page number. The inferred offset is shown in the terminal (and stored in the JSON) so that bookmarks align with the book’s logical pagination, even when the document contains unnumbered front matter.

After the scan finishes, the CLI prompts whether to save the TOC JSON or embed bookmarks into the PDF (you can skip the prompts with `--save-json` / `--apply-toc`). By default JSON files go to `output/json/`, PDF copies with bookmarks go to `output/pdf/`, using names derived from the source document. If you opt out of saving JSON, the entries are printed directly to the terminal; if you run with `--dry-run`, the tool only prints a preview list and leaves the file system untouched. The saved JSON also contains lightweight page fingerprints and a canonical `page_map` (logical page → PDF page) computed from dominant page dimensions, so `apply` can align bookmarks even if apps like GoodNotes inserted extra pages later.

You can rerun bookmark creation later with `ebook-toc apply`, passing the previously saved JSON file.

## How It Works

- Input handling (`ebooktoc/cli.py`): validates local files or downloads remote PDFs; optionally creates a GoodNotes‑cleaned copy by keeping only dominant page sizes.
- Page extraction (`ebooktoc/siliconflow_api.py`): extracts per‑page text (or renders JPEG when text is empty), batches VLM requests, and parses JSON robustly.
- TOC parsing (`ebooktoc/toc_parser.py`): normalizes entries, deduplicates, filters, and infers missing trailing numeric targets.
- Offset and mapping (`ebooktoc/fingerprints.py`, `ebooktoc/cli.py`): computes dominant dimensions, builds a canonical index map (logical → PDF), and estimates printed‑page offsets by sampling pages with the VLM; stores `toc`, `page_offset`, `fingerprints`, and `page_map` in JSON.
- Apply phase (`ebooktoc/pdf_writer.py`, `ebooktoc/cli.py`): rebuilds the canonical map, refines the offset, resolves target pages, and writes bookmarks.

Primary modules:
- `ebooktoc/cli.py`: CLI commands (`scan`, `apply`), coordination, prompts, and IO
- `ebooktoc/siliconflow_api.py`: batching, VLM calls, JSON parsing, offset estimation
- `ebooktoc/toc_parser.py`: TOC normalization, deduplication, filtering, heuristics
- `ebooktoc/fingerprints.py`: dominant‑dimension detection and canonical index mapping
- `ebooktoc/pdf_writer.py`: bookmark embedding and result reporting
- `ebooktoc/utils.py`: filesystem and small helpers

## Development Guide

- Environment setup
  - Install PDM, then run `pdm install -G test`.
  - Python 3.10+ is required.
- Commands
  - Run locally: `pdm run ebook-toc ...`
  - Tests: `pdm run pytest` (coverage is enabled by default via pyproject)
- CI
  - GitHub Actions runs tests across Python 3.10–3.14 with PDM.
  - Coverage is uploaded to Codecov; see badges above.
- Style and structure
  - Follow PEP 8; prefer `Path`, type hints, and shared `rich.Console` output.
  - Keep generated artifacts under `output/json` and `output/pdf`.
- Security and privacy
  - Never commit API keys or proprietary PDFs; pass keys via `--api-key` or env vars.
  - Extend `.gitignore` for new caches or artifacts before adding tools that persist them.

## Project Status

- Alpha quality. The current solution is intentionally quick‑and‑dirty to validate the end‑to‑end flow with real PDFs.
- API and JSON schema may evolve; minor breaking changes are possible before v1.0.
- The SiliconFlow integration is a testbed; additional VLM backends will be supported.

## Roadmap / TODO

- Support additional VLM backends beyond SiliconFlow
- Expand and harden the test suite
- Improve and extend developer documentation
- Add an interactive TUI for local use

## Acknowledgements

- Powered by [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF parsing and bookmark embedding.
- SiliconFlow Qwen3‑VL‑32B‑Instruct for TOC detection and printed‑page sampling.

## License

- License to be determined before v1.0. Until then, please consider this code provided for evaluation and prototyping.
