# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `ebooktoc/`: `cli.py` handles argument parsing, `toc_parser.py` cleans SiliconFlow responses, `siliconflow_api.py` speaks to the API, `pdf_writer.py` embeds bookmarks, and `utils.py` hosts filesystem helpers. Generated artifacts belong in `output/json` or `output/pdf` so the repository root stays clean. `pyproject.toml` and `pdm.lock` define dependencies and the `ebook-toc` script entry.

## Build, Test, and Development Commands
- `pdm install` – install the Python 3.9+ environment declared in `pyproject.toml`.
- `pdm run ebook-toc scan sample.pdf --api-key $SILICONFLOW_API_KEY --output output/json/sample_toc.json` – scan a PDF and write TOC JSON.
- `pdm run ebook-toc apply sample.pdf output/json/sample_toc.json --output output/pdf/sample_with_toc.pdf` – embed an existing TOC into a PDF copy.
- `pdm run python -m ebooktoc.cli help scan` – inspect CLI help without invoking the console script.
- `pdm run pytest tests/` – future test entry once `tests/` lands.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and type hints on public helpers (match the current modules). Prefer `Path` objects, descriptive CLI flags, `f`-strings, and small reusable helpers in `utils.py`. Emit user-facing text through the shared `rich.Console` so formatting stays consistent across commands.

## Testing Guidelines
Automated tests are not yet present, so introduce a `tests/` package with `pytest`. Mock SiliconFlow calls by patching `fetch_document_json`, keep fixture PDFs tiny, and cover parser utilities plus CLI validation before touching PDF writes. Run `pdm run pytest` prior to any PR and note skipped scenarios in the PR description.

## Commit & Pull Request Guidelines
Commits use concise imperative subjects (see `git log`, e.g., “Add apply subcommand to reuse saved TOC JSON”). Keep each commit focused on one concern and mention any user-facing change in the body. Pull requests should explain motivation, summarize CLI impact, list commands/tests executed, link issues, and add JSON or console snippets when introducing new prompt flows.

## Security & Configuration Tips
Never commit SiliconFlow API keys; export them in your shell profile and pass them via `--api-key $SILICONFLOW_API_KEY`. Keep proprietary PDFs out of version control and clean intermediate files under `output/` before pushing. Prefer HTTPS for `--remote-url` downloads and extend `.gitignore` before adding new caches or logs so sensitive data stays local.
