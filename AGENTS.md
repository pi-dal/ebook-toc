# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `ebooktoc/`:
  - `cli.py` – CLI entrypoint (`scan`, `apply`), argument parsing, prompts.
  - `vlm_api.py` – OpenAI‑format VLM client (SiliconFlow by default), batching, JSON parsing, offset inference.
  - `toc_parser.py` – TOC normalization, filtering, and heuristics.
  - `pdf_writer.py` – writes bookmarks into PDFs.
  - `fingerprints.py` – page fingerprinting and canonical page mapping.
  - `utils.py` – filesystem and small helpers.
- Tests live in `tests/` (`test_*.py`).
- Generated artifacts should go under `output/json/` and `output/pdf/` (not committed).

## Build, Test, and Development Commands
- Install dependencies: `pdm install`  
- Run CLI: `pdm run ebook-toc scan input.pdf --api-key $KEY --output output/json/input_toc.json`  
  and `pdm run ebook-toc apply input.pdf output/json/input_toc.json --output output/pdf/input_with_toc.pdf`.
- Inspect help: `pdm run python -m ebooktoc.cli help scan`.
- Run tests with coverage: `pdm run pytest`.

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4‑space indentation.
- Use type hints on public functions, `Path` for filesystem paths, and f‑strings for formatting.
- Prefer small, reusable helpers in `utils.py`.
- Emit user‑facing messages via a shared `rich.Console` instance (`console` in `cli.py`).

## Testing Guidelines
- Test framework: `pytest` (with `pytest-cov` configured in `pyproject.toml`).
- Place tests in `tests/` and name files `test_*.py`.
- When hitting the VLM, mock at the API boundary (e.g., patch `ebooktoc.vlm_api.fetch_document_json` or `_call_chat_completion`) to keep tests fast and deterministic.
- Aim to keep or improve coverage for touched modules.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., “Add apply subcommand for saved TOC JSON”).
- Keep each PR focused; describe motivation, key changes, and CLI impact.
- Include example commands and/or JSON snippets when changing prompt flows or VLM behavior.
- Link related issues and note which tests/commands were run.

## Security & Configuration Tips
- Never commit API keys; pass them via `--api-key` or environment variables.
- Do not commit proprietary PDFs; keep them local under `output/`.
- Prefer HTTPS for `--remote-url` sources and extend `.gitignore` before adding new caches or logs.

## Agent-Specific Instructions
You are operating in an environment where `ast-grep` is installed. For any code search that requires understanding of syntax or code structure, you should default to using:

```bash
ast-grep --lang python -p '<pattern>'
```

Adjust the `--lang` flag as needed for the specific programming language. Avoid using text-only search tools unless a plain-text search is explicitly requested.
