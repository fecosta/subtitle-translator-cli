# AGENTS.md — Subtitle Translator CLI

## Project conventions

- Python 3.12+, `uv` for dependency and project management.
- Package source lives in `src/subtranslate/`.
- Use `dataclasses` for simple data containers; avoid heavy abstractions.
- Keep CLI code in `cli.py`, business logic in `translator.py`, OpenAI client in `openai_client.py`.
- All filesystem writes must go through `fileio.atomic_write_path`.
- Never modify the source subtitle file; never hardcode credentials.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv build
uv run subtranslate translate <file.srt> [options]
```

## Translation flow

1. Parse SRT with `pysubs2` and assign stable 1-based ids.
2. Build overlapping batches (`batching.py`) with context entries.
3. Send each batch to OpenAI JSON mode. Context entries are marked `"translate": false` and are **not** expected in the response.
4. Validate that the response contains exactly the target identifiers, each with a non-empty translation.
5. Retry recoverable errors with exponential backoff (max 3 retries).
6. Write to a temp SRT, validate it, then atomically install the final output.

## Testing

- Mock OpenAI calls; never use real credentials in tests.
- Add tests under `tests/` using pytest.
- Include fixtures in `tests/fixtures/`.

## Constraints

- Only SRT input/output, only OpenAI API.
- Do not add web frameworks, databases, Docker, local LLM inference, or media-server integrations.
