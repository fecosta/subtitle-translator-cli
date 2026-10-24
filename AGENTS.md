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
3. Call OpenAI JSON mode; validate returned ids against batch ids.
4. Retry recoverable errors with exponential backoff (max 3 retries).
5. Write to a temp SRT, validate it, then `os.replace` atomically.

## Testing

- Mock OpenAI calls; never use real credentials in tests.
- Add tests under `tests/` using pytest.
- Include fixtures in `tests/fixtures/`.

## Constraints

- Only SRT input/output, only OpenAI API.
- Do not add web frameworks, databases, Docker, local LLM inference, or media-server integrations.
