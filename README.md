# Subtitle Translator CLI

Translate English SRT subtitle files into natural Brazilian Portuguese using the OpenAI API.

## Installation

```bash
git clone https://github.com/fecosta/subtitle-translator-cli.git
cd subtitle-translator-cli

uv sync --extra dev
```

Requires Python 3.12 or newer.

## Configuration

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4.1-mini"   # optional
```

Copy `.env.example` to `.env` and fill in your key if your shell loads dotenv files.

## Usage

```bash
uv run subtranslate translate movie.en.srt
```

Output is written to `movie.pt-BR.srt` by default.

### Options

```bash
subtranslate translate movie.en.srt \
  --from en \
  --to pt-BR \
  --model gpt-4.1-mini \
  --output movie.pt-BR.srt \
  --batch-size 40 \
  --force \
  --dry-run \
  --verbose
```

| Option | Description |
|--------|-------------|
| `--from` | Source language (default: `en`) |
| `--to` | Target language (default: `pt-BR`) |
| `--model` | OpenAI model (default: `OPENAI_MODEL` or `gpt-4.1-mini`) |
| `--output`, `-o` | Output SRT file |
| `--batch-size` | Subtitles per translation batch (default: 40) |
| `--dry-run` | Parse input and estimate batches/tokens without calling the API |
| `--force` | Overwrite an existing output file |
| `--verbose` | Enable verbose output |

### NAS example

```bash
uv run subtranslate translate \
  "/mnt/nas/data/media/movies/Interstellar (2014)/Interstellar.en.srt"
```

The CLI works with any locally mounted filesystem, including NAS mounts. It does not modify or remount NAS infrastructure.

## Output naming

If `--output` is omitted, the default output path is derived from the input:

- `movie.en.srt` -> `movie.pt-BR.srt`
- `movie.srt` -> `movie.pt-BR.srt`

## API configuration

- `OPENAI_API_KEY` is required.
- `OPENAI_MODEL` overrides the default model.
- `SUBTRANSLATE_BATCH_SIZE` overrides the default batch size.

Never commit API credentials to version control.

## Checkpoints and resume

The CLI saves progress after every successful batch to a local checkpoint in `~/.cache/subtranslate/checkpoints/`. Checkpoints are keyed to the source file's content hash and the translation settings (source/target language, model, batch size).

If a batch fails, the error message shows the checkpoint location. Rerun the exact same command to resume from the last completed batch without retransmitting finished work:

```bash
uv run subtranslate translate movie.en.srt
```

The checkpoint is deleted automatically once the final SRT is written and validated.

## Structured outputs

For supported OpenAI models (gpt-4o and gpt-4.1 families), the CLI requests strict JSON-schema responses to improve reliability. Other models fall back to JSON-mode with the same response contract.

## Cost considerations

Translation cost depends on subtitle length, batch size, and model pricing. The CLI reports actual token usage after translation and shows an estimated cost for known models during dry-runs. Use `--dry-run` to preview the expected number of batches and tokens before spending API credits.

## Troubleshooting

- **Missing API key**: export `OPENAI_API_KEY`.
- **Rate limits**: translation uses bounded retries with exponential backoff; large files may still hit rate limits.
- **Batch failed**: the error message includes the batch number, expected ids, and checkpoint path. Rerun the same command to resume.
- **Output already exists**: use `--force` to overwrite.
- **Malformed SRT**: ensure the file is valid UTF-8 SRT with correct timestamps.
- **NAS permission errors**: verify the mount and directory permissions; the CLI will report filesystem errors without remounting.

## Development commands

```bash
uv run pytest
uv run ruff check .
uv build
```

## Limitations

- Only SRT files are supported.
- Only the OpenAI API is supported.
- Requires an active internet connection and valid OpenAI credentials.
- Cost estimates are approximate and based on built-in pricing for common models.

## License

MIT
