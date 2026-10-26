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

If `--output` is omitted, the default output path is derived from the input filename. The CLI recognizes common English-language suffixes case-insensitively, replaces the source language tag with the target language, and preserves recognized subtitle attributes:

| Input | Default output |
|---|---|
| `movie.en.srt` | `movie.pt-BR.srt` |
| `movie.eng.srt` | `movie.pt-BR.srt` |
| `movie.english.srt` | `movie.pt-BR.srt` |
| `movie.srt` | `movie.pt-BR.srt` |
| `movie.en.forced.srt` | `movie.pt-BR.forced.srt` |
| `movie.eng.sdh.srt` | `movie.pt-BR.sdh.srt` |

Recognized source-language tags: `en`, `eng`, `english`.
Recognized subtitle attributes: `forced`, `sdh`, `hi`, `cc`.

Use `--output` to override the generated path. The original subtitle file is never modified.

## Library scanning

The `scan` command recursively discovers SRT subtitles and produces a read-only translation plan. No API calls are made.

```bash
uv run subtranslate scan /mnt/nas/data/media/movies
```

The scanner groups subtitles by directory and movie basename, then decides per movie:

| Status | Meaning |
|---|---|
| `TRANSLATE` | One unambiguous English source exists and no matching Portuguese subtitle was found. |
| `SKIP` | A matching Portuguese subtitle already exists, no recognized English source was found, or the source is a specialized variant. |
| `AMBIGUOUS` | Multiple plausible English sources exist for the same movie. |
| `ERROR` | The file could not be read or parsed. |

Recognized English source tags: `en`, `eng`, `english`.
Recognized Portuguese tags: `pt-BR`, `pt_BR`, `pob` (confirmed Brazilian), `pt`, `por` (generic).

By default `forced`, `sdh`, `hi`, and `cc` variants are excluded from translation. Use `--include-variants` to treat them as sources.

Use `--require-mount` to fail closed when the library path is not under any mounted filesystem. The path itself does not need to be the mount point; a parent such as `/mnt/nas/data` is sufficient when scanning `/mnt/nas/data/media/movies`.

Use `--mount-point /mnt/nas/data` when you want to validate that the library is under a *specific* mount point, not just any mounted filesystem. This is the safest option for the homelab NAS.

## Subtitle coverage discovery

The `coverage` command discovers movies by their video files and reports which ones already have external subtitles and embedded subtitle tracks. It is read-only: it does not call the API, translate, rename, or modify files, and it does not extract embedded subtitles.

```bash
uv run subtranslate coverage /mnt/nas/data/media/movies
```

For each discovered movie, it chooses one primary category based on the available external SRT files:

| Category | Meaning |
|---|---|
| `HAS_PT_BR` | Confirmed Brazilian Portuguese subtitle exists. |
| `HAS_GENERIC_PT` | Generic Portuguese subtitle exists; Brazilian variant is not confirmed. |
| `HAS_EN_ONLY` | Recognized English subtitle exists without Portuguese. |
| `HAS_UNLABELED` | Unlabeled SRT exists; language is unknown. |
| `HAS_SPECIALIZED_ONLY` | Only specialized variants (`forced`/`sdh`/`hi`/`cc`) exist; no full subtitle coverage. |
| `NO_EXTERNAL_SUBS` | No associated external SRT found. |
| `AMBIGUOUS` | Subtitle-to-video association cannot be established safely. |

Specialized variants (`forced`, `sdh`, `hi`, `cc`) are preserved in the inventory but do not count as full subtitle coverage.

### Embedded subtitle discovery

`coverage` also inspects supported video files with `ffprobe` and lists embedded subtitle streams:

- Stream index and subtitle-relative index.
- Codec and codec kind (`text`, `image`, or `unknown`).
- Normalized language tag (`pt-BR`, `pt`, `en`, or `null` when unknown).
- Raw language tag as reported by `ffprobe`.
- Track title.
- Forced and hearing-impaired (SDH) disposition flags.

Supported video extensions: `.mkv`, `.mp4`, `.avi`, `.m4v`.

Text-based codecs considered convertible to SRT include: `subrip`, `srt`, `ass`, `ssa`, `webvtt`, `text`, `mov_text`.
Image-based codecs detected as not directly translatable include: `hdmv_pgs_subtitle`, `dvd_subtitle`, `vobsub`, `pgs`, `dvb_subtitle`.

`ffprobe` must be installed and available on `PATH`. Probes use a 60-second timeout, run with argument arrays, and never modify media files. A failed probe is recorded as an error and does not erase external subtitle information already discovered.

Options:

```bash
# Show only movies without confirmed Brazilian Portuguese subtitles
uv run subtranslate coverage /mnt/nas/data/media/movies --missing-only

# Machine-readable JSON output
uv run subtranslate coverage /mnt/nas/data/media/movies --format json
```

## Extract embedded English subtitles

The `extract-embedded` command reads supported video files with `ffprobe`, selects eligible embedded English text subtitle tracks, and writes them to external `.en.srt` files using `ffmpeg`. It does not modify video files and does not call the OpenAI API.

```bash
uv run subtranslate extract-embedded /mnt/nas/data/media/movies
```

Pass a library root, a single movie directory, or a single video file. When a single video file is given, its parent directory is scanned and only that video is processed.

Preview extractions without running `ffmpeg` or writing files:

```bash
uv run subtranslate extract-embedded /mnt/nas/data/media/movies --dry-run --require-mount --mount-point /mnt/nas/data
```

### Track selection

A track is eligible when it is:

- English (`en`, `eng`, `english`),
- a supported text codec (`subrip`, `srt`, `ass`, `ssa`, `webvtt`, `text`, `mov_text`),
- not forced,
- not SDH / hearing impaired.

Forced, SDH, image-based (PGS, VobSub, DVB), and unknown-language tracks are ignored. When multiple eligible tracks exist for one video, the command picks `subrip`/`srt` first, then the lowest subtitle stream index.

### Skip reasons

A movie or video is skipped when:

- it already has a non-specialized external English subtitle (`.en.srt`, `.eng.srt`, `.english.srt`, etc.),
- `ffprobe` reports an error,
- subtitle-to-video association is ambiguous,
- no eligible embedded English text track is found,
- the target output path already exists (race protection; never overwrites).

Movies with multiple video files are handled per video; each eligible video receives its own `<video stem>.en.srt`.

### Exit codes

- `0`: no failures.
- `1`: one or more extractions failed (other movies are still processed).
- `2`: mount or path validation error.

JSON output contains stable field names, no Rich formatting, and both `summary` (external SRT categories) and `embedded_summary` blocks with explicit counts. Each movie object includes:

- `subtitles`: existing external SRT inventory.
- `embedded_subtitles`: list of discovered embedded streams with `stream_index`, `subtitle_stream_index`, `video_file`, `codec`, `codec_kind`, `language_tag`, `raw_language_tag`, `title`, and boolean flags.
- `probe_error`: `null` on success, or an error message when `ffprobe` failed, timed out, or returned malformed output.

Movies marked `NO_EXTERNAL_SUBS` may still contain embedded subtitles; the embedded inventory is reported separately and does not change the external-only category.

## Batch execution

The `batch` command runs `scan`, shows the plan and estimated cost, asks for confirmation, and translates all eligible files sequentially.

```bash
uv run subtranslate batch /mnt/nas/data/media/movies
```

Noninteractive execution (use only with explicit authorization):

```bash
uv run subtranslate batch /mnt/nas/data/media/movies --yes
```

### Cost controls

Show the estimated cost before translation and set a spending ceiling:

```bash
uv run subtranslate batch /mnt/nas/data/media/movies --max-cost-usd 5.00
```

- The command refuses to start if the estimated plan exceeds the ceiling.
- It fails closed if pricing is unknown for the selected model.
- It rechecks the remaining budget before starting each movie and stops if the next file would exceed the ceiling.
- Reported spending is an estimate based on actual token usage; retries or resumed checkpoints can shift the real cost.

### Existing subtitles and resume

- Movies with a matching Portuguese subtitle are skipped.
- Existing Portuguese subtitles are never overwritten.
- Source subtitle files are never modified.
- Each movie uses the existing checkpoint mechanism, so interrupted translations resume without re-translating completed batches.
- A library lock prevents concurrent `batch` runs on the same path.

## NAS example on the Mac Mini

The Fedora Asahi Mac Mini mounts the PiNAS at `/mnt/nas/data`.

Safe read-only scan of the entire movie library:

```bash
uv run subtranslate scan /mnt/nas/data/media/movies
```

If you want to be extra sure the NAS mount is present:

```bash
uv run subtranslate scan /mnt/nas/data/media/movies --require-mount

# Validate against the specific NAS mount point
uv run subtranslate coverage /mnt/nas/data/media/movies --mount-point /mnt/nas/data
uv run subtranslate scan /mnt/nas/data/media/movies --mount-point /mnt/nas/data
uv run subtranslate batch /mnt/nas/data/media/movies --mount-point /mnt/nas/data
```

Translate a small test set (limit the batch to a few movies by temporarily moving others out of the scan path, or run `batch` and review the plan before confirming):

```bash
uv run subtranslate batch /mnt/nas/data/media/movies --max-cost-usd 1.00
```

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

Translation cost depends on subtitle length, batch size, and model pricing. The CLI reports actual token usage after translation and shows an estimated cost for known models during dry-runs and batch planning. Cost estimates are approximate; actual OpenAI billing may differ slightly due to retries, resumed checkpoints, or model pricing updates. Use `--dry-run` or `scan` to preview costs before spending API credits.

## Troubleshooting

- **Missing API key**: export `OPENAI_API_KEY`.
- **Rate limits**: translation uses bounded retries with exponential backoff; large files may still hit rate limits.
- **Batch failed**: the error message includes the batch number, expected ids, and checkpoint path. Rerun the same command to resume.
- **Output already exists**: use `--force` to overwrite, or remove the existing file first.
- **Malformed SRT**: ensure the file is valid UTF-8 SRT with correct timestamps. Malformed files are reported as scan errors.
- **Ambiguous source**: when a movie folder has multiple English SRT files (e.g., `Movie.en.srt` and `Movie.eng.srt`), the movie is skipped. Keep only the desired source or use `--output` with `translate`.
- **NAS permission errors / empty mount**: verify the mount and directory permissions; the CLI will report filesystem errors without remounting. Use `--require-mount` to fail closed when no ancestor is mounted, or `--mount-point /mnt/nas/data` to validate the specific expected NAS mount.

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
- Embedded subtitle discovery is read-only; embedded tracks are not extracted or translated in this release.
- Language and codec detection depend on metadata provided by `ffprobe`; missing tags are reported as `unknown`.
- Tracks whose language tag is outside the recognized `pt-BR`/`pt`/`en` set are reported as `unknown` while preserving the raw tag.
- Some containers store BCP-47 language tags in fields that `ffprobe` may not expose as the `language` tag, so real Brazilian Portuguese tracks may appear as generic `pt`.

## License

MIT
