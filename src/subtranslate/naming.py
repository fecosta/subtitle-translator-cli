from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SOURCE_LANG_TAGS = frozenset({"en", "eng", "english"})
BRAZILIAN_PORTUGUESE_TAGS = frozenset({"pt-br", "pt_br", "pob"})
GENERIC_PORTUGUESE_TAGS = frozenset({"pt", "por"})
TARGET_LANG_TAGS = BRAZILIAN_PORTUGUESE_TAGS | GENERIC_PORTUGUESE_TAGS
SUBTITLE_ATTRIBUTES = frozenset({"forced", "sdh", "hi", "cc"})


@dataclass(frozen=True, slots=True)
class SubtitleName:
    """Parsed components of an external SRT subtitle filename."""

    path: Path
    movie_name: str
    language_tag: str | None
    attributes: tuple[str, ...]
    extension: str

    @property
    def is_source_language(self) -> bool:
        return self.language_tag is not None and self.language_tag.lower() in SOURCE_LANG_TAGS

    @property
    def is_target_language(self) -> bool:
        return self.language_tag is not None and self.language_tag.lower() in TARGET_LANG_TAGS

    @property
    def is_brazilian_portuguese(self) -> bool:
        if self.language_tag is None:
            return False
        return self.language_tag.lower() in BRAZILIAN_PORTUGUESE_TAGS

    @property
    def is_generic_portuguese(self) -> bool:
        if self.language_tag is None:
            return False
        return self.language_tag.lower() in GENERIC_PORTUGUESE_TAGS

    @property
    def is_unlabeled(self) -> bool:
        return self.language_tag is None

    @property
    def has_specialized_variant(self) -> bool:
        return any(attr.lower() in SUBTITLE_ATTRIBUTES for attr in self.attributes)

    def output_path(self, target_lang: str) -> Path:
        """Return the Jellyfin-compatible output path for the target language."""
        parts = [self.movie_name, target_lang, *self.attributes]
        return self.path.with_name(".".join(parts) + self.extension)


def _classify_token(token: str) -> str | None:
    lower = token.lower()
    if lower in SOURCE_LANG_TAGS or lower in TARGET_LANG_TAGS:
        return "language"
    if lower in SUBTITLE_ATTRIBUTES:
        return "attribute"
    return None


def parse_subtitle_name(path: Path) -> SubtitleName:
    """Parse a subtitle filename into its structural components.

    The movie name is everything before the first recognized language or
    attribute token. The first recognized language tag becomes the language;
    subsequent recognized tokens are treated as attributes. This keeps the
    parser conservative and predictable for Jellyfin-style filenames.
    """
    extension = path.suffix.lower()
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    tokens = stem.split(".")

    first_tag_index: int | None = None
    for index, token in enumerate(tokens):
        if _classify_token(token) is not None:
            first_tag_index = index
            break

    if first_tag_index is None:
        return SubtitleName(
            path=path,
            movie_name=stem,
            language_tag=None,
            attributes=(),
            extension=extension,
        )

    movie_name = ".".join(tokens[:first_tag_index])
    language_tag: str | None = None
    attributes: list[str] = []
    for token in tokens[first_tag_index:]:
        classification = _classify_token(token)
        if classification == "language" and language_tag is None:
            language_tag = token
        elif classification == "attribute":
            attributes.append(token)
        # Unknown tokens after the first tag are ignored; they do not affect
        # naming and are not preserved, which avoids accidents with hashes,
        # release groups, or other unrelated suffixes.

    return SubtitleName(
        path=path,
        movie_name=movie_name,
        language_tag=language_tag,
        attributes=tuple(attributes),
        extension=extension,
    )


def default_output_path(input_path: Path, target_lang: str) -> Path:
    """Generate a default output path for a translated subtitle file."""
    parsed = parse_subtitle_name(input_path)
    return parsed.output_path(target_lang)
