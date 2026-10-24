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
        safe_lang = _sanitize_language_tag(target_lang)
        parts = [self.movie_name, safe_lang, *self.attributes]
        return self.path.with_name(".".join(parts) + self.extension)


def _sanitize_language_tag(tag: str) -> str:
    """Make a language tag safe for a filename.

    ponytail: full filesystem sanitization is overkill here; the CLI only
    accepts language codes. Upgrade to a broader charset filter if users ever
    pass arbitrary strings through --to.
    """
    return (
        tag.replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace("\\", "-")
    )


def parse_subtitle_name(path: Path) -> SubtitleName:
    """Parse a subtitle filename into its structural components.

    Tags are read from the right, which is the Jellyfin/Bazarr convention for
    external subtitles: extension, then optional attributes, then an optional
    language tag, with everything else forming the movie name. This avoids
    mistaking title words such as "Hi" or "En" for language tags.
    """
    extension = path.suffix.lower()
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    tokens = stem.split(".")

    # Consume trailing recognized attributes.
    attributes: list[str] = []
    index = len(tokens) - 1
    while index >= 0 and tokens[index].lower() in SUBTITLE_ATTRIBUTES:
        attributes.append(tokens[index])
        index -= 1
    attributes.reverse()

    # Consume a single language tag immediately before the attributes.
    language_tag: str | None = None
    if index >= 0:
        lower = tokens[index].lower()
        if lower in SOURCE_LANG_TAGS or lower in TARGET_LANG_TAGS:
            language_tag = tokens[index]
            index -= 1

    movie_name = ".".join(tokens[: index + 1])
    if not movie_name:
        # Avoid producing a hidden dotfile when the stem is only a tag.
        movie_name = stem

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
