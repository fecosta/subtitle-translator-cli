class SubtranslateError(Exception):
    """Base exception for the application."""


class SubtitleError(SubtranslateError):
    """Raised when a subtitle file cannot be parsed or is invalid."""


class TranslationError(SubtranslateError):
    """Raised when translation fails after retries."""


class BatchValidationError(TranslationError):
    """Raised when a batch repeatedly returns invalid model output."""


class OutputExistsError(SubtranslateError):
    """Raised when the output file already exists and --force was not given."""


class FileConflictError(SubtranslateError):
    """Raised when the output path conflicts with the input path."""


class ConfigurationError(SubtranslateError):
    """Raised when required configuration is missing or invalid."""
