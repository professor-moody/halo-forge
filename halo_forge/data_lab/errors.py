"""Errors raised by the Dataset Lab backend."""


class DatasetLabError(Exception):
    """Base class for actionable Dataset Lab failures."""


class SourceError(DatasetLabError):
    """A source cannot be loaded or no longer matches its fingerprint."""


class SchemaError(DatasetLabError):
    """A record cannot be converted to its declared canonical schema."""


class RecipeError(DatasetLabError):
    """A recipe is invalid or cannot be executed."""


class VersionError(DatasetLabError):
    """An immutable version is incomplete or has failed verification."""


class JobError(DatasetLabError):
    """A persistent background job cannot be operated on."""
