"""Review Lab domain errors shared by dashboard, API, CLI, and workers."""


class ReviewLabError(RuntimeError):
    """Base class for review-domain failures."""


class ReviewValidationError(ReviewLabError, ValueError):
    """A supplied schema, annotation, acquisition, or policy is invalid."""


class ReviewEligibilityError(ReviewLabError):
    """A protected or non-mineable source was requested for review."""


class ReviewConflictError(ReviewLabError):
    """An optimistic-concurrency or idempotency conflict occurred."""


class ReviewStateError(ReviewLabError):
    """The requested lifecycle transition is not currently permitted."""


class ReviewIntegrityError(ReviewLabError):
    """A published label-set artifact failed checksum or manifest verification."""


__all__ = [
    "ReviewConflictError",
    "ReviewEligibilityError",
    "ReviewIntegrityError",
    "ReviewLabError",
    "ReviewStateError",
    "ReviewValidationError",
]
