"""Project-specific exceptions with deliberately explicit failure messages."""


class VGCFError(Exception):
    """Base class for expected project failures."""


class SchemaError(VGCFError):
    """Raised when model JSON does not satisfy the logic schema."""


class DatasetSchemaError(VGCFError):
    """Raised instead of guessing the meaning of unknown dataset fields."""


class ModelError(VGCFError):
    """Raised when a model request exhausts its retries."""


class LogicError(VGCFError):
    """Raised for an invalid or inconsistent logic program."""
