"""Typed failures shared by the core, SDK, and future transport adapters."""


class MathForgeError(Exception):
    """Base error with a stable, machine-readable code."""

    code = "mathforge_error"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class InvalidInput(MathForgeError, ValueError):
    code = "invalid_input"


class UnsupportedOperation(MathForgeError):
    code = "unsupported_operation"


class SerializationError(MathForgeError, ValueError):
    code = "invalid_serialization"


class OperationError(MathForgeError):
    """An SDK convenience call failed; ``result`` retains its full contract."""

    def __init__(self, result):
        self.result = result
        error = result.error
        super().__init__(
            error.message if error else "The operation did not produce a value.",
            code=error.code if error else "operation_failed",
        )
