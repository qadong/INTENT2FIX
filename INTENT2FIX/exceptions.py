from typing import Any, List

from INTENT2FIX.completion.model import Usage


class INTENT2FIXError(Exception):
    """Base exception class for all INTENT2FIX exceptions."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class RuntimeError(INTENT2FIXError):
    """Exception raised when an operation encounters a fundamental error that prevents further execution."""

    pass


class InfrastructureRuntimeError(RuntimeError):
    """Raised when the execution environment is no longer usable."""

    pass


class RejectError(INTENT2FIXError):
    """Exception raised when an operation should be rejected but the flow can continue."""

    pass


class CompletionError(INTENT2FIXError):
    """Base exception for completion-related errors."""

    def __init__(
        self,
        message: str,
        last_completion: Any | None = None,
        messages: List[dict] | None = None,
        accumulated_usage: Usage | None = None,
    ):
        super().__init__(message)
        self.last_completion = last_completion
        self.messages = messages or []
        self.accumulated_usage = accumulated_usage


class CompletionRuntimeError(RuntimeError, CompletionError):
    """Exception raised when completion encounters an unrecoverable error."""

    pass


class CompletionRejectError(RejectError, CompletionError):
    """Raised when a completion response is rejected due to validation failure."""

    pass


class CompletionValidationError(INTENT2FIXError):
    """Exception raised when completion encounters a validation error."""

    pass
