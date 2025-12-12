"""Custom exception hierarchy for the LLM Gateway."""

from __future__ import annotations


class GatewayError(Exception):
    """Base exception for all gateway errors."""

    def __init__(self, message: str = "", *, provider: str = "", model: str = "") -> None:
        self.provider = provider
        self.model = model
        super().__init__(message)


class ProviderError(GatewayError):
    """Upstream provider returned a non-retryable error."""

    def __init__(
        self,
        message: str = "Provider error",
        *,
        provider: str = "",
        model: str = "",
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message, provider=provider, model=model)


class ProviderUnavailable(GatewayError):
    """Provider is temporarily unreachable (5xx, timeout, connection refused)."""

    def __init__(
        self,
        message: str = "Provider unavailable",
        *,
        provider: str = "",
        model: str = "",
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider, model=model)


class RateLimitExceeded(GatewayError):
    """Rate limit hit — either gateway-level or upstream 429."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        provider: str = "",
        model: str = "",
        retry_after: float | None = None,
        limit: int | None = None,
    ) -> None:
        self.retry_after = retry_after
        self.limit = limit
        super().__init__(message, provider=provider, model=model)


class AuthenticationError(GatewayError):
    """Invalid or missing API key / credentials."""

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        provider: str = "",
        model: str = "",
    ) -> None:
        super().__init__(message, provider=provider, model=model)


class ModelNotFound(GatewayError):
    """Requested model does not exist or is not enabled."""

    def __init__(
        self,
        message: str = "Model not found",
        *,
        provider: str = "",
        model: str = "",
    ) -> None:
        super().__init__(message, provider=provider, model=model)


class BudgetExceeded(GatewayError):
    """Spending limit or token budget has been exceeded."""

    def __init__(
        self,
        message: str = "Budget exceeded",
        *,
        provider: str = "",
        model: str = "",
        budget_limit: float | None = None,
        current_spend: float | None = None,
    ) -> None:
        self.budget_limit = budget_limit
        self.current_spend = current_spend
        super().__init__(message, provider=provider, model=model)
