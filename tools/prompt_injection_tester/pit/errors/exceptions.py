"""Custom exception hierarchy for PIT."""

from __future__ import annotations


class PitError(Exception):
    """Base exception for all PIT errors."""

    pass


# Configuration Errors
class ConfigError(PitError):
    """Base configuration error."""

    pass


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""

    pass


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""

    pass


# Target Errors
class TargetError(PitError):
    """Base target error."""

    pass


class TargetUnreachableError(TargetError):
    """Target is unreachable."""

    def __init__(self, url: str, reason: str = ""):
        self.url = url
        self.reason = reason
        super().__init__(f"Target unreachable: {url}")


class AuthenticationError(TargetError):
    """Authentication failed."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Authentication failed: {status_code}")


class RateLimitError(TargetError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded, retry after {retry_after}s")


# Discovery Errors
class DiscoveryError(PitError):
    """Base discovery error."""

    pass


class NoEndpointsFoundError(DiscoveryError):
    """No endpoints found during discovery."""

    pass


class ScanFailedError(DiscoveryError):
    """Discovery scan failed."""

    pass


# Attack Errors
class AttackError(PitError):
    """Base attack error."""

    pass


class PatternLoadError(AttackError):
    """Failed to load attack pattern."""

    pass


class ExecutionError(AttackError):
    """Attack execution failed."""

    pass


# Verification Errors
class VerificationError(PitError):
    """Base verification error."""

    pass


class DetectionFailedError(VerificationError):
    """Detection/verification failed."""

    pass


# Reporting Errors
class ReportingError(PitError):
    """Base reporting error."""

    pass


class FormatError(ReportingError):
    """Invalid report format."""

    pass


class FileWriteError(ReportingError):
    """Failed to write report file."""

    pass
