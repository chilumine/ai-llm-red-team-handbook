"""Configuration module."""

from .loader import load_config, load_config_file
from .schema import (
    AttackConfig,
    AuthorizationConfig,
    Config,
    ReportingConfig,
    TargetConfig,
)

__all__ = [
    "Config",
    "TargetConfig",
    "AttackConfig",
    "ReportingConfig",
    "AuthorizationConfig",
    "load_config",
    "load_config_file",
]
