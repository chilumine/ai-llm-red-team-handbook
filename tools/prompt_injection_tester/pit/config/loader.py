"""Configuration file loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from .schema import Config


def load_config_file(config_path: Path) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Expand environment variables in token
    if "target" in data and "token" in data["target"]:
        token = data["target"]["token"]
        if isinstance(token, str) and token.startswith("${") and token.endswith("}"):
            env_var = token[2:-1]
            data["target"]["token"] = os.getenv(env_var)

    try:
        return Config(**data)
    except Exception as e:
        raise ValueError(f"Invalid configuration file: {e}") from e


def load_config(
    config_path: Optional[Path],
    target_url: Optional[str] = None,
    **cli_overrides,
) -> Config:
    """
    Load configuration from file or CLI arguments.

    Args:
        config_path: Path to configuration file (optional)
        target_url: Target URL from CLI (optional)
        **cli_overrides: CLI argument overrides

    Returns:
        Merged configuration

    Raises:
        ValueError: If neither config_path nor target_url is provided
    """
    if config_path:
        config = load_config_file(config_path)

        # Apply CLI overrides
        if target_url:
            config.target.url = target_url
        if "token" in cli_overrides and cli_overrides["token"]:
            config.target.token = cli_overrides["token"]

        return config

    elif target_url:
        # Build config from CLI args
        return Config.from_cli_args(target_url=target_url, **cli_overrides)

    else:
        raise ValueError("Either --config or target URL must be provided")
