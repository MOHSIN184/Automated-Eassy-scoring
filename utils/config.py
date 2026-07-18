"""Typed access to YAML configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Small immutable wrapper supporting dotted configuration keys."""

    values: Mapping[str, Any]
    source: Path

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.values
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    def require(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise ConfigError(f"Required configuration key is missing: {key}")
        return value

    def section(self, key: str) -> Mapping[str, Any]:
        value = self.require(key)
        if not isinstance(value, Mapping):
            raise ConfigError(f"Configuration key '{key}' must be a mapping")
        return value


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and minimally validate a YAML configuration file."""
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, Mapping):
        raise ConfigError("The YAML root must be a mapping")
    return Config(values=values, source=config_path)


def resolve_path(config: Config, key: str) -> Path:
    """Resolve a configured path relative to the configuration file."""
    path = Path(str(config.require(key)))
    return path if path.is_absolute() else (config.source.parent / path).resolve()
