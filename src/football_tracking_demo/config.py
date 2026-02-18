"""Configuration management for football tracking demo."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required sections are missing.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    required_sections = ["video", "detection", "tracking", "output"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    return config


def get_detection_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract detection-related configuration."""
    return config.get("detection", {})


def get_tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract tracking-related configuration."""
    return config.get("tracking", {})


def get_hud_mask_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract HUD masking configuration."""
    return config.get("hud_mask", {})


def get_detection_shape_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract detection shape filter configuration (size and aspect ratio)."""
    return config.get("detection_shape_filter", {})


def get_playing_field_mask_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract playing field color mask configuration."""
    return config.get("playing_field_mask", {})


def get_visualization_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract visualization configuration."""
    return config.get("visualization", {})


def get_output_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract output paths configuration."""
    return config.get("output", {})
