"""Utility functions for image processing and dataset management."""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_dir(path: str | Path) -> Path:
    """Validate that a directory exists."""
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")
    return directory
