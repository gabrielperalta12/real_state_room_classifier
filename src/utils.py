"""Utility functions shared across scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from .config import IMAGE_EXTENSIONS


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_file(path: str | Path) -> Path:
    """Validate that a file exists."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path


def validate_dir(path: str | Path) -> Path:
    """Validate that a directory exists."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return directory


def list_image_files(data_dir: str | Path, class_names: Iterable[str] | None = None) -> list[Path]:
    """List supported image files, optionally restricted to class subfolders."""
    root = validate_dir(data_dir)
    if class_names is None:
        candidates = root.rglob("*")
    else:
        candidates = []
        for class_name in class_names:
            class_dir = root / class_name
            if class_dir.is_dir():
                candidates.extend(class_dir.rglob("*"))
    return sorted(path for path in candidates if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(image_path: str | Path) -> Image.Image:
    """Open an image and convert it to RGB."""
    path = validate_file(image_path)
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupted image: {path}") from exc
