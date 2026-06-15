"""CLIP model and processor loading."""

from __future__ import annotations

import torch
from transformers import CLIPModel, CLIPProcessor

from ..config import DEFAULT_CLIP_MODEL


def get_device() -> torch.device:
    """Return CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip(
    model_name: str = DEFAULT_CLIP_MODEL,
    device: torch.device | None = None,
) -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Load CLIP model and processor from Hugging Face."""
    selected_device = device or get_device()
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(selected_device)
    model.eval()
    return model, processor, selected_device
