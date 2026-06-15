"""
Bounding box annotation utilities for YOLO detections.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont


# Color palette for different furniture classes
COLORS = [
    "#2196F3",  # Blue
    "#4CAF50",  # Green
    "#FF9800",  # Orange
    "#E91E63",  # Pink
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#FF5722",  # Deep Orange
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
    "#8BC34A",  # Light Green
    "#FFC107",  # Amber
    "#3F51B5",  # Indigo
    "#009688",  # Teal
    "#F44336",  # Red
    "#CDDC39",  # Lime
    "#03A9F4",  # Light Blue
]


def get_color(class_name: str, class_colors: dict[str, str] | None = None) -> str:
    """Get a consistent color for a class name."""
    if class_colors and class_name in class_colors:
        return class_colors[class_name]

    hash_val = sum(ord(c) for c in class_name)
    return COLORS[hash_val % len(COLORS)]


def annotate_image_pil(
    image_path: str | Path,
    detections: list[dict],
    output_path: str | Path | None = None,
    show_labels: bool = True,
    show_confidence: bool = True,
    line_width: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes on an image using PIL.

    Args:
        image_path: Path to the input image.
        detections: List of detections from FurnitureDetector.detect().
        output_path: Optional path to save the annotated image.
        show_labels: Whether to show class labels.
        show_confidence: Whether to show confidence scores.
        line_width: Width of bounding box lines.

    Returns:
        Annotated PIL Image.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    class_colors: dict[str, str] = {}

    for det in detections:
        if "error" in det:
            continue

        cls_name = det["class_name"]
        conf = det["confidence"]
        bbox = det["bbox"]

        color = get_color(cls_name, class_colors)
        if cls_name not in class_colors:
            class_colors[cls_name] = color

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        if show_labels or show_confidence:
            label_parts = []
            if show_labels:
                label_parts.append(cls_name)
            if show_confidence:
                label_parts.append(f"{conf:.0%}")

            label = " ".join(label_parts)

            bbox_text = draw.textbbox((0, 0), label, font=font_small)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            text_y = max(0, y1 - text_height - 4)
            draw.rectangle(
                [x1, text_y, x1 + text_width + 6, text_y + text_height + 4],
                fill=color,
            )
            draw.text((x1 + 3, text_y + 2), label, fill="white", font=font_small)

    if output_path:
        image.save(output_path)

    return image


def annotate_image_matplotlib(
    image_path: str | Path,
    detections: list[dict],
    output_path: str | Path,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Draw bounding boxes using matplotlib (for saved figures).

    Args:
        image_path: Path to the input image.
        detections: List of detections from FurnitureDetector.detect().
        output_path: Path to save the annotated figure.
        title: Optional title for the figure.
        figsize: Figure size (width, height).
    """
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    class_colors: dict[str, str] = {}

    for det in detections:
        if "error" in det:
            continue

        cls_name = det["class_name"]
        conf = det["confidence"]
        bbox = det["bbox"]

        color = get_color(cls_name, class_colors)
        if cls_name not in class_colors:
            class_colors[cls_name] = color

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        label = f"{cls_name} {conf:.0%}"
        ax.text(
            x1, y1 - 5,
            label,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
        )

    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_comparison_grid(
    image_paths: list[str | Path],
    detections_list: list[list[dict]],
    output_path: str | Path,
    cols: int = 3,
    figsize: tuple[int, int] = (16, 12),
) -> None:
    """
    Create a grid of annotated images for comparison.

    Args:
        image_paths: List of image paths.
        detections_list: List of detection lists (one per image).
        output_path: Path to save the grid figure.
        cols: Number of columns in the grid.
        figsize: Figure size.
    """
    n_images = len(image_paths)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if idx < n_images:
            image = Image.open(image_paths[idx]).convert("RGB")
            ax.imshow(image)

            for det in detections_list[idx]:
                if "error" in det:
                    continue

                cls_name = det["class_name"]
                conf = det["confidence"]
                bbox = det["bbox"]
                color = get_color(cls_name)

                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 5,
                    f"{cls_name} {conf:.0%}",
                    color="white",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                )

            ax.set_title(Path(image_paths[idx]).name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
