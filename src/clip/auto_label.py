"""Auto-label images with CLIP and organize into folders."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from ..config import DEFAULT_CLIP_MODEL, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from ..labels import DISPLAY_LABELS, ROOM_LABELS, ZERO_SHOT_PROMPTS
from ..utils import ensure_dir
from .loader import load_clip


class AutoLabeler:
    """
    Clasifica imágenes con CLIP y las organiza en carpetas por clase.

    Atributos:
        model: modelo CLIP cargado
        processor: processor de CLIP
        device: dispositivo (CPU/GPU)
        labels: lista de etiquetas de clase
        prompts: prompts de texto para cada clase
        stats: estadísticas de la clasificación
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        confidence_threshold: float = 0.45,
        margin_threshold: float = 0.12,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

        self.model, self.processor, self.device = load_clip(model_name)
        self.labels = list(ZERO_SHOT_PROMPTS.keys())
        self.prompts = [ZERO_SHOT_PROMPTS[label] for label in self.labels]
        self.stats = {"classified": 0, "review": 0, "skipped": 0}

    def classify_image(self, image_path: Path) -> tuple[str, float, float]:
        """
        Clasifica una imagen y retorna (label, probability, margin).
        """
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                inputs = self.processor(
                    text=self.prompts, images=rgb_image, return_tensors="pt", padding=True
                )
        except (UnidentifiedImageError, OSError):
            return "", 0.0, 0.0

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()

        ranked_indices = probabilities.argsort()[::-1]
        best_index = int(ranked_indices[0])
        top_prob = float(probabilities[best_index])
        second_prob = float(probabilities[int(ranked_indices[1])]) if len(ranked_indices) > 1 else 0.0
        margin = top_prob - second_prob

        return self.labels[best_index], top_prob, margin

    def label_images(
        self,
        source_dir: str | Path,
        output_dir: str | Path = "data/raw",
        review_dir: str | Path = "data/pending_review",
        report_csv: str | Path = "data/auto_label_report.csv",
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        Clasifica imágenes y las mueve a carpetas por clase o a review.
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        review_path = Path(review_dir)
        report_path = Path(report_csv)

        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = sorted(
            p for p in source_path.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        )

        if not image_paths:
            print(f"No images found in {source_path}")
            return self.stats

        self.stats = {"classified": 0, "review": 0, "skipped": 0}
        rows = []

        for image_path in tqdm(image_paths, desc="Classifying images"):
            label, prob, margin = self.classify_image(image_path)

            if not label:
                self.stats["skipped"] += 1
                continue

            review_needed = prob < self.confidence_threshold or margin < self.margin_threshold

            row = {
                "source_path": str(image_path),
                "predicted_label": label,
                "display_label": DISPLAY_LABELS[label],
                "probability": prob,
                "margin": margin,
                "review_needed": review_needed,
                "manual_label": "",
            }

            if review_needed:
                dest_dir = review_path
                self.stats["review"] += 1
                row["destination"] = str(review_path / image_path.name)
            else:
                dest_dir = output_path / label
                self.stats["classified"] += 1
                row["destination"] = str(dest_dir / image_path.name)

            if not dry_run:
                ensure_dir(dest_dir)
                dest_file = dest_dir / image_path.name
                if dest_file.exists():
                    stem = dest_file.stem
                    suffix = dest_file.suffix
                    dest_file = dest_dir / f"{stem}_{image_path.parent.name}{suffix}"
                shutil.move(str(image_path), str(dest_file))

            rows.append(row)

        ensure_dir(report_path.parent)
        pd.DataFrame(rows).to_csv(report_path, index=False)

        self._print_results(report_path, dry_run)
        return self.stats

    def _print_results(self, report_path: Path, dry_run: bool):
        """Imprime resumen de resultados."""
        print(f"\nResults:")
        print(f"  Classified: {self.stats['classified']}")
        print(f"  Review needed: {self.stats['review']}")
        print(f"  Skipped: {self.stats['skipped']}")
        print(f"  Report: {report_path}")

        if not dry_run:
            print(f"\nClassified images organized by class folder")
            print(f"Review images in pending_review folder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label images with CLIP and organize into folders.")
    parser.add_argument("--source_dir", default="data/scrapped", help="Directory with images to classify.")
    parser.add_argument("--output_dir", default="data/raw", help="Directory for classified images.")
    parser.add_argument("--review_dir", default="data/pending_review", help="Directory for review images.")
    parser.add_argument("--model_name", default=DEFAULT_CLIP_MODEL, help="Hugging Face CLIP model name.")
    parser.add_argument("--confidence_threshold", type=float, default=0.45, help="Mark for review below this probability.")
    parser.add_argument("--margin_threshold", type=float, default=0.12, help="Mark for review below this margin.")
    parser.add_argument("--report_csv", default="data/auto_label_report.csv", help="CSV path for report.")
    parser.add_argument("--dry_run", action="store_true", help="Only classify without moving files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labeler = AutoLabeler(
        model_name=args.model_name,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )
    labeler.label_images(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        review_dir=args.review_dir,
        report_csv=args.report_csv,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
