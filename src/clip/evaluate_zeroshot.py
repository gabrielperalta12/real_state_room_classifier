"""
Evalúa la eficacia del zero-shot con imágenes etiquetadas.

Uso:
    python -m src.clip.evaluate_zeroshot --data_dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..config import DEFAULT_CLIP_MODEL, DEFAULT_OUTPUT_DIR
from ..labels import DISPLAY_LABELS, ROOM_LABELS, ZERO_SHOT_PROMPTS
from ..clip.loader import load_clip


def evaluate_zeroshot(
    data_dir: str | Path,
    model_name: str = DEFAULT_CLIP_MODEL,
    output_csv: str | Path = DEFAULT_OUTPUT_DIR / "zeroshot_evaluation.csv",
) -> dict:
    """
    Evalúa zero-shot usando imágenes etiquetadas en carpetas por clase.

    Estructura esperada:
        data/raw/
            sala/
                img1.jpg
                img2.jpg
            cocina/
                img1.jpg
            ...

    Returns:
        Diccionario con métricas de evaluación.
    """
    from pathlib import Path
    import torch

    data_path = Path(data_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Recopilar imágenes por clase
    true_labels = []
    image_paths = []
    class_counts = {}

    for class_name in ROOM_LABELS:
        class_dir = data_path / class_name
        if not class_dir.is_dir():
            continue

        count = 0
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                true_labels.append(class_name)
                image_paths.append(img_path)
                count += 1

        class_counts[class_name] = count

    if not image_paths:
        print(f"No images found in {data_path}")
        return {}

    print(f"Imágenes por clase:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {DISPLAY_LABELS.get(cls, cls)}: {count}")
    print(f"  Total: {len(image_paths)}")
    print()

    # Cargar modelo
    model, processor, device = load_clip(model_name)
    labels = list(ZERO_SHOT_PROMPTS.keys())
    prompts = [ZERO_SHOT_PROMPTS[label] for label in labels]

    # Clasificar
    pred_labels = []
    confidences = []

    for image_path in tqdm(image_paths, desc="Evaluando"):
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                inputs = processor(text=prompts, images=rgb_image, return_tensors="pt", padding=True)

            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()

            best_idx = probabilities.argmax()
            pred_labels.append(labels[best_idx])
            confidences.append(float(probabilities[best_idx]))
        except Exception as e:
            pred_labels.append("")
            confidences.append(0.0)

    # Calcular métricas
    results_df = pd.DataFrame({
        "image_path": [str(p) for p in image_paths],
        "true_label": true_labels,
        "pred_label": pred_labels,
        "confidence": confidences,
        "correct": [t == p for t, p in zip(true_labels, pred_labels)],
    })

    # Accuracy por clase
    per_class = []
    for cls in ROOM_LABELS:
        if cls not in class_counts:
            continue
        cls_df = results_df[results_df["true_label"] == cls]
        correct = cls_df["correct"].sum()
        total = len(cls_df)
        accuracy = correct / total if total > 0 else 0
        avg_conf = cls_df["confidence"].mean() if total > 0 else 0

        per_class.append({
            "class": cls,
            "display_label": DISPLAY_LABELS.get(cls, cls),
            "total": total,
            "correct": int(correct),
            "accuracy": round(accuracy, 4),
            "avg_confidence": round(avg_conf, 4),
        })

    # Accuracy general
    overall_accuracy = results_df["correct"].sum() / len(results_df)

    # Guardar resultados
    output_path = Path(output_csv)
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / "detailed_results.csv", index=False)
    pd.DataFrame(per_class).to_csv(output_path / "per_class_metrics.csv", index=False)

    # Imprimir resumen
    print("\n" + "=" * 60)
    print("RESULTADOS DE EVALUACIÓN ZERO-SHOT")
    print("=" * 60)
    print(f"\nAccuracy general: {overall_accuracy:.1%}")
    print(f"\nPor clase:")
    print(f"{'Clase':<20} {'Total':>6} {'Correct':>8} {'Accuracy':>10} {'Confianza':>10}")
    print("-" * 60)
    for row in per_class:
        print(f"{row['display_label']:<20} {row['total']:>6} {row['correct']:>8} {row['accuracy']:>9.1%} {row['avg_confidence']:>9.1%}")

    print(f"\nResultados guardados en: {output_path}")

    return {
        "overall_accuracy": overall_accuracy,
        "per_class": per_class,
        "total_images": len(image_paths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot CLIP on labeled images.")
    parser.add_argument("--data_dir", default="data/raw", help="Directory with class subfolders.")
    parser.add_argument("--model_name", default=DEFAULT_CLIP_MODEL, help="CLIP model name.")
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT_DIR / "zeroshot_evaluation"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_zeroshot(args.data_dir, args.model_name, args.output_csv)


if __name__ == "__main__":
    main()
