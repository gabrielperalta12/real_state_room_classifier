"""
Predecir clase de habitación usando modelo entrenado.

Uso:
    python -m src.ml.predict --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from ..clip.loader import load_clip
from ..config import DEFAULT_CLIP_MODEL
from ..labels import DISPLAY_LABELS
from ..utils import validate_file


def predict_image(
    image_path: str | Path,
    model_path: str | Path = "outputs/comparison/models/clip/best_classifier.joblib",
    clip_model_name: str = DEFAULT_CLIP_MODEL,
    top_k: int = 3,
) -> list[dict]:
    """
    Predice la clase de una imagen usando CLIP + ML classifier.

    Returns:
        Lista de predicciones ordenadas por probabilidad.
    """
    image_path = Path(image_path)
    validate_file(image_path)

    # Cargar modelo ML
    artifact = joblib.load(model_path)
    classifier = artifact["model"]
    classes = artifact["classes"]
    label_encoder = artifact.get("label_encoder", None)

    # Cargar CLIP
    model, processor, device = load_clip(clip_model_name)

    # Extraer embedding
    import torch
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        inputs = processor(images=rgb_image, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        # Use vision_model directly for compatibility with transformers v5
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs[1]  # pooler_output
        embedding = model.visual_projection(pooled_output)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    embedding_np = embedding.squeeze(0).cpu().numpy().reshape(1, -1)

    # Predecir
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(embedding_np)[0]
        indices = probabilities.argsort()[::-1][:top_k]

        results = []
        for idx in indices:
            if label_encoder is not None:
                label = label_encoder.inverse_transform([idx])[0]
            else:
                label = classes[idx]
            results.append({
                "label": label,
                "display_label": DISPLAY_LABELS.get(label, label),
                "probability": round(float(probabilities[idx]), 4),
            })
        return results
    else:
        prediction = classifier.predict(embedding_np)[0]
        if label_encoder is not None:
            prediction = label_encoder.inverse_transform([prediction])[0]
        return [{
            "label": prediction,
            "display_label": DISPLAY_LABELS.get(prediction, prediction),
            "probability": 1.0,
        }]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict room class with trained ML model.")
    parser.add_argument("--image", required=True, help="Path to image.")
    parser.add_argument("--model_path", default="outputs/comparison/models/clip/best_classifier.joblib", help="Trained model path.")
    parser.add_argument("--clip_model", default=DEFAULT_CLIP_MODEL, help="CLIP model name.")
    parser.add_argument("--top_k", type=int, default=3, help="Top k predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_image(args.image, args.model_path, args.clip_model, args.top_k)

    print(f"\nPredicciones para: {args.image}")
    print("-" * 40)
    for i, pred in enumerate(predictions, 1):
        print(f"#{i} {pred['display_label']}: {pred['probability']:.1%}")


if __name__ == "__main__":
    main()
