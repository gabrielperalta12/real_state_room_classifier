"""
Comparación de modelos visuales: CLIP vs DINOv2 vs Place365 vs CLIP Zero-shot.

Extrae embeddings con todos los modelos, entrena clasificadores y compara resultados.

Uso:
    # Comparar todos los modelos
    python -m src.ml.compare_models --data_dir data/splits --output_dir outputs/comparison

    # Incluir Place365 como baseline
    python -m src.ml.compare_models --data_dir data/splits --output_dir outputs/comparison --models clip dinov2 place365 clip_zeroshot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .extract_embeddings import extract_embeddings, MODEL_CONFIGS
from .train import train_classifier
from ..config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_MODEL_DIR, DEFAULT_CLIP_MODEL
from ..labels import DISPLAY_LABELS, ROOM_LABELS, ZERO_SHOT_PROMPTS
from ..clip.loader import load_clip
from ..utils import ensure_dir, validate_dir

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib


def evaluate_zeroshot(
    test_dir: Path,
    model_name: str = DEFAULT_CLIP_MODEL,
) -> dict:
    """
    Evalúa CLIP zero-shot en un directorio de test.
    
    Returns:
        dict con accuracy y F1 score
    """
    import torch
    
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Collect test images
    true_labels = []
    image_paths = []
    
    for class_name in ROOM_LABELS:
        class_dir = test_dir / class_name
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                true_labels.append(class_name)
                image_paths.append(img_path)
    
    if not image_paths:
        return {"accuracy": 0.0, "f1_weighted": 0.0}
    
    # Load model
    model, processor, device = load_clip(model_name)
    labels = list(ZERO_SHOT_PROMPTS.keys())
    prompts = [ZERO_SHOT_PROMPTS[label] for label in labels]
    
    # Classify
    pred_labels = []
    
    for image_path in tqdm(image_paths, desc="Zero-shot eval"):
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
        except Exception:
            pred_labels.append("")
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    prec = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    rec = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    
    return {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "total_images": len(image_paths),
    }


def compare_models(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "comparison",
    models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compara múltiples modelos visuales.
    
    Steps:
      1. Extrae embeddings para cada modelo (excepto clip_zeroshot)
      2. Entrena clasificadores
      3. Evalúa zero-shot si se solicita
      4. Compara métricas
    """
    if models is None:
        models = ["clip", "dinov2", "place365"]
    
    data_path = validate_dir(data_dir)
    output_path = Path(output_dir)
    embeddings_base = ensure_dir(output_path / "embeddings")
    models_base = ensure_dir(output_path / "models")
    
    test_dir = data_path / "test"
    if not test_dir.is_dir():
        raise ValueError(f"Test directory not found: {test_dir}")
    
    all_results = []
    
    for model_key in models:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {model_key.upper()}")
        print(f"{'='*60}")
        
        if model_key == "clip_zeroshot":
            # Zero-shot evaluation (no training needed)
            result = evaluate_zeroshot(test_dir, DEFAULT_CLIP_MODEL)
            all_results.append({
                "model": "clip_zeroshot",
                "model_name": "CLIP Zero-shot",
                "best_classifier": "none",
                "f1_weighted": result["f1_weighted"],
                "accuracy": result["accuracy"],
                "precision_weighted": result["precision_weighted"],
                "recall_weighted": result["recall_weighted"],
            })
        else:
            # Extract embeddings
            extract_embeddings(data_path, embeddings_base, model_key)
            
            # Train classifier
            emb_dir = embeddings_base / model_key
            model_dir = models_base / model_key
            result = train_classifier(
                input_dir=emb_dir,
                output_dir=model_dir,
            )
            
            # Compute full metrics on test set
            X_test = np.load(emb_dir / "test" / "X_embeddings.npy")
            y_test = np.load(emb_dir / "test" / "y_labels.npy")
            
            artifact = joblib.load(result["model_path"])
            clf = artifact["model"]
            scaler = artifact.get("scaler")
            
            if scaler is not None:
                X_test = scaler.transform(X_test)
            
            y_pred = clf.predict(X_test)
            
            # Convert numeric predictions if needed
            le = artifact.get("label_encoder")
            if le is not None and not np.issubdtype(y_pred.dtype, np.str_):
                try:
                    y_pred = le.inverse_transform(y_pred.astype(int))
                except Exception:
                    pass
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            
            all_results.append({
                "model": model_key,
                "model_name": MODEL_CONFIGS[model_key]["name"],
                "best_classifier": result["best_model"],
                "f1_weighted": round(f1, 4),
                "accuracy": round(acc, 4),
                "precision_weighted": round(prec, 4),
                "recall_weighted": round(rec, 4),
            })
    
    # Create comparison
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values("f1_weighted", ascending=False)
    
    # Save comparison
    comparison_path = output_path / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    # Plot comparison
    plot_comparison(comparison_df, output_path / "comparison.png")
    
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved to: {output_path}")
    
    return comparison_df


def plot_comparison(results: pd.DataFrame, output_path: Path):
    """Genera gráfico de comparación."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(results))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]
    bars = ax.bar(x, results["f1_weighted"], color=colors[:len(results)])
    
    # Create labels
    labels = []
    for _, row in results.iterrows():
        if row["model"] == "clip_zeroshot":
            labels.append("CLIP\nZero-shot")
        elif row["model"] == "place365":
            labels.append("Place365\n(ResNet50)")
        elif row["model"] == "dinov2":
            labels.append("DINOv2\n(ViT-B/14)")
        else:
            labels.append(f"{row['model']}\n({row['best_classifier']})")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 Weighted Score")
    ax.set_title("Visual Model Comparison")
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, (_, row) in zip(bars, results.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{row['f1_weighted']:.3f}", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare visual models for room classification.")
    parser.add_argument("--data_dir", default=str(Path("data/splits")), help="Directory with train/test splits.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "comparison"), help="Output directory.")
    parser.add_argument("--models", nargs="+", default=["clip", "dinov2", "place365"], 
                       choices=["clip", "clip-large", "dinov2", "place365", "clip_zeroshot"],
                       help="Models to compare.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_models(args.data_dir, args.output_dir, args.models)


if __name__ == "__main__":
    main()
