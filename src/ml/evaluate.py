"""
Evaluación de modelos pre-entrenados con imágenes de test.

Carga un modelo .joblib ya entrenado, extrae embeddings de imágenes de test
y genera métricas + confusion matrices. NO re-entrena.

Uso:
    # Evaluar el mejor modelo (CLIP + SVM)
    python -m src.ml.evaluate --model_dir outputs/comparison/models/clip

    # Evaluar DINOv2
    python -m src.ml.evaluate --model_dir outputs/comparison/models/dinov2

    # Evaluar Place365
    python -m src.ml.evaluate --model_dir outputs/comparison/models/place365
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from ..config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from ..labels import DISPLAY_LABELS
from ..utils import ensure_dir


def evaluate_pretrained_model(
    model_dir: str | Path,
    test_data: str | Path = "data/splits/test",
) -> dict:
    """
    Evalúa un modelo ya entrenado con imágenes de test.

    No re-entrena. Solo carga el modelo y evalúa.
    Los gráficos y métricas se guardan junto al modelo en outputs/comparison/models/{model}/.

    Args:
        model_dir: Directorio con best_classifier.joblib (ej: outputs/comparison/models/clip).
        test_data: Directorio con imágenes de test organizadas por clase.
    """
    import torch
    from PIL import Image
    from ..clip.loader import load_clip

    model_path = Path(model_dir) / "best_classifier.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return {}

    # ── Cargar modelo entrenado ──
    artifact = joblib.load(model_path)
    classifier = artifact["model"]
    label_encoder = artifact.get("label_encoder", None)

    print(f"Loaded model: {model_path}")
    print(f"Classifier: {type(classifier).__name__}")

    # ── Cargar CLIP para extraer embeddings ──
    clip_model, processor, device = load_clip("openai/clip-vit-base-patch32")

    # ── Extraer embeddings de test ──
    test_data_path = Path(test_data)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    X_test = []
    y_true = []

    for class_dir in sorted(test_data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() not in image_extensions:
                continue
            try:
                with Image.open(img_path) as img:
                    rgb_img = img.convert("RGB")
                    inputs = processor(images=rgb_img, return_tensors="pt")

                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
                    pooled_output = vision_outputs[1]
                    embedding = clip_model.visual_projection(pooled_output)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                X_test.append(embedding.squeeze(0).cpu().numpy())
                y_true.append(class_name)
            except Exception as e:
                print(f"Error: {img_path}: {e}")

    X_test = np.array(X_test)
    y_true = np.array(y_true)

    print(f"Test images: {len(X_test)}")
    print(f"Classes: {len(set(y_true))}")

    # ── Predecir ──
    y_pred = classifier.predict(X_test)
    if label_encoder is not None:
        y_pred_labels = label_encoder.inverse_transform(y_pred)
    else:
        y_pred_labels = y_pred

    # ── Métricas ──
    accuracy = accuracy_score(y_true, y_pred_labels)
    report = classification_report(y_true, y_pred_labels, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred_labels, zero_division=0)

    # ── Guardar resultados junto al modelo ──
    output_path = Path(model_dir)
    metrics_path = ensure_dir(output_path)
    figures_path = ensure_dir(output_path)

    # Reporte completo (no sobreescribe el de train)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(metrics_path / "test_classification_report.csv")

    # Per-class metrics CSV (solo clases, sin summary rows)
    summary_rows = {"accuracy", "macro avg", "weighted avg"}
    class_rows = [idx for idx in report_df.index if idx not in summary_rows]
    per_class_df = report_df.loc[class_rows, ["precision", "recall", "f1-score", "support"]]
    per_class_df.index = [DISPLAY_LABELS.get(idx, idx) for idx in per_class_df.index]
    per_class_df.index.name = "clase"
    per_class_df.to_csv(metrics_path / "test_per_class_metrics.csv")

    # Confusion matrix (raw)
    display_names = [DISPLAY_LABELS.get(c, c) for c in sorted(set(y_true))]
    cm_path = figures_path / "confusion_matrix.png"
    _plot_confusion_matrix(y_true, y_pred_labels, display_names, "Confusion Matrix", cm_path)
    print(f"  Saved: {cm_path}")

    # Confusion matrix (normalized)
    cm_norm_path = figures_path / "confusion_matrix_normalized.png"
    _plot_confusion_matrix(y_true, y_pred_labels, display_names, "Confusion Matrix (Normalized)", cm_norm_path, normalize=True)
    print(f"  Saved: {cm_norm_path}")

    # Per-class metrics bar chart
    metrics_path_fig = figures_path / "per_class_metrics.png"
    _plot_per_class_metrics(report_df, "Test Evaluation", metrics_path_fig)
    print(f"  Saved: {metrics_path_fig}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Test images: {len(X_test)}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"\n{report_text}")

    # Per-class metrics table
    print("\nPer-class metrics:")
    print(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 62)
    summary_rows = {"accuracy", "macro avg", "weighted avg"}
    class_rows = [idx for idx in report_df.index if idx not in summary_rows]
    for cls in class_rows:
        display = DISPLAY_LABELS.get(cls, cls)
        p = report_df.loc[cls, "precision"]
        r = report_df.loc[cls, "recall"]
        f = report_df.loc[cls, "f1-score"]
        s = int(report_df.loc[cls, "support"])
        print(f"{display:<22} {p:>10.3f} {r:>10.3f} {f:>10.3f} {s:>10}")

    # Best and worst classes
    f1_per_class = {cls: report_df.loc[cls, "f1-score"] for cls in class_rows}
    best_cls = max(f1_per_class, key=f1_per_class.get)
    worst_cls = min(f1_per_class, key=f1_per_class.get)
    print(f"\nBest class:  {DISPLAY_LABELS.get(best_cls, best_cls)} (F1={f1_per_class[best_cls]:.3f})")
    print(f"Worst class: {DISPLAY_LABELS.get(worst_cls, worst_cls)} (F1={f1_per_class[worst_cls]:.3f})")

    # Support distribution
    print("\nSupport distribution (test samples per class):")
    for cls in sorted(class_rows, key=lambda c: report_df.loc[c, "support"], reverse=True):
        display = DISPLAY_LABELS.get(cls, cls)
        s = int(report_df.loc[cls, "support"])
        bar = "█" * (s // 5)
        print(f"  {display:<22} {s:>5} {bar}")

    print(f"\nResults saved to: {output_path}")

    return {"accuracy": accuracy, "report": report, "report_df": report_df}


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
    normalize: bool = False,
) -> None:
    """Genera y guarda una confusion matrix como heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        xlabel = "Predicted (normalized)"
    else:
        fmt = "d"
        xlabel = "Predicted"

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0 if normalize else None,
        vmax=1 if normalize else None,
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_metrics(
    report_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
) -> None:
    """Genera bar chart de precision, recall, F1 por clase."""
    summary_rows = {"accuracy", "macro avg", "weighted avg"}
    class_rows = [idx for idx in report_df.index if idx not in summary_rows]

    if not class_rows:
        return

    metrics_data = report_df.loc[class_rows, ["precision", "recall", "f1-score"]]
    display_names = [DISPLAY_LABELS.get(idx, idx) for idx in metrics_data.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(display_names))
    width = 0.25

    bars1 = ax.bar(x - width, metrics_data["precision"], width, label="Precision", color="#2196F3")
    bars2 = ax.bar(x, metrics_data["recall"], width, label="Recall", color="#4CAF50")
    bars3 = ax.bar(x + width, metrics_data["f1-score"], width, label="F1-Score", color="#FF9800")

    ax.set_xlabel("Clase", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Métricas por Clase - {model_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained model on test images (no retraining)."
    )
    parser.add_argument(
        "--model_dir",
        default="outputs/comparison/models/clip",
        help="Directory with best_classifier.joblib (from compare_models).",
    )
    parser.add_argument(
        "--test_data",
        default="data/splits/test",
        help="Directory with test images organized by class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_pretrained_model(args.model_dir, args.test_data)


if __name__ == "__main__":
    main()
