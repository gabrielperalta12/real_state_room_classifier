"""
Fine-tune YOLO11 for indoor furniture detection.

Uso:
    python -m src.detection.train_yolo
    python -m src.detection.train_yolo --data data/yolo/combined/data.yaml --epochs 50
    python -m src.detection.train_yolo --model yolo11s.pt --freeze_backbone
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import PROJECT_ROOT, DEFAULT_MODEL_DIR
from ..utils import ensure_dir


def train_yolo(
    data_yaml: str | Path,
    model_name: str = "yolo11n.pt",
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    output_dir: str | Path | None = None,
    freeze_backbone: bool = False,
    project_name: str = "yolo_furniture",
    device: str | None = None,
    patience: int = 20,
    learning_rate: float = 0.01,
) -> Path:
    """
    Train a YOLO11 model for furniture detection.

    Args:
        data_yaml: Path to data.yaml configuration file.
        model_name: Pretrained model name or path (e.g., 'yolo11n.pt').
        epochs: Number of training epochs.
        img_size: Input image size.
        batch_size: Batch size.
        output_dir: Output directory for training runs.
        freeze_backbone: If True, freeze backbone layers for first 10 epochs.
        project_name: Name of the training run.
        device: Device to train on ('cuda', 'cpu', or None for auto).
        patience: Early stopping patience.
        learning_rate: Initial learning rate.

    Returns:
        Path to the directory containing best.pt weights.
    """
    from ultralytics import YOLO

    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    output_path = Path(output_dir) if output_dir else DEFAULT_MODEL_DIR
    ensure_dir(output_path)

    print("=" * 60)
    print("YOLO11 FINE-TUNING FOR FURNITURE DETECTION")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device or 'auto'}")
    print(f"  Freeze backbone: {freeze_backbone}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Patience: {patience}")
    print("=" * 60)

    # Load pretrained model
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)

    # Train with freeze if requested
    if freeze_backbone:
        print("\nStage 1: Training with frozen backbone (10 epochs)...")
        model.train(
            data=str(data_path),
            epochs=10,
            imgsz=img_size,
            batch=batch_size,
            freeze=10,
            lr0=learning_rate,
            patience=patience,
            device=device,
            project=str(output_path),
            name=f"{project_name}_stage1",
            exist_ok=True,
            verbose=True,
        )

        # Load best weights from stage 1
        stage1_weights = output_path / f"{project_name}_stage1" / "weights" / "best.pt"
        if stage1_weights.exists():
            print(f"\nLoading stage 1 best weights: {stage1_weights}")
            model = YOLO(str(stage1_weights))

        print("\nStage 2: Fine-tuning all layers...")
        model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            freeze=0,
            lr0=learning_rate * 0.1,
            patience=patience,
            device=device,
            project=str(output_path),
            name=f"{project_name}_stage2",
            exist_ok=True,
            verbose=True,
        )

        final_weights = output_path / f"{project_name}_stage2" / "weights" / "best.pt"

    else:
        print(f"\nTraining for {epochs} epochs...")
        model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            freeze=0,
            lr0=learning_rate,
            patience=patience,
            device=device,
            project=str(output_path),
            name=project_name,
            exist_ok=True,
            verbose=True,
        )

        final_weights = output_path / project_name / "weights" / "best.pt"

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    if final_weights.exists():
        print(f"\nBest weights saved to: {final_weights}")

        # Load and validate
        print("\nRunning validation...")
        model = YOLO(str(final_weights))
        metrics = model.val(data=str(data_path))

        print(f"\nValidation Results:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
    else:
        print(f"\nWarning: Best weights not found at {final_weights}")
        print("Check training logs for errors.")

    return final_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11 for indoor furniture detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "yolo" / "combined" / "data.yaml"),
        help="Path to data.yaml configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained model name or path (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Input image size (default: 640).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for training runs.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone layers for first 10 epochs (two-stage training).",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="yolo_furniture",
        help="Name of the training run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda', 'cpu', or None for auto).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        freeze_backbone=args.freeze_backbone,
        project_name=args.project_name,
        device=args.device,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
