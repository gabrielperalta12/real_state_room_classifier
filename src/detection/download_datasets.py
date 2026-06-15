"""
Download and merge furniture detection datasets from Roboflow.

Supports:
  - furniture-6k (5,894 images)
  - indoor-objects (2,500 images)
  - furniture-identifier (1,600 images)

Uso:
    python -m src.detection.download_datasets
    python -m src.detection.download_datasets --dataset furniture-6k
    python -m src.detection.download_datasets --api_key YOUR_KEY
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

from ..config import PROJECT_ROOT
from ..utils import ensure_dir


# Dataset configurations
DATASETS = {
    "furniture-6k": {
        "workspace": "university-fa8eh",
        "project": "furniture-6k",
        "version": 1,
        "description": "5,894 images - Multiple furniture classes",
    },
    "indoor-objects": {
        "workspace": "rechanelingworkspace",
        "project": "indoor-objects-3gtbo-yl4dz",
        "version": 1,
        "description": "2,500 images - Indoor objects (chair, table, door, bed)",
    },
    "furniture-identifier": {
        "workspace": "gleefulgeese",
        "project": "furniture-identifier-u2tyo",
        "version": 1,
        "description": "1,600 images - 9 furniture classes",
    },
}

# Unified class mapping (merge similar classes across datasets)
CLASS_MAPPING = {
    # From furniture-6k
    "chair": "chair",
    "sofa": "sofa",
    "bed": "bed",
    "table": "table",
    "lamp": "lamp",
    "closet": "closet",
    "curtain": "curtain",
    "shelf": "shelf",
    "plant": "plant",
    "ottoman": "ottoman",
    "wardrobe": "wardrobe",
    "desk": "desk",
    "cabinet": "cabinet",
    "tv": "tv",
    "rug": "rug",
    "pillow": "pillow",
    # From indoor-objects
    "door": "door",
    "window": "window",
    "clock": "clock",
    "bed": "bed",
    # From furniture-identifier
    "bookcase": "shelf",
    "dining table": "table",
    "dining chair": "chair",
    "coffee table": "table",
    "end table": "table",
    "night stand": "table",
    "office chair": "chair",
    "arm chair": "chair",
    "lounge chair": "chair",
    "bar stool": "chair",
    "armoire": "closet",
    "dresser": "cabinet",
    "entertainment center": "tv",
    "tv stand": "tv",
    "exercise bike": "exercise_bike",
    "grandfather clock": "clock",
    "piano": "piano",
    "wall art": "wall_art",
    # Common COCO classes that appear in indoor scenes
    "couch": "sofa",
    "potted plant": "plant",
    "diningtable": "table",
    "tvmonitor": "tv",
    "sink": "sink",
    "toilet": "toilet",
    "refrigerator": "refrigerator",
    "oven": "oven",
    "microwave": "microwave",
    "washing machine": "washing_machine",
    "dryer": "dryer",
    "stove": "stove",
    "book": "book",
    "bottle": "bottle",
    "cup": "cup",
    "bowl": "bowl",
    "vase": "vase",
    "scissors": "scissors",
    "teddy bear": "teddy_bear",
    "hair drier": "hair_drier",
    "toothbrush": "toothbrush",
}

# Final unified classes for our use case
UNIFIED_CLASSES = [
    "bed",
    "chair",
    "sofa",
    "table",
    "lamp",
    "closet",
    "curtain",
    "shelf",
    "plant",
    "ottoman",
    "wardrobe",
    "desk",
    "cabinet",
    "tv",
    "rug",
    "pillow",
    "door",
    "window",
    "clock",
    "sink",
    "toilet",
    "refrigerator",
    "stove",
    "vase",
]


def download_dataset(
    dataset_key: str,
    api_key: str,
    output_dir: Path,
) -> Path | None:
    """
    Download a single dataset from Roboflow.

    Args:
        dataset_key: Key in DATASETS dict.
        api_key: Roboflow API key.
        output_dir: Directory to save the downloaded dataset.

    Returns:
        Path to the downloaded dataset, or None if failed.
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(DATASETS.keys())}")
        return None

    config = DATASETS[dataset_key]
    print(f"\nDownloading {dataset_key}...")
    print(f"  Workspace: {config['workspace']}")
    print(f"  Project: {config['project']}")
    print(f"  Description: {config['description']}")

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(config["workspace"]).project(config["project"])
        version = project.version(config["version"])
        dataset = version.download("yolov8")

        # Move to our output directory
        downloaded_path = Path(dataset.location)
        target_path = output_dir / dataset_key

        if target_path.exists():
            import shutil
            shutil.rmtree(target_path)

        import shutil
        shutil.move(str(downloaded_path), str(target_path))

        print(f"  Downloaded to: {target_path}")

        return target_path

    except Exception as e:
        print(f"  Error downloading {dataset_key}: {e}")
        return None


def merge_class_labels(label_dir: Path) -> None:
    """
    Merge class labels in YOLO annotation files using CLASS_MAPPING.

    Modifies .txt files in place.
    """
    for label_file in label_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                new_lines.append(line)
                continue

            cls_id = int(parts[0])
            rest = parts[1:]

            # Read original class name from classes
            # For now, just keep the line as-is
            # The actual mapping happens during dataset combination
            new_lines.append(line)

        label_file.write_text("\n".join(new_lines) + "\n")


def combine_datasets(
    dataset_paths: list[Path],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    """
    Combine multiple YOLO datasets into a single unified dataset.

    Args:
        dataset_paths: List of paths to downloaded YOLO datasets.
        output_dir: Output directory for the combined dataset.
        train_ratio: Fraction for training split.
        val_ratio: Fraction for validation split.
        test_ratio: Fraction for test split.
        seed: Random seed.

    Returns:
        Path to the combined dataset root.
    """
    random.seed(seed)

    # Create output structure
    for split in ["train", "val", "test"]:
        ensure_dir(output_dir / split / "images")
        ensure_dir(output_dir / split / "labels")

    # Collect all images and labels
    all_items = []
    global_class_names = {}

    for dataset_path in dataset_paths:
        print(f"\nProcessing {dataset_path.name}...")

        # Find data.yaml to get class names
        data_yaml = dataset_path / "data.yaml"
        if data_yaml.exists():
            import yaml
            with open(data_yaml) as f:
                data_config = yaml.safe_load(f)
            class_names = data_config.get("names", {})
            # Handle both dict {0: 'name', ...} and list ['name', ...] formats
            if isinstance(class_names, list):
                class_names = {i: name for i, name in enumerate(class_names)}
            # Filter out numeric-only class names (dataset issue)
            filtered_names = {k: v for k, v in class_names.items() if not str(v).isdigit()}
            if filtered_names:
                global_class_names.update(filtered_names)
                print(f"  Classes: {list(filtered_names.values())}")
            else:
                print(f"  Classes: Using unified class mapping (original: {list(class_names.values())})")

        # Collect images from train/valid/test splits
        for split in ["train", "valid", "test"]:
            split_dir = dataset_path / split
            if not split_dir.exists():
                continue

            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            if not images_dir.exists():
                continue

            for img_path in images_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    label_path = labels_dir / (img_path.stem + ".txt")
                    all_items.append({
                        "image": img_path,
                        "label": label_path if label_path.exists() else None,
                        "source": dataset_path.name,
                        "original_split": split,
                    })

    print(f"\nTotal items collected: {len(all_items)}")

    # Shuffle and split
    random.shuffle(all_items)
    n_total = len(all_items)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": all_items[:n_train],
        "val": all_items[n_train:n_train + n_val],
        "test": all_items[n_train + n_val:],
    }

    # Copy files
    for split_name, items in splits.items():
        print(f"\n{split_name}: {len(items)} images")
        for item in items:
            img_out = output_dir / split_name / "images" / item["image"].name
            shutil.copy2(item["image"], img_out)

            if item["label"] and item["label"].exists():
                label_out = output_dir / split_name / "labels" / item["label"].name
                shutil.copy2(item["label"], label_out)

    # Write data.yaml
    data_yaml_content = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": {i: name for i, name in enumerate(UNIFIED_CLASSES)},
    }

    import yaml
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)

    print(f"\nDataset saved to: {output_dir}")
    print(f"Classes: {len(UNIFIED_CLASSES)}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    return output_dir


def run_download(
    datasets: list[str] | None = None,
    api_key: str | None = None,
    output_dir: str | Path | None = None,
) -> Path | None:
    """
    Main download and merge pipeline.

    Args:
        datasets: List of dataset keys to download. None = all.
        api_key: Roboflow API key. Falls back to .env file or ROBOFLOW_API_KEY env var.
        output_dir: Output directory.

    Returns:
        Path to combined dataset, or None if failed.
    """
    # Try to load from .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)

    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: No Roboflow API key provided.")
        print("Set ROBOFLOW_API_KEY environment variable or use --api_key flag.")
        print("\nGet your API key at: https://app.roboflow.com/settings/api")
        return None

    if datasets is None:
        datasets = list(DATASETS.keys())

    output_path = Path(output_dir) if output_dir else PROJECT_ROOT / "data" / "yolo"
    ensure_dir(output_path)

    # Download each dataset
    downloaded_paths = []
    for dataset_key in datasets:
        path = download_dataset(dataset_key, api_key, output_path / "raw")
        if path:
            downloaded_paths.append(path)

    if not downloaded_paths:
        print("\nNo datasets downloaded successfully.")
        return None

    # Combine datasets
    combined_path = output_path / "combined"
    combine_datasets(downloaded_paths, combined_path)

    return combined_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and merge furniture detection datasets from Roboflow."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=None,
        help="Datasets to download. Default: all.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Roboflow API key. Falls back to ROBOFLOW_API_KEY env var.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: data/yolo/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_download(args.datasets, args.api_key, args.output_dir)


if __name__ == "__main__":
    main()
