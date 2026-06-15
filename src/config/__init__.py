"""Central configuration for paths and model names."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_YOLO_MODEL = "yolo11n.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_METRICS_DIR = DEFAULT_OUTPUT_DIR / "metrics"
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures"
DEFAULT_DETECTION_DIR = DEFAULT_OUTPUT_DIR / "detections"

YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.7
