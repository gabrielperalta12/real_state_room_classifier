"""
YOLO object detection wrapper for indoor furniture.

Supports YOLO11/YOLOv8 models from Ultralytics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..config import DEFAULT_MODEL_DIR


class FurnitureDetector:
    """
    YOLO-based detector for indoor furniture objects.

    Detects furniture items (bed, chair, sofa, table, lamp, etc.)
    and generates natural language descriptions.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.25,
        device: str | None = None,
    ):
        """
        Initialize the furniture detector.

        Args:
            model_path: Path to custom YOLO weights. If None, uses yolov8n pretrained.
            confidence: Minimum confidence threshold for detections.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
        """
        from ultralytics import YOLO

        self.confidence = confidence
        self.device = device

        if model_path is None:
            self.model = YOLO("yolo11n.pt")
            self.model_name = "yolo11n (pretrained)"
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model = YOLO(str(model_path))
            self.model_name = model_path.stem

    def detect(
        self,
        image_path: str | Path,
        conf: float | None = None,
    ) -> list[dict]:
        """
        Detect furniture objects in an image.

        Args:
            image_path: Path to the image file.
            conf: Confidence threshold (overrides default).

        Returns:
            List of detections, each with:
                - class_name: str
                - confidence: float
                - bbox: dict with x1, y1, x2, y2 (pixel coordinates)
                - area: float (bounding box area in pixels)
        """
        conf = conf or self.confidence
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox_area = (x2 - x1) * (y2 - y1)

                    detections.append({
                        "class_name": cls_name,
                        "confidence": confidence,
                        "bbox": {
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                        },
                        "area": round(bbox_area, 1),
                    })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect_batch(
        self,
        image_paths: list[str | Path],
        conf: float | None = None,
    ) -> dict[str, list[dict]]:
        """
        Detect furniture in multiple images.

        Returns:
            Dictionary mapping image path to list of detections.
        """
        results = {}
        for image_path in image_paths:
            try:
                results[str(image_path)] = self.detect(image_path, conf)
            except Exception as e:
                results[str(image_path)] = [{"error": str(e)}]
        return results

    def describe_room(self, detections: list[dict]) -> str:
        """
        Generate a natural language description of detected objects.

        Args:
            detections: List of detections from detect().

        Returns:
            Spanish description of the room contents.
        """
        if not detections:
            return "No se detectaron muebles en la imagen."

        furniture_counts: dict[str, int] = {}
        for det in detections:
            name = det["class_name"]
            furniture_counts[name] = furniture_counts.get(name, 0) + 1

        SPANISH_NAMES = {
            "chair": "silla",
            "sofa": "sofá",
            "bed": "cama",
            "table": "mesa",
            "lamp": "lámpara",
            "tv": "televisor",
            "dining table": "mesa de comedor",
            "couch": "sofá",
            "bed": "cama",
            "potted plant": "planta",
            "plant": "planta",
            "shelf": "estante",
            "wardrobe": "armario",
            "desk": "escritorio",
            "cabinet": "gabinete",
            "rug": "alfombra",
            "pillow": "almohada",
            "curtain": "cortina",
            "ottoman": "puf",
            "door": "puerta",
            "window": "ventana",
            "clock": "reloj",
            "bookcase": "estantería",
            "sink": "lavabo",
            "toilet": "inodoro",
            "refrigerator": "refrigerador",
            "oven": "horno",
            "microwave": "microondas",
            "washing machine": "lavadora",
            "dryer": "secadora",
            "stove": "cocina",
        }

        items = []
        for cls_name, count in sorted(furniture_counts.items()):
            spanish = SPANISH_NAMES.get(cls_name, cls_name)
            if count > 1:
                items.append(f"{count} {spanish}s")
            else:
                items.append(f"un {spanish}" if spanish[0].lower() in "aeiou" else f"una {spanish}")

        if len(items) == 1:
            return f"Se detecta {items[0]} en la imagen."
        elif len(items) == 2:
            return f"Se detectan {items[0]} y {items[1]} en la imagen."
        else:
            last = items.pop()
            return f"Se detectan {', '.join(items)} y {last} en la imagen."

    def get_object_summary(self, detections: list[dict]) -> dict:
        """
        Get a summary of detected objects by class.

        Returns:
            Dictionary with class counts and total objects.
        """
        counts: dict[str, int] = {}
        for det in detections:
            name = det["class_name"]
            counts[name] = counts.get(name, 0) + 1

        return {
            "total_objects": len(detections),
            "unique_classes": len(counts),
            "by_class": counts,
        }
