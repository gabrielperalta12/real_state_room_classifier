"""
Extrae embeddings de imágenes usando diferentes modelos visuales.

Soporta:
  - CLIP (OpenAI)
  - DINO (Facebook)
  - Place365 (MIT CSAIL, ResNet50)

Uso:
    # CLIP (default)
    python -m src.ml.extract_embeddings --data_dir data/splits --output_dir outputs/embeddings --model clip

    # DINO
    python -m src.ml.extract_embeddings --data_dir data/splits --output_dir outputs/embeddings --model dino

    # Place365
    python -m src.ml.extract_embeddings --data_dir data/splits --output_dir outputs/embeddings --model place365

    # Comparar modelos
    python -m src.ml.compare_models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from ..labels import ROOM_LABELS
from ..utils import ensure_dir, validate_dir


MODEL_CONFIGS = {
    "clip": {
        "name": "openai/clip-vit-base-patch32",
        "loader": "clip",
    },
    "clip-large": {
        "name": "openai/clip-vit-large-patch14",
        "loader": "clip",
    },
    "dinov2": {
        "name": "facebook/dinov2-base",
        "loader": "dinov2",
    },
    "place365": {
        "name": "resnet50_places365",
        "loader": "place365",
    },
}


def load_model(model_key: str, device: str | None = None):
    """Carga modelo y retorna (model, processor, device, embed_dim)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = MODEL_CONFIGS.get(model_key)
    if config is None:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    if config["loader"] == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(config["name"])
        processor = CLIPProcessor.from_pretrained(config["name"])
        model = model.to(device).eval()
        embed_dim = model.config.projection_dim
        return model, processor, device, embed_dim, "clip"
    
    elif config["loader"] == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(config["name"])
        model = AutoModel.from_pretrained(config["name"])
        model = model.to(device).eval()
        embed_dim = model.config.hidden_size
        return model, processor, device, embed_dim, "dinov2"
    
    elif config["loader"] == "place365":
        import torchvision.models as tmodels
        import torch.nn as nn
        import urllib.request
        
        weight_path = "/tmp/resnet50_places365.pth.tar"
        if not Path(weight_path).exists():
            print("Downloading Place365 weights...")
            url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
            urllib.request.urlretrieve(url, weight_path)
        
        model = tmodels.resnet50(num_classes=365)
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        
        # Feature extractor: everything except the final FC layer
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor = feature_extractor.to(device).eval()
        
        # Simple processor: resize + normalize with ImageNet stats
        processor = None
        embed_dim = 2048
        return feature_extractor, processor, device, embed_dim, "place365"
    
    else:
        raise ValueError(f"Unknown loader: {config['loader']}")


def extract_clip_embedding(model, processor, device, image: Image.Image) -> np.ndarray:
    """Extrae embedding de CLIP."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs[1]
        embedding = model.visual_projection(pooled_output)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0).cpu().numpy()


def extract_dino_embedding(model, processor, device, image: Image.Image) -> np.ndarray:
    """Extrae embedding de DINO."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0).cpu().numpy()


def extract_dinov2_embedding(model, processor, device, image: Image.Image) -> np.ndarray:
    """Extrae embedding de DINOv2."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0).cpu().numpy()


def extract_place365_embedding(model, processor, device, image: Image.Image) -> np.ndarray:
    """Extrae embedding de Place365 (ResNet50 features before FC)."""
    import torchvision.transforms as transforms
    
    # Standard ImageNet normalization (same as Place365 training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(tensor).flatten(1)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze(0).cpu().numpy()


def extract_from_directory(
    data_path: Path,
    model,
    processor,
    device,
    model_type: str,
    desc: str = "Extracting embeddings",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extrae embeddings de todas las imágenes en un directorio."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    image_paths = []
    labels = []

    for class_name in ROOM_LABELS:
        class_dir = data_path / class_name
        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                image_paths.append(img_path)
                labels.append(class_name)

    if not image_paths:
        return np.array([]), np.array([]), []

    embeddings = []
    valid_labels = []
    valid_paths = []
    skipped = 0

    extract_fn_map = {
        "clip": extract_clip_embedding,
        "dinov2": extract_dinov2_embedding,
        "place365": extract_place365_embedding,
    }
    extract_fn = extract_fn_map[model_type]

    for image_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc=desc):
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                embedding = extract_fn(model, processor, device, rgb_image)

            embeddings.append(embedding)
            valid_labels.append(label)
            valid_paths.append(str(image_path))
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            skipped += 1

    if not embeddings:
        return np.array([]), np.array([]), []

    X = np.vstack(embeddings)
    y = np.array(valid_labels)
    
    return X, y, valid_paths


def extract_embeddings(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "embeddings",
    model_key: str = "clip",
) -> dict[str, Path]:
    """
    Extrae embeddings de train/test splits usando el modelo especificado.
    
    Output structure:
        output_dir/
            {model_key}/
                train/
                    X_embeddings.npy
                    y_labels.npy
                    metadata.csv
                test/
                    X_embeddings.npy
                    y_labels.npy
                    metadata.csv
    """
    data_path = validate_dir(data_dir)
    output_path = ensure_dir(output_dir) / model_key
    ensure_dir(output_path)

    # Cargar modelo
    print(f"\nLoading model: {model_key}")
    model, processor, device, embed_dim, model_type = load_model(model_key)
    print(f"Device: {device}, Embedding dim: {embed_dim}")

    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not (train_dir.is_dir() and test_dir.is_dir()):
        raise ValueError(f"Expected train/test structure in {data_path}")
    
    results = {}
    
    # Extract train
    print(f"\nExtracting TRAIN embeddings...")
    X_train, y_train, paths_train = extract_from_directory(
        train_dir, model, processor, device, model_type, "Train"
    )
    
    if len(X_train) > 0:
        train_path = ensure_dir(output_path / "train")
        np.save(train_path / "X_embeddings.npy", X_train)
        np.save(train_path / "y_labels.npy", y_train)
        pd.DataFrame({"image_path": paths_train, "label": y_train}).to_csv(
            train_path / "metadata.csv", index=False
        )
        results["train"] = train_path
        print(f"  Shape: {X_train.shape}")
    
    # Extract test
    print(f"\nExtracting TEST embeddings...")
    X_test, y_test, paths_test = extract_from_directory(
        test_dir, model, processor, device, model_type, "Test"
    )
    
    if len(X_test) > 0:
        test_path = ensure_dir(output_path / "test")
        np.save(test_path / "X_embeddings.npy", X_test)
        np.save(test_path / "y_labels.npy", y_test)
        pd.DataFrame({"image_path": paths_test, "label": y_test}).to_csv(
            test_path / "metadata.csv", index=False
        )
        results["test"] = test_path
        print(f"  Shape: {X_test.shape}")
    
    # Save model info
    info = {
        "model_key": model_key,
        "model_name": MODEL_CONFIGS[model_key]["name"],
        "embed_dim": embed_dim,
        "train_images": len(X_train),
        "test_images": len(X_test),
    }
    pd.DataFrame([info]).to_csv(output_path / "model_info.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE: {model_key}")
    print(f"{'='*60}")
    print(f"Train: {X_train.shape[0]} images → {output_path / 'train'}")
    print(f"Test: {X_test.shape[0]} images → {output_path / 'test'}")
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract visual embeddings from labeled images.")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="Directory with train/test splits.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "embeddings"), help="Output directory.")
    parser.add_argument("--model", default="clip", choices=list(MODEL_CONFIGS.keys()), 
                       help="Model to use for extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_embeddings(args.data_dir, args.output_dir, args.model)


if __name__ == "__main__":
    main()
