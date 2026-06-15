"""
Data augmentation para balancear clases.

Transformaciones suaves diseñadas para fotos de bienes raíces:
  - RandomResizedCrop suave
  - HorizontalFlip
  - ColorJitter moderado
  - Rotation ±5°
  - Perspective leve
  - GaussianBlur leve
  - JPEG compression

Evita: VerticalFlip, rotaciones 90°/180°, recortes extremos,
       CutMix, MixUp, distorsiones fuertes, cambios de color exagerados.

Uso:
    # Sin split (Data leakage - NO recomendado)
    python -m src.ml.augment --data_dir data/raw --output_dir outputs/augmented --target_count 500

    # Con split (Recomendado - evita data leakage)
    python -m src.ml.augment --data_dir data/raw --output_dir outputs/augmented --target_count 500 --test_split 0.15
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from ..config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from ..labels import ROOM_LABELS
from ..utils import ensure_dir, validate_dir


def split_data(
    data_dir: Path,
    output_dir: Path,
    test_split: float = 0.15,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """
    Divide el dataset en train y test ANTES de augmentation.
    
    Evita data leakage: las imágenes de test nunca se augmentan.
    
    Args:
        data_dir: Directorio con imágenes originales
        output_dir: Directorio base de salida
        test_split: Fracción de imágenes para test (0.0 - 1.0)
        random_state: Semilla aleatoria
        
    Returns:
        Tupla (train_dir, test_dir)
    """
    data_path = validate_dir(data_dir)
    train_dir = ensure_dir(output_dir / "train")
    test_dir = ensure_dir(output_dir / "test")
    
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    print(f"\n{'='*60}")
    print(f"SPLITTING DATA: {test_split:.0%} test / {1-test_split:.0%} train")
    print(f"{'='*60}")
    
    random.seed(random_state)
    
    for class_name in ROOM_LABELS:
        class_dir = data_path / class_name
        if not class_dir.is_dir():
            continue
        
        # Obtener todas las imágenes de la clase
        images = [
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not images:
            continue
        
        # Shuffle y dividir
        random.shuffle(images)
        n_test = max(1, int(len(images) * test_split))
        test_images = images[:n_test]
        train_images = images[n_test:]
        
        # Copiar train
        train_class_dir = ensure_dir(train_dir / class_name)
        for img in train_images:
            dest = train_class_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        
        # Copiar test
        test_class_dir = ensure_dir(test_dir / class_name)
        for img in test_images:
            dest = test_class_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        
        print(f"  {class_name}: {len(train_images)} train, {len(test_images)} test")
    
    # Resumen
    total_train = sum(1 for _ in train_dir.rglob("*") if _.is_file() and _.suffix.lower() in image_extensions)
    total_test = sum(1 for _ in test_dir.rglob("*") if _.is_file() and _.suffix.lower() in image_extensions)
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_train} train, {total_test} test")
    print(f"Train: {train_dir}")
    print(f"Test: {test_dir}")
    print(f"{'='*60}")
    
    return train_dir, test_dir


def get_augmentation_pipeline() -> transforms.Compose:
    """
    Pipeline de augmentation suave para fotos de bienes raíces.
    Cada transformación se aplica con probabilidad moderada.
    """
    return transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=1.0),
        ], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02,
            ),
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation(
                degrees=5,
                interpolation=transforms.InterpolationMode.BICUBIC,
                fill=0,
            ),
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomPerspective(
                distortion_scale=0.1,
                p=1.0,
                fill=0,
            ),
        ], p=0.3),
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 0.5),
            ),
        ], p=0.2),
    ])


def jpeg_compress(image: Image.Image, quality: int | None = None) -> Image.Image:
    """
    Simula compresión JPEG (artefactos leves).
    quality: 70-95 (mayor = menos compresión).
    """
    if quality is None:
        quality = random.randint(75, 92)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def augment_image(
    image: Image.Image,
    transform: transforms.Compose,
    seed: int | None = None,
    jpeg_quality: int | None = None,
) -> Image.Image:
    """
    Aplica augmentation random a una imagen usando torchvision transforms.
    JPEG compression se aplica por separado (20% de probabilidad).
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    augmented = transform(image)

    # JPEG compression (20% de probabilidad)
    if random.random() > 0.8:
        augmented = jpeg_compress(augmented, quality=jpeg_quality)

    return augmented


def augment_class(
    class_dir: Path,
    output_dir: Path,
    target_count: int,
) -> dict:
    """
    Augmenta una clase hasta alcanzar target_count imágenes.

    Returns:
        Diccionario con estadísticas de la clase.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Contar imágenes existentes
    existing_images = [
        f for f in class_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    current_count = len(existing_images)

    if current_count >= target_count:
        # Aún si ya tiene suficientes, copiar imágenes a la carpeta de salida
        class_output = output_dir / class_dir.name
        ensure_dir(class_output)
        for img_path in existing_images:
            output_path = class_output / img_path.name
            if not output_path.exists():
                try:
                    with Image.open(img_path) as img:
                        img.convert("RGB").save(output_path)
                except Exception as e:
                    print(f"Error copying {img_path}: {e}")

        return {
            "class": class_dir.name,
            "existing": current_count,
            "augmented": 0,
            "total": current_count,
            "status": "already sufficient"
        }

    # Crear carpeta de salida
    class_output = output_dir / class_dir.name
    ensure_dir(class_output)

    # Copiar imágenes originales
    for img_path in existing_images:
        output_path = class_output / img_path.name
        if not output_path.exists():
            with Image.open(img_path) as img:
                img.convert("RGB").save(output_path)

    # Calcular cuántas augmented necesitamos
    needed = target_count - current_count
    augmented_count = 0

    # Crear pipeline de augmentation
    transform = get_augmentation_pipeline()

    # Augmentar hasta alcanzar el objetivo
    with tqdm(total=needed, desc=f"  {class_dir.name}", leave=False) as pbar:
        while augmented_count < needed:
            for img_path in existing_images:
                if augmented_count >= needed:
                    break

                try:
                    with Image.open(img_path) as img:
                        rgb_img = img.convert("RGB")

                    # Generar augmented con seed único
                    augmented = augment_image(rgb_img, transform, seed=random.randint(0, 10000))

                    # Guardar
                    aug_name = f"aug_{augmented_count:05d}_{img_path.stem}.jpg"
                    aug_path = class_output / aug_name
                    augmented.save(aug_path, quality=90)

                    augmented_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Error augmenting {img_path}: {e}")

    return {
        "class": class_dir.name,
        "existing": current_count,
        "augmented": augmented_count,
        "total": current_count + augmented_count,
        "status": "augmented"
    }


def run_augmentation(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "augmented",
    target_count: int | None = None,
    test_split: float | None = None,
    random_state: int = 42,
) -> list[dict]:
    """
    Augmenta todas las clases hasta balancear el dataset.
    
    Si test_split se especifica, divide primero en train/test.
    Solo se augmenta el conjunto de train.
    
    Output structure:
        output_dir/
            train/
                sala/
                    img1.jpg
                    aug_00001_img1.jpg
            test/
                sala/
                    img2.jpg

    Si target_count no se especifica, usa el conteo de la clase más grande.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Si test_split está definido, dividir primero
    if test_split is not None and test_split > 0:
        train_dir, test_dir = split_data(data_path, output_path, test_split, random_state)
        # Augmentar solo el train
        data_path = train_dir
        print(f"\nAugmenting TRAIN split only...")
    else:
        print(f"\n⚠️  WARNING: No test split specified. Data leakage possible!")
    
    # Contar imágenes por clase
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    class_counts = {}

    for class_name in ROOM_LABELS:
        class_dir = data_path / class_name
        if class_dir.is_dir():
            count = sum(
                1 for f in class_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            class_counts[class_name] = count

    if not class_counts:
        print(f"No classes found in {data_path}")
        return []

    # Determinar target_count
    if target_count is None:
        target_count = max(class_counts.values())
        print(f"Target count (max class): {target_count}")

    print(f"\nClass distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1]):
        status = "✅" if count >= target_count else f"⬆️ +{target_count - count}"
        print(f"  {cls}: {count} {status}")

    print(f"\nAugmenting to {target_count} per class...")

    # Augmentar cada clase
    # Si hay test_split, las imágenes de train ya están en output_path/train/
    # y las augmented deben ir ahí también
    results = []
    for class_name in ROOM_LABELS:
        class_dir = data_path / class_name
        if not class_dir.is_dir():
            continue

        # When test_split is used, data_path = train_dir = output_path/train
        # Augmented images must go into the same train directory
        result = augment_class(class_dir, data_path, target_count)
        results.append(result)

    # Resumen
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    print(f"{'Class':<20} {'Original':>10} {'Augmented':>10} {'Total':>10} {'Status':<15}")
    print("-" * 60)

    for r in results:
        print(f"{r['class']:<20} {r['existing']:>10} {r['augmented']:>10} {r['total']:>10} {r['status']:<15}")

    total_original = sum(r['existing'] for r in results)
    total_augmented = sum(r['augmented'] for r in results)
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_original:>10} {total_augmented:>10} {total_original + total_augmented:>10}")

    print(f"\nAugmented dataset saved to: {output_path}")
    if test_split is not None:
        print(f"Test set (no augmented): {output_path / 'test'}")
    print(f"\nNext step: python -m src.ml.extract_embeddings --data_dir {output_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment images to balance classes.")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="Input directory with class subfolders.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "augmented"), help="Output directory.")
    parser.add_argument("--target_count", type=int, default=None, help="Target images per class. Default: max class count.")
    parser.add_argument("--test_split", type=float, default=None, help="Fraction for test set (0.0-1.0). If specified, splits data BEFORE augmentation to prevent data leakage.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducible splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_augmentation(args.data_dir, args.output_dir, args.target_count, args.test_split, args.random_state)


if __name__ == "__main__":
    main()
