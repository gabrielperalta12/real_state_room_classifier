"""
Utilidad para detectar imágenes duplicadas o similares.

Uso:
    python -m src.utils.duplicates --data_dir data/raw
    python -m src.utils.duplicates --data_dir data/augmented --threshold 5
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

from PIL import Image
import numpy as np


def get_image_hash(image_path: Path) -> str:
    """Calcula hash MD5 del archivo."""
    return hashlib.md5(image_path.read_bytes()).hexdigest()


def get_perceptual_hash(image_path: Path, size: int = 8) -> np.ndarray:
    """Calcula hash perceptual (pHash) simplificado."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((size, size), Image.LANCZOS)
            pixels = np.array(img, dtype=np.float32)
            avg = pixels.mean()
            return (pixels > avg).flatten().astype(np.uint8)
    except Exception:
        return None


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Calcula distancia de Hamming entre dos hashes."""
    return int(np.sum(hash1 != hash2))


def find_duplicates(
    data_dir: str | Path,
    threshold: int = 5,
    check_subdirs: bool = True,
) -> dict:
    """
    Encuentra imágenes duplicadas y similares.

    Args:
        data_dir: Directorio con imágenes.
        threshold: Distancia máxima de Hamming para considerar similares (0-64).
        check_subdirs: Si True, busca en subdirectorios.

    Returns:
        Diccionario con duplicados exactos y similares.
    """
    data_path = Path(data_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    # Recopilar imágenes
    if check_subdirs:
        images = [
            f for f in data_path.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    else:
        images = [
            f for f in data_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

    print(f"Found {len(images)} images in {data_path}")

    # 1. Buscar duplicados exactos (MD5)
    print("\n=== Duplicados Exactos (MD5) ===")
    hash_groups = defaultdict(list)
    for img_path in images:
        try:
            h = get_image_hash(img_path)
            hash_groups[h].append(img_path)
        except Exception as e:
            print(f"Error: {img_path}: {e}")

    exact_duplicates = {
        h: paths for h, paths in hash_groups.items()
        if len(paths) > 1
    }

    if exact_duplicates:
        print(f"\nFound {len(exact_duplicates)} groups of exact duplicates:")
        for h, paths in exact_duplicates.items():
            print(f"\n  Hash: {h[:16]}...")
            for p in paths:
                print(f"    - {p}")
    else:
        print("No exact duplicates found.")

    # 2. Buscar similares (pHash)
    print(f"\n=== Imágenes Similares (threshold={threshold}) ===")
    phash_data = []
    for img_path in images:
        try:
            ph = get_perceptual_hash(img_path)
            if ph is not None:
                phash_data.append((img_path, ph))
        except Exception:
            pass

    similar_pairs = []
    n = len(phash_data)
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming_distance(phash_data[i][1], phash_data[j][1])
            if dist <= threshold:
                similar_pairs.append((phash_data[i][0], phash_data[j][0], dist))

    if similar_pairs:
        print(f"\nFound {len(similar_pairs)} similar pairs:")
        for p1, p2, dist in sorted(similar_pairs, key=lambda x: x[2]):
            print(f"  Distance {dist}:")
            print(f"    - {p1}")
            print(f"    - {p2}")
    else:
        print("No similar images found.")

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Total imágenes: {len(images)}")
    print(f"Duplicados exactos: {sum(len(p) - 1 for p in exact_duplicates.values())}")
    print(f"Pares similares: {len(similar_pairs)}")

    return {
        "total": len(images),
        "exact_duplicates": exact_duplicates,
        "similar_pairs": similar_pairs,
    }


def remove_duplicates(
    data_dir: str | Path,
    threshold: int = 5,
    dry_run: bool = True,
) -> list[Path]:
    """
    Elimina imágenes duplicadas (mantiene una de cada grupo).

    Args:
        data_dir: Directorio con imágenes.
        threshold: Distancia de Hamming para considerar similares.
        dry_run: Si True, solo muestra qué se eliminaría.

    Returns:
        Lista de archivos eliminados (o que se eliminarían).
    """
    result = find_duplicates(data_dir, threshold)
    to_remove = []

    # Marcar duplicados exactos (mantener el primero)
    for h, paths in result["exact_duplicates"].items():
        to_remove.extend(paths[1:])

    # Marcar similares (mantener el primero del par)
    seen = set()
    for p1, p2, dist in result["similar_pairs"]:
        if p2 not in seen:
            to_remove.append(p2)
            seen.add(p2)

    if dry_run:
        print(f"\n[DRY RUN] Would remove {len(to_remove)} files:")
        for p in to_remove:
            print(f"  - {p}")
    else:
        print(f"\nRemoving {len(to_remove)} files...")
        for p in to_remove:
            try:
                p.unlink()
                print(f"  Removed: {p}")
            except Exception as e:
                print(f"  Error removing {p}: {e}")

    return to_remove


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicate images.")
    parser.add_argument("--data_dir", required=True, help="Directory with images.")
    parser.add_argument("--threshold", type=int, default=5, help="Hamming distance threshold (0-64, default=5).")
    parser.add_argument("--remove", action="store_true", help="Remove duplicates (keeps one of each group).")
    parser.add_argument("--no_subdirs", action="store_true", help="Don't search subdirectories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.remove:
        remove_duplicates(args.data_dir, args.threshold, dry_run=False)
    else:
        find_duplicates(args.data_dir, args.threshold, not args.no_subdirs)


if __name__ == "__main__":
    main()
