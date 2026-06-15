# Guía de Distribución

## ¿Qué va al repo de Git?

Solo **código fuente** (~2MB):

```
real_estate_room_classifier/
├── src/                          # Todo el código Python
├── requirements.txt
├── README.md
├── .gitignore
└── .gitkeep                      # Mantiene carpetas vacías
```

## ¿Qué va al ZIP para Google Drive?

**Código + modelos entrenados + datos de ejemplo** (~400MB):

```
room_classifier_export.zip
├── src/                          # Código fuente
├── requirements.txt
├── README.md
├── models/
│   └── yolo_furniture/
│       └── best.pt               # YOLO fine-tuned (21MB)
├── outputs/
│   └── comparison/
│       ├── models/
│       │   ├── clip/
│       │   │   ├── best_classifier.joblib      # SVM CLIP (89% F1)
│       │   │   └── *.csv, *.png
│       │   ├── dinov2/
│       │   │   ├── best_classifier.joblib      # SVM DINOv2 (88% F1)
│       │   │   └── *.csv, *.png
│       │   └── place365/
│       │       ├── best_classifier.joblib      # XGBoost Place365 (86% F1)
│       │       └── *.csv, *.png
│       ├── model_comparison.csv
│       └── comparison.png
└── report/
    ├── paper_ieee/
    │   ├── paper.tex
    │   └── paper.pdf
    └── presentation/
        ├── presentation.tex
        └── presentation.pdf
```

## ¿Qué NO subir (queda en tu máquina)?

| Carpeta | Tamaño | Contenido |
|---------|--------|-----------|
| `data/raw/` | ~200MB | 10,150 imágenes originales |
| `data/splits/` | ~300MB | Train/test + augmented |
| `data/scrapped/` | variable | Listings scrapeados |
| `outputs/embeddings/` | ~500MB | Embeddings .npy |
| `.venv/` | ~8GB | Entorno virtual |

## Comandos para crear el ZIP

```bash
# Solo código + modelos (para compartir proyecto)
zip -r room_classifier_code.zip \
    src/ \
    requirements.txt \
    README.md \
    report/ \
    models/yolo_furniture/best.pt \
    outputs/comparison/models/ \
    outputs/comparison/model_comparison.csv \
    outputs/comparison/comparison.png

# Código + modelos + datos de ejemplo (para reproducción completa)
zip -r room_classifier_full.zip \
    src/ \
    requirements.txt \
    README.md \
    report/ \
    models/ \
    outputs/comparison/ \
    data/splits/ \
    --exclude "data/splits/train/*"  # Excluir augmented para reducir tamaño
```

## Para importar desde Google Drive

1. Subir el ZIP a Google Drive
2. En Colab o Streamlit Cloud:
```python
# Descargar y descomprimir
!gdown <FILE_ID> -O room_classifier.zip
!unzip room_classifier.zip -d /content/real_estate_room_classifier
```

## Modelo mínimo viable (Streamlit Cloud)

Para deploy en Streamlit Cloud, solo necesitas:
- `src/web/app.py`
- `src/` (todo el código)
- `requirements.txt`
- `outputs/comparison/models/*/best_classifier.joblib` (~3MB total)
- `report/references/*.png` (imágenes del README)
