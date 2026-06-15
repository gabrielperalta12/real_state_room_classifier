# Indoor Real Estate Room Classifier

Sistema de clasificación automática de imágenes de ambientes inmobiliarios. Combina CLIP/DINOv2/Place365 para embeddings visuales, classifiers ML entrenados (SVM, LogReg, RF, XGBoost) y YOLO11 para detección de muebles — todo servido via una aplicación Streamlit con 3 modos de operación.

## Stack Tecnológico

| Categoría | Tecnología | Versión |
|-----------|-----------|---------|
| **Lenguaje** | Python | 3.10+ |
| **Deep Learning** | PyTorch | 2.12.0 (CUDA 13.2, cu126) |
| **Vision-Language** | HuggingFace Transformers | ≥4.38.0 |
| **Modelos** | CLIP ViT-B/32, DINOv2 ViT-B/14, Place365 ResNet50 | — |
| **ML Clasificación** | scikit-learn (SVM, LogReg, RF) | ≥1.3.0 |
| **Gradient Boosting** | XGBoost | ≥2.0.0 |
| **Detección Objetos** | Ultralytics YOLO11 | ≥8.0.0 |
| **Web Scraping** | Selenium | ≥4.15.0 |
| **Web App** | Streamlit | ≥1.30.0 |
| **Manipulación Datos** | NumPy, Pandas | ≥1.24.0, ≥2.0.0 |
| **Visualización** | Matplotlib, Seaborn | ≥3.7.0, ≥0.12.0 |
| **Imágenes** | Pillow | ≥10.0.0 |

## Arquitectura

```
Urbania.pe (Selenium) → CLIP Auto-label → Augmentation → Train/Test Split
                                                                 ↓
                                           ┌──────────────────────────────────┐
                                           │    Embeddings: CLIP ViT-B/32     │
                                           │              o DINOv2 ViT-B/14   │
                                           │              o Place365 ResNet50  │
                                           └──────────────────────────────────┘
                                                                 ↓
                                                 SVM / LogReg / RF / XGBoost
                                                                 ↓
                                               ┌────────────────────────────┐
                                               │  Streamlit Web App          │
                                               │  • Zero-shot CLIP           │
                                               │  • ML Classifier (SVM)      │
                                               │  • ML + YOLO Objects        │
                                               └────────────────────────────┘
```

## Pipeline

1. **Scraping**: Selenium para extraer imágenes de Urbania.pe (carousel, lazy-load)
2. **Auto-labeling**: CLIP clasifica imágenes y las organiza en carpetas
3. **Revisión manual + Airbnb**: Revisión manual de carpetas + descarga de imágenes adicionales de Airbnb para complementar clases débiles
4. **Split train/test**: Separación ANTES de augmentation para evitar data leakage
5. **Data augmentation**: Balanceo de clases con torchvision (flip, brillo, contraste, saturación, blur, recorte suave)
6. **Preprocesamiento**: Cada modelo usa su pipeline de preprocesamiento antes de generar embeddings
7. **Embeddings**: Extracción de representaciones vectoriales con CLIP, DINOv2 o Place365
8. **Entrenamiento**: 4 clasificadores con comparación automática + XGBoost tuning
9. **Evaluación**: Confusion matrices, métricas por clase, comparación de modelos
10. **Web app**: Streamlit con 3 modos (zero-shot, ML, ML+YOLO)

## Preprocesamiento por Modelo

Cada modelo de embeddings requiere preprocesamiento específico antes de la extracción de features:

| Modelo | Preprocesamiento | Resolución | Normalización |
|--------|------------------|------------|---------------|
| **CLIP** | CLIPProcessor (Rescale → CenterCrop → Normalize) | 224×224 | Stats propias de CLIP |
| **DINOv2** | AutoImageProcessor (Rescale → CenterCrop → Normalize) | 224×244 | ImageNet (mean=[0.485, 0.456, 0.406]) |
| **Place365** | torchvision (Resize → CenterCrop → ToTensor → Normalize) | 224×224 | ImageNet (mean=[0.485, 0.456, 0.406]) |

**Diferencias clave:**
- CLIP usa sus propias estadísticas de normalización (entrenado con WIT)
- Place365 aplica Resize(256) antes de CenterCrop(224), preservando diferente relación de aspecto
- Todas las imágenes se convierten a RGB y producen tensores (1, 3, 224, 224)

## Clases (17 categorías de ambientes)

| Carpeta | Etiqueta | Prompt CLIP |
|---|---|---|
| `sala` | sala | a photo of a living room or event hall in a real estate listing |
| `cocina` | cocina | a photo of a kitchen in a real estate listing |
| `dormitorio` | dormitorio | a photo of a bedroom in a real estate listing |
| `bano` | baño | a photo of a bathroom in a real estate listing |
| `comedor` | comedor | a photo of a dining room in a real estate listing |
| `lavanderia` | lavandería | a photo of a laundry room in a real estate listing |
| `balcon` | balcón | a photo of a balcony in a real estate listing |
| `terraza` | terraza | a photo of a terrace or rooftop in a real estate listing |
| `gimnasio` | gimnasio | a photo of a gym or fitness room in a real estate listing |
| `exterior` | exterior | a photo of an outdoor or exterior view in a real estate listing |
| `area_trabajo` | área de trabajo | a photo of an office or workspace in a real estate listing |
| `patio_jardin` | patio/jardín | a photo of a patio or garden in a real estate listing |
| `estacionamiento` | estacionamiento | a photo of a parking or garage in a real estate listing |
| `pasadizo` | pasadizo/corredor | a photo of a hallway or corridor in a real estate listing |
| `recepcion` | recepción | a photo of a reception or lobby in a real estate listing |
| `walking_closet` | walking closet | a photo of a walk-in closet or dressing room in a real estate listing |
| `piscina` | piscina | a photo of a swimming pool in a real estate listing |

## Clases YOLO (24 objetos de mobiliario)

`bed`, `chair`, `sofa`, `table`, `lamp`, `closet`, `curtain`, `shelf`, `plant`, `ottoman`, `wardrobe`, `desk`, `cabinet`, `tv`, `rug`, `pillow`, `door`, `window`, `clock`, `sink`, `toilet`, `refrigerator`, `stove`, `vase`

## Estructura

```text
real_estate_room_classifier/
├── README.md
├── requirements.txt
├── setup.sh                          # Setup automático desde Google Drive
├── .gitignore
├── data/
│   ├── raw/                         # 10,150 imágenes originales por clase
│   ├── splits/
│   │   ├── train/                   # 14,492 imágenes (8,663 original + 5,829 augmented, 700/clase)
│   │   └── test/                    # 1,520 imágenes (original, sin augmentation)
│   └── preprocessed/                # Listings scrapeados
├── models/
│   └── yolo_furniture/
│       └── best.pt                  # YOLO11 fine-tuned (24 clases, complemento)
├── outputs/
│   ├── comparison/                  # CLIP vs DINOv2 vs Place365 vs Zero-shot
│   │   ├── model_comparison.csv
│   │   ├── comparison.png
│   │   ├── embeddings/              # Embeddings generados por compare_models
│   │   └── models/                  # Clasificadores entrenados por backbone
│   │       ├── clip/
│   │       │   ├── best_classifier.joblib   # SVM RBF (89.0% F1)
│   │       │   ├── model_comparison.csv
│   │       │   ├── test_classification_report.csv
│   │       │   └── test_per_class_metrics.csv
│   │       ├── dinov2/
│   │       │   ├── best_classifier.joblib   # SVM RBF (88.4% F1)
│   │       │   ├── model_comparison.csv
│   │       │   ├── test_classification_report.csv
│   │       │   └── test_per_class_metrics.csv
│   │       └── place365/
│   │           ├── best_classifier.joblib   # XGBoost (86.2% F1)
│   │           ├── model_comparison.csv
│   │           ├── test_classification_report.csv
│   │           └── test_per_class_metrics.csv
├── report/
│   ├── paper_ieee/                  # Paper IEEE (English)
│   │   ├── paper.tex
│   │   └── paper.pdf
│   ├── presentation/                # Beamer presentation (Spanish)
│   │   ├── presentation.tex
│   │   └── presentation.pdf
│   └── references/                  # Imágenes de referencia
├── src/
│   ├── config/__init__.py           # Rutas del proyecto
│   ├── labels.py                    # 17 clases + 24 clases YOLO
│   ├── utils/__init__.py            # ensure_dir, validate_dir
│   ├── utils/duplicates.py          # Detección de duplicados (MD5 + pHash)
│   ├── scraping/
│   │   ├── urbania.py               # UrbaniaScraper con Selenium
│   │   └── extract_images.py        # CLI para scraping
│   ├── clip/
│   │   ├── loader.py                # Carga de modelo CLIP
│   │   ├── auto_label.py            # Auto-labeling con CLIP
│   │   └── evaluate_zeroshot.py     # Evaluación zero-shot
│   ├── ml/
│   │   ├── extract_embeddings.py    # Multi-model (CLIP/DINOv2/Place365)
│   │   ├── train.py                 # Entrenamiento + tuning
│   │   ├── predict.py               # Predicción con modelo guardado
│   │   ├── evaluate.py              # Evaluación + confusion matrices
│   │   ├── augment.py               # Data augmentation + split train/test
│   │   └── compare_models.py        # CLIP vs DINOv2 vs Place365 comparison
│   ├── detection/
│   │   ├── detector.py              # FurnitureDetector (YOLO11 wrapper)
│   │   ├── train_yolo.py            # YOLO11 fine-tuning
│   │   └── download_datasets.py     # Roboflow download + merge
│   └── web/
│       └── app.py                   # Streamlit web app (3 modos)
```

## Instalación

```bash
# 1. Clonar el repo
git clone https://github.com/gabrielperalta12/real_state_room_classifier.git
cd real_state_room_classifier

# 2. Descargar datos y modelos desde Google Drive
# Link: https://drive.google.com/file/d/1g82NR2-zNttrewfwm0NgD9_qlwyAM1wz/view?usp=sharing
# Descargar project_data.zip y ejecutar:
bash setup.sh <FILE_ID>

# 3. Ejecutar la app
source .venv/bin/activate
streamlit run src/web/app.py
```

Requisitos: Python 3.10+, CUDA 13.2 con PyTorch cu126.

## Uso

### 1. Scraping de imágenes (opcional)

```bash
python -m src.scraping.extract_images --output_dir data/scrapped --max_pages 10
```

**Artefactos generados en `data/scrapped/`:**

```
data/scrapped/
├── listing_0/
│   ├── image_0.jpg
│   ├── image_1.jpg
│   └── ...
├── listing_1/
├── ...
└── listings.json                 # Metadata de todas las propiedades scrapeadas
```

### 2. Auto-labeling con CLIP

```bash
# Clasifica imágenes de data/scrapped/ → data/raw/<clase>/ + data/pending_review/
python -m src.clip.auto_label
```

**Artefactos generados:**

```
data/
├── raw/                          # Imágenes clasificadas con alta confianza
│   ├── sala/
│   ├── cocina/
│   ├── dormitorio/
│   └── ...                       # 17 clases
├── pending_review/               # Imágenes con baja confianza o ambigüedad
└── auto_label_report.csv         # Reporte completo de clasificación
```

### 3. Split train/test + Data augmentation

```bash
# Split 80/20 ANTES de augmentation, luego augmenta train a 700/clase
# Resultado: 14,492 train (8,663 original + 5,829 augmented) / 1,520 test
python -m src.ml.augment \
    --data_dir data/raw \
    --output_dir data/splits \
    --target_count 700 \
    --test_split 0.2
```

### 4. Extraer embeddings

```bash
# CLIP (ViT-B/32)
python -m src.ml.extract_embeddings \
    --data_dir data/splits \
    --output_dir outputs/embeddings \
    --model clip

# DINOv2 (ViT-B/14)
python -m src.ml.extract_embeddings \
    --data_dir data/splits \
    --output_dir outputs/embeddings \
    --model dinov2

# Place365 (ResNet50)
python -m src.ml.extract_embeddings \
    --data_dir data/splits \
    --output_dir outputs/embeddings \
    --model place365
```

### 5. Entrenar modelos

```bash
# Entrenar los 4 modelos (auto-detecta train/test splits: 14,492 train / 1,520 test)
python -m src.ml.train \
    --input_dir outputs/embeddings/clip \
    --output_dir outputs/comparison/models/clip

# Con XGBoost tuning
python -m src.ml.train \
    --input_dir outputs/embeddings/clip \
    --output_dir outputs/comparison/models/clip \
    --tune_xgboost \
    --use_gpu
```

**Artefactos generados en `outputs/comparison/models/`:**

```
outputs/comparison/models/clip/
├── best_classifier.joblib        # Pipeline: StandardScaler + Classifier
├── model_comparison.csv          # Comparación de los 4 classifiers
└── test_classification_report.csv # Métricas detalladas del mejor modelo
```

### 6. Evaluar modelo pre-entrenado

```bash
# Evaluar CLIP (mejor modelo, 89.0% F1)
python -m src.ml.evaluate --model_dir outputs/comparison/models/clip

# Evaluar DINOv2
python -m src.ml.evaluate --model_dir outputs/comparison/models/dinov2

# Evaluar Place365
python -m src.ml.evaluate --model_dir outputs/comparison/models/place365
```

Los gráficos y métricas se guardan junto al modelo:

```
outputs/comparison/models/clip/
├── best_classifier.joblib              ← de compare_models
├── model_comparison.csv                ← de compare_models
├── classification_report.csv           ← de compare_models (train)
├── test_classification_report.csv      ← de evaluate (test completo)
├── test_per_class_metrics.csv          ← de evaluate (por clase)
├── confusion_matrix.png                ← de evaluate
├── confusion_matrix_normalized.png     ← de evaluate
└── per_class_metrics.png               ← de evaluate
```

### 7. Comparar modelos (CLIP vs DINOv2 vs Place365)

```bash
python -m src.ml.compare_models \
    --data_dir data/splits \
    --output_dir outputs/comparison \
    --models clip dinov2 place365 clip_zeroshot
```

**Artefactos generados en `outputs/comparison/`:**

```
outputs/comparison/
├── model_comparison.csv          # Métricas de todos los modelos (Accuracy, F1, Precision, Recall)
├── comparison.png                # Gráfico de barras comparativo
├── embeddings/                   # Embeddings extraídos por modelo
│   ├── clip/
│   │   ├── train/                # X_embeddings.npy + y_labels.npy
│   │   └── test/
│   ├── dinov2/
│   │   ├── train/
│   │   └── test/
│   └── place365/
│       ├── train/
│       └── test/
└── models/                       # Clasificadores entrenados por modelo
    ├── clip/
    │   ├── best_classifier.joblib    # SVM RBF (89.0% F1)
    │   ├── model_comparison.csv
    │   └── test_classification_report.csv
    ├── dinov2/
    │   ├── best_classifier.joblib    # SVM RBF (88.4% F1)
    │   ├── model_comparison.csv
    │   └── test_classification_report.csv
    └── place365/
        ├── best_classifier.joblib    # XGBoost (86.2% F1)
        ├── model_comparison.csv
        └── test_classification_report.csv
```

### 8. Detectar objetos con YOLO

```bash
python -m src.detection.detector \
    --image data/splits/test/cocina/example.jpg \
    --model_path models/yolo_furniture/best.pt
```

### 9. Web app

```bash
streamlit run src/web/app.py
```

La app permite elegir entre:
- **Zero-shot CLIP**: Clasificación sin entrenamiento
- **ML Classifier (SVM)**: Modelo entrenado
- **ML + YOLO Objects**: SVM + descripción mejorada con objetos detectados

## Resultados

### Comparación de modelos (test split, 17 clases, 1,520 imágenes)

| Modelo | Best Classifier | Accuracy | F1 (weighted) | Precision | Recall |
|---|---|---:|---:|---:|---:|
| **CLIP ViT-B/32 + SVM RBF** | SVM RBF | **89.0%** | **0.890** | **0.891** | **0.890** |
| DINOv2 ViT-B/14 + SVM RBF | SVM RBF | 88.3% | 0.884 | 0.887 | 0.883 |
| Place365 ResNet50 + XGBoost | XGBoost | 86.2% | 0.862 | 0.864 | 0.862 |
| CLIP Zero-shot | — | 72.2% | 0.752 | — | — |

### Métricas por clase (CLIP + SVM RBF, test split)

| Clase | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| gimnasio | 1.000 | 1.000 | 1.000 | 39 |
| baño | 0.939 | 0.959 | 0.949 | 97 |
| cocina | 0.955 | 0.935 | 0.945 | 248 |
| piscina | 0.906 | 0.960 | 0.932 | 50 |
| dormitorio | 0.894 | 0.899 | 0.897 | 198 |
| sala | 0.923 | 0.881 | 0.901 | 379 |
| walking closet | 0.842 | 0.941 | 0.889 | 17 |
| pasadizo | 0.865 | 0.914 | 0.889 | 70 |
| exterior | 0.859 | 0.888 | 0.873 | 89 |
| comedor | 0.818 | 0.920 | 0.866 | 88 |
| lavandería | 0.873 | 0.842 | 0.857 | 57 |
| estacionamiento | 0.833 | 0.833 | 0.833 | 24 |
| recepción | 0.733 | 0.917 | 0.815 | 12 |
| terraza | 0.707 | 0.829 | 0.763 | 35 |
| balcón | 0.767 | 0.733 | 0.750 | 45 |
| patio/jardín | 0.765 | 0.684 | 0.722 | 57 |
| área de trabajo | 0.750 | 0.600 | 0.667 | 15 |

### CLIP Zero-shot por clase

| Clase | Accuracy | Confianza promedio |
|---|---:|---:|
| gimnasio | 100.0% | 97.2% |
| walking_closet | 100.0% | 91.9% |
| bano | 95.1% | 84.0% |
| comedor | 92.7% | 65.7% |
| terraza | 89.0% | 71.3% |
| area_trabajo | 83.5% | 66.2% |
| cocina | 83.3% | 67.9% |
| balcon | 77.5% | 63.2% |
| sala | 76.8% | 56.4% |
| patio_jardin | 74.1% | 61.8% |
| recepcion | 72.7% | 62.3% |
| pasadizo | 69.7% | 75.4% |
| dormitorio | 67.3% | 63.1% |
| lavanderia | 65.3% | 68.5% |
| estacionamiento | 59.5% | 69.7% |
| exterior | 34.9% | 58.8% |
| piscina | 58.2% | 64.1% |

### CLIP vs DINOv2 vs Place365

| Backbone | Embed dim | Mejor clasificador | Accuracy | F1 | Precision | Recall |
|---|---:|---|---:|---:|---:|---:|
| **CLIP ViT-B/32** | 512 | SVM RBF | **89.0%** | **0.890** | **0.891** | **0.890** |
| DINOv2 ViT-B/14 | 768 | SVM RBF | 88.3% | 0.884 | 0.887 | 0.883 |
| Place365 ResNet50 | 2048 | XGBoost | 86.2% | 0.862 | 0.864 | 0.862 |
| CLIP Zero-shot | — | — | 72.2% | 0.752 | 0.849 | 0.722 |

> **Nota:** YOLO11 (mAP50=32.6%) se usa como complemento en la web app para describir objetos detectados en cada categoría. No es parte del modelo principal de clasificación.

## Métricas calculadas

| Métrica | Descripción |
|---|---|
| Accuracy | Proporción total de predicciones correctas |
| Precision | Exactitud de las predicciones positivas por clase |
| Recall | Cobertura de ejemplos reales por clase |
| F1-score | Media armónica entre Precision y Recall |
| Confusion Matrix | Distribución de aciertos y errores por clase (normalizada y absoluta) |
| Per-class Report | Métricas detalladas por cada una de las 17 clases |
| Support | Número de ejemplos reales por clase en el test set |

## Archivos generados

Los archivos de evaluación se guardan junto al modelo en `outputs/comparison/models/`:

```
outputs/comparison/models/{clip,dinov2,place365}/
├── best_classifier.joblib              # Modelo entrenado (Pipeline)
├── model_comparison.csv                # Métricas del backbone
├── test_classification_report.csv      # Métricas detalladas (test)
├── test_per_class_metrics.csv          # Métricas por clase
├── confusion_matrix.png                # Matriz de confusión
├── confusion_matrix_normalized.png     # Matriz normalizada
└── per_class_metrics.png               # Gráfico de barras por clase
```

Otros archivos generados:
- `outputs/comparison/model_comparison.csv` — Comparación numérica de todos los modelos
- `outputs/comparison/comparison.png` — Gráfico de barras comparativo

## Limitaciones

- CLIP zero-shot puede confundir clases visualmente similares (sala/comedor, terraza/balcon).
- El rendimiento depende de la cantidad y diversidad de imágenes por clase.
- Las imágenes inmobiliarias pueden incluir espacios mixtos (cocina-comedor).
- El scraping de Urbania requiere Selenium y puede fallar con cambios en el sitio web.

## Trabajo futuro

- Fine-tuning de CLIP o uso de SigLIP para mejorar zero-shot.
- Manejo de imágenes multi-etiqueta para ambientes integrados.
- Calibración de probabilidades y umbrales de confianza.
- Despliegue del modelo en producción (API REST).
- Más imágenes para YOLO (mejorar mAP50).
- Evaluar con dataset MIT Indoor Scenes para validación externa.

## Nota ética

No se hace scraping no autorizado de plataformas privadas. Las imágenes de Urbania se obtienen de listings públicos. Para entrenamiento, se recomienda usar datasets públicos con licencia adecuada o imágenes propias.

## Fuentes de datos adicionales

### MIT Indoor Scenes Dataset

Para mejorar el entrenamiento con más imágenes de ambientes interiores:

- **URL**: https://web.mit.edu/torralba/www/indoor.html
- **Descripción**: Dataset de 15,620 imágenes categorizadas en 67 categorías de interiores
- **Uso**: Descargar imágenes y organizarlas en carpetas `data/raw/<clase>/`
- **Categorías relevantes**: bedroom, kitchen, bathroom, living room, dining room, office, garage, etc.

```bash
# Después de descargar, mover imágenes a las carpetas correspondientes
mv downloaded_images/bedroom/* data/raw/dormitorio/
mv downloaded_images/kitchen/* data/raw/cocina/
# etc.
```
