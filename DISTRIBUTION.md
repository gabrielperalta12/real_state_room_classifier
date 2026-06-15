# Guía de Distribución

## ¿Qué va al repo de Git?

Solo **código fuente** (~2MB):

```
real_estate_room_classifier/
├── src/                          # Todo el código Python
├── requirements.txt
├── README.md
├── setup.sh                      # Script de setup automático
├── .gitignore
└── .gitkeep                      # Mantiene carpetas vacías
```

## ¿Qué va al ZIP para Google Drive?

**Datos + modelos entrenados** (~2.9GB):

```
project_data.zip
├── data/
│   ├── raw/                      # 10,150 imágenes etiquetadas
│   ├── splits/                   # Train (14,492) / Test (1,520)
│   └── preprocessed/             # Listings scrapeados
├── models/
│   ├── yolo_furniture/
│   │   └── best.pt               # YOLO fine-tuned (21MB)
│   └── merged_50ep/              # YOLO experimento anterior
└── outputs/
    └── comparison/
        ├── models/
        │   ├── clip/
        │   │   ├── best_classifier.joblib      # SVM CLIP (89% F1)
        │   │   └── *.csv, *.png
        │   ├── dinov2/
        │   │   ├── best_classifier.joblib      # SVM DINOv2 (88% F1)
        │   │   └── *.csv, *.png
        │   └── place365/
        │       ├── best_classifier.joblib      # XGBoost Place365 (86% F1)
        │       └── *.csv, *.png
        ├── embeddings/                        # Embeddings .npy
        ├── model_comparison.csv
        └── comparison.png
```

## ¿Qué NO subir (queda en tu máquina)?

| Carpeta | Tamaño | Contenido |
|---------|--------|-----------|
| `.venv/` | ~8GB | Entorno virtual |
| `yolo11n.pt` | ~5MB | Modelo base YOLO (descargable) |
| `yolo26n.pt` | ~5MB | Modelo base YOLO (descargable) |

## Setup automático (en otra máquina)

### Opción 1: Script setup.sh (recomendado)

```bash
# 1. Clonar el repo
git clone https://github.com/gabrielperalta12/real_state_room_classifier.git
cd real_state_room_classifier

# 2. Subir project_data.zip a Google Drive
# 3. Obtener el FILE_ID del link de compartir
#    https://drive.google.com/file/d/<FILE_ID>/view
# 4. Ejecutar setup
bash setup.sh <FILE_ID>
```

El script:
1. Descarga el ZIP de Google Drive
2. Extrae datos, modelos y embeddings
3. Crea entorno virtual
4. Instala dependencias
5. Verifica que todo funcione

### Opción 2: Manual

```bash
# Clonar repo
git clone https://github.com/gabrielperalta12/real_state_room_classifier.git
cd real_estate_room_classifier

# Crear entorno
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Descargar y extraer datos
# (subir project_data.zip a Google Drive, descargar, descomprimir)
unzip project_data.zip

# Verificar
python -c "from src.labels import ROOM_LABELS; print(f'{len(ROOM_LABELS)} labels loaded')"
```

## Para deploy en Streamlit Cloud

Solo necesitas el repo de Git (sin el ZIP):
- `src/web/app.py`
- `src/` (código completo)
- `requirements.txt`
- `outputs/comparison/models/*/best_classifier.joblib` (~3MB total)
- `report/references/*.png`

Los modelos `.joblib` se incluyen en el repo Git porque son pequeños (~3MB total).

## Crear el ZIP

```bash
# ZIP completo (datos + modelos + embeddings) ~2.9GB
zip -r /tmp/project_data.zip data/ models/ outputs/ \
    -x "data/scrapped/*" "data/pending_review/*"
```
