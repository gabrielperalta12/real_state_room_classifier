"""
Web app para clasificación de imágenes con CLIP + detección de objetos con YOLO.

Uso:
    streamlit run src/web/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.clip.loader import load_clip
from src.labels import DISPLAY_LABELS, ZERO_SHOT_PROMPTS


# ─── Constants ───────────────────────────────────────────────────────────────────

ML_MODELS = {
    "SVM (CLIP)": "outputs/comparison/models/clip/best_classifier.joblib",
    "SVM (DINOv2)": "outputs/comparison/models/dinov2/best_classifier.joblib",
    "XGBoost (Place365)": "outputs/comparison/models/place365/best_classifier.joblib",
}

MODEL_EMBED_TYPE = {
    "clip": "clip",
    "dinov2": "dinov2",
    "place365": "place365",
}

# Colors
COLORS = {
    "primary": "#1e40af",
    "primary_light": "#3b82f6",
    "success": "#059669",
    "success_bg": "#d1fae5",
    "warning": "#d97706",
    "warning_bg": "#fef3c7",
    "danger": "#dc2626",
    "danger_bg": "#fee2e2",
    "gray": "#6b7280",
    "gray_light": "#f3f4f6",
    "gray_border": "#e5e7eb",
}


# ─── Custom CSS ──────────────────────────────────────────────────────────────────

def inject_custom_css():
    st.markdown("""
    <style>
    /* ── Global ──────────────────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    /* Gray background so white cards stand out */
    .stApp > header {
        background: transparent;
    }
    .main .block-container {
        background: #f1f5f9;
        min-height: 100vh;
    }

    /* ── Header ──────────────────────────────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .app-header h1 {
        color: white !important;
        font-size: 1.8rem;
        margin: 0 0 0.3rem 0;
        padding: 0;
    }
    .app-header p {
        color: rgba(255,255,255,0.85);
        margin: 0;
        font-size: 0.95rem;
    }

    /* ── Cards ───────────────────────────────────────────────────────── */
    .room-card {
        background: white;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .room-card:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.18);
        transform: translateY(-2px);
    }

    /* ── Category badges ─────────────────────────────────────────────── */
    .category-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: white;
        border: 1px solid #93c5fd;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #1e40af;
        font-weight: 600;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(30,64,175,0.15);
    }
    .category-badge .count {
        background: #1e40af;
        color: white;
        border-radius: 12px;
        padding: 1px 8px;
        font-size: 0.75rem;
    }

    /* ── Confidence bar ──────────────────────────────────────────────── */
    .confidence-bar-container {
        background: #e5e7eb;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin: 6px 0;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }
    .confidence-high { background: #059669; }
    .confidence-medium { background: #d97706; }
    .confidence-low { background: #dc2626; }

    .confidence-label {
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 0 0 0;
    }
    .confidence-high-text { color: #059669; }
    .confidence-medium-text { color: #d97706; }
    .confidence-low-text { color: #dc2626; }

    /* ── Metric cards ────────────────────────────────────────────────── */
    .metric-card {
        background: white;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
        margin: 0;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        font-weight: 600;
        color: #374151;
    }

    /* ── Image grid ──────────────────────────────────────────────────── */
    .stImage > img {
        border-radius: 8px;
    }

    /* ── Divider ─────────────────────────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 2px solid #e5e7eb;
        margin: 1.5rem 0;
    }

    /* ── Navigation pills ────────────────────────────────────────────── */
    .nav-pill {
        display: inline-block;
        background: #f0f9ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #1e40af;
        text-decoration: none;
        margin: 2px;
        transition: all 0.2s;
    }
    .nav-pill:hover {
        background: #1e40af;
        color: white;
    }

    /* ── Probabilities table ──────────────────────────────────────────── */
    .prob-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid #f3f4f6;
    }
    .prob-item:last-child {
        border-bottom: none;
    }
    .prob-label {
        font-size: 0.85rem;
        color: #374151;
    }
    .prob-value {
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* ── YOLO objects tag ────────────────────────────────────────────── */
    .object-tag {
        display: inline-block;
        background: #faf5ff;
        border: 1px solid #e9d5ff;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #7c3aed;
        margin: 2px;
    }

    /* ── Empty state ─────────────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #64748b;
        background: white;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .empty-state .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Classification functions ────────────────────────────────────────────────────

def classify_zero_shot(
    image_paths: list[Path],
    model,
    processor,
    device,
    labels: list[str],
    prompts: list[str],
) -> list[dict]:
    import torch
    results = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                inputs = processor(text=prompts, images=rgb_image, return_tensors="pt", padding=True)

            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()

            ranked_indices = probabilities.argsort()[::-1]
            best_index = int(ranked_indices[0])

            results.append({
                "image_path": image_path,
                "predicted_label": labels[best_index],
                "display_label": DISPLAY_LABELS[labels[best_index]],
                "probability": float(probabilities[best_index]),
                "top_3": [
                    {
                        "label": DISPLAY_LABELS[labels[int(idx)]],
                        "probability": float(probabilities[int(idx)])
                    }
                    for idx in ranked_indices[:3]
                ]
            })
        except Exception as e:
            results.append({
                "image_path": image_path,
                "predicted_label": "error",
                "display_label": f"Error: {e}",
                "probability": 0.0,
                "top_3": []
            })

    return results


def classify_ml(
    image_paths: list[Path],
    clip_model,
    processor,
    device,
    ml_model_path: str = "outputs/comparison/models/clip/best_classifier.joblib",
) -> list[dict]:
    import torch
    import joblib

    artifact = joblib.load(ml_model_path)
    classifier = artifact["model"]
    label_encoder = artifact.get("label_encoder", None)

    embed_type = "clip"
    for key, etype in MODEL_EMBED_TYPE.items():
        if key in ml_model_path.lower():
            embed_type = etype
            break

    if embed_type == "dinov2":
        @st.cache_resource
        def load_dinov2():
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            mdl = AutoModel.from_pretrained("facebook/dinov2-base")
            mdl = mdl.to(device).eval()
            return proc, mdl
        dino_processor, dino_model = load_dinov2()

    elif embed_type == "place365":
        @st.cache_resource
        def load_place365():
            import torchvision.models as tmodels
            import torch.nn as nn
            import urllib.request
            weight_path = "/tmp/resnet50_places365.pth.tar"
            if not Path(weight_path).exists():
                url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
                urllib.request.urlretrieve(url, weight_path)
            model = tmodels.resnet50(num_classes=365)
            checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
            state_dict = checkpoint["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model = nn.Sequential(*list(model.children())[:-1])
            model = model.to(device).eval()
            return model
        place365_model = load_place365()

    results = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")

                if embed_type == "dinov2":
                    inputs = dino_processor(images=rgb_image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = dino_model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :]
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                elif embed_type == "place365":
                    import torchvision.transforms as transforms
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    tensor = transform(rgb_image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = place365_model(tensor).flatten(1)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                else:
                    inputs = processor(images=rgb_image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
                        pooled_output = vision_outputs[1]
                        embedding = clip_model.visual_projection(pooled_output)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            embedding_np = embedding.squeeze(0).cpu().numpy().reshape(1, -1)

            if hasattr(classifier, "predict_proba"):
                probabilities = classifier.predict_proba(embedding_np)[0]
                indices = probabilities.argsort()[::-1][:3]

                top_3 = []
                for idx in indices:
                    if label_encoder is not None:
                        label = label_encoder.inverse_transform([idx])[0]
                    else:
                        label = str(idx)
                    top_3.append({
                        "label": DISPLAY_LABELS.get(label, label),
                        "probability": float(probabilities[idx])
                    })

                best_label = top_3[0]["label"]
                best_prob = top_3[0]["probability"]

                predicted_key = ""
                for key, display in DISPLAY_LABELS.items():
                    if display == best_label:
                        predicted_key = key
                        break

                results.append({
                    "image_path": image_path,
                    "predicted_label": predicted_key,
                    "display_label": best_label,
                    "probability": best_prob,
                    "top_3": top_3,
                })
            else:
                prediction = classifier.predict(embedding_np)[0]
                if label_encoder is not None:
                    prediction = label_encoder.inverse_transform([prediction])[0]

                results.append({
                    "image_path": image_path,
                    "predicted_label": prediction,
                    "display_label": DISPLAY_LABELS.get(prediction, prediction),
                    "probability": 1.0,
                    "top_3": [{
                        "label": DISPLAY_LABELS.get(prediction, prediction),
                        "probability": 1.0
                    }],
                })

        except Exception as e:
            results.append({
                "image_path": image_path,
                "predicted_label": "error",
                "display_label": f"Error: {e}",
                "probability": 0.0,
                "top_3": []
            })

    return results


def detect_objects(
    image_paths: list[Path],
    yolo_model_path: str | None = None,
    confidence: float = 0.25,
) -> list[dict]:
    from src.detection.detector import FurnitureDetector

    detector = FurnitureDetector(model_path=yolo_model_path, confidence=confidence)

    results = []
    for image_path in image_paths:
        try:
            detections = detector.detect(image_path, conf=confidence)
            description = detector.describe_room(detections)
            summary = detector.get_object_summary(detections)

            results.append({
                "image_path": image_path,
                "detections": detections,
                "description": description,
                "summary": summary,
            })
        except Exception as e:
            results.append({
                "image_path": image_path,
                "detections": [],
                "description": f"Error: {e}",
                "summary": {"total_objects": 0, "unique_classes": 0, "by_class": {}},
            })

    return results


def group_by_label(results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        label = result["display_label"]
        if label not in grouped:
            grouped[label] = []
        grouped[label].append(result)
    return grouped


# ─── UI helpers ──────────────────────────────────────────────────────────────────

def confidence_color(prob: float) -> str:
    if prob >= 0.8:
        return "high"
    elif prob >= 0.5:
        return "medium"
    return "low"


def render_confidence_bar(prob: float):
    level = confidence_color(prob)
    pct = int(prob * 100)
    st.markdown(f"""
    <div class="confidence-bar-container">
        <div class="confidence-bar confidence-{level}" style="width: {pct}%"></div>
    </div>
    <p class="confidence-label confidence-{level}-text">{pct}%</p>
    """, unsafe_allow_html=True)


def render_top3(top_3: list[dict]):
    for item in top_3:
        level = confidence_color(item["probability"])
        color_map = {"high": "#059669", "medium": "#d97706", "low": "#dc2626"}
        color = color_map[level]
        pct = f"{item['probability']:.1%}"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid #f3f4f6">'
            f'<span style="font-size:0.85rem;color:#374151">{item["label"]}</span>'
            f'<span style="font-size:0.85rem;font-weight:600;color:{color}">{pct}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Room Classifier",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # ── Header ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>🏠 Room Classifier</h1>
        <p>Clasificación automática de ambientes inmobiliarios con CLIP, DINOv2 y Place365</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")

        model_option = st.selectbox(
            "Modo de clasificación",
            options=["Zero-shot CLIP", "ML Classifier", "ML + YOLO Objects"],
            help="Zero-shot: CLIP compara imagen vs texto. ML: Embeddings + classifier. ML + YOLO: classifier + detección de muebles."
        )

        ml_model_name = None
        ml_model_path = None
        if model_option in ["ML Classifier", "ML + YOLO Objects"]:
            st.markdown("---")
            st.markdown("#### 🧠 Modelo ML")
            ml_model_name = st.selectbox(
                "Modelo",
                options=list(ML_MODELS.keys()),
                label_visibility="collapsed",
            )
            ml_model_path = ML_MODELS[ml_model_name]

            if Path(ml_model_path).exists():
                st.success("Modelo cargado", icon="✅")

        if model_option == "Zero-shot CLIP":
            st.markdown("---")
            st.markdown("#### 🎯 Parámetros CLIP")
            confidence_threshold = st.slider(
                "Umbral de confianza",
                min_value=0.0, max_value=1.0, value=0.45, step=0.05,
            )
        else:
            confidence_threshold = 0.45

        if model_option == "ML + YOLO Objects":
            st.markdown("---")
            st.markdown("#### 🪑 Detección YOLO")
            yolo_confidence = st.slider(
                "Umbral YOLO",
                min_value=0.05, max_value=0.95, value=0.25, step=0.05,
            )
            yolo_model_path = Path("models/yolo_furniture/best.pt")
            if yolo_model_path.exists():
                st.success("YOLO custom cargado", icon="✅")
            else:
                st.warning("Usando YOLO11 preentrenado (COCO)")

    # ── Load CLIP ───────────────────────────────────────────────────────────────
    @st.cache_resource
    def load_clip_model():
        return load_clip("openai/clip-vit-base-patch32")

    with st.spinner("Cargando modelo CLIP..."):
        clip_model, processor, device = load_clip_model()

    labels = list(ZERO_SHOT_PROMPTS.keys())
    prompts = [ZERO_SHOT_PROMPTS[label] for label in labels]

    # ── Upload ──────────────────────────────────────────────────────────────────
    st.markdown("#### 📤 Subir imágenes")

    uploaded_files = st.file_uploader(
        "Arrastra o selecciona imágenes de la propiedad",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📸</div>
            <h3>Sin imágenes</h3>
            <p>Sube fotos de una propiedad para clasificar los ambientes automáticamente.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Process ─────────────────────────────────────────────────────────────────
    temp_dir = Path(tempfile.mkdtemp())
    image_paths = []
    for uploaded_file in uploaded_files:
        temp_path = temp_dir / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())
        image_paths.append(temp_path)

    if model_option == "ML + YOLO Objects":
        with st.spinner(f"Clasificando con {ml_model_name} + detectando muebles..."):
            ml_results = classify_ml(image_paths, clip_model, processor, device, ml_model_path)
            yolo_results = detect_objects(
                image_paths,
                yolo_model_path=str(yolo_model_path) if yolo_model_path.exists() else None,
                confidence=yolo_confidence,
            )

        results = []
        for ml_res, yolo_res in zip(ml_results, yolo_results):
            detections = yolo_res.get("detections", [])
            room_type = ml_res.get("display_label", "unknown")
            if detections:
                unique_objs = list(set([d["class_name"] for d in detections]))
                obj_list = unique_objs[:5]
                enhanced_description = f"{room_type.title()} con {', '.join(obj_list)}"
            else:
                enhanced_description = f"{room_type.title()} (sin objetos detectados)"

            results.append({
                **ml_res,
                "detections": detections,
                "description": yolo_res.get("description", ""),
                "enhanced_description": enhanced_description,
                "yolo_summary": yolo_res.get("summary", {}),
            })

    elif model_option == "Zero-shot CLIP":
        with st.spinner("Clasificando con CLIP zero-shot..."):
            results = classify_zero_shot(image_paths, clip_model, processor, device, labels, prompts)
    else:
        with st.spinner(f"Clasificando con {ml_model_name}..."):
            results = classify_ml(image_paths, clip_model, processor, device, ml_model_path)

    grouped = group_by_label(results)
    label_list = list(grouped.keys())

    # ── Summary metrics ─────────────────────────────────────────────────────────
    st.markdown("---")

    avg_confidence = sum(r["probability"] for r in results) / len(results)
    total_objects = 0
    if model_option == "ML + YOLO Objects":
        total_objects = sum(r.get("yolo_summary", {}).get("total_objects", 0) for r in results)

    if model_option == "ML + YOLO Objects":
        m1, m2, m3, m4, m5 = st.columns(5)
    else:
        m1, m2, m3, m4 = st.columns(4)
        m5 = None

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(results)}</p>
            <p class="metric-label">Imágenes</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(grouped)}</p>
            <p class="metric-label">Clases</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        conf_color = confidence_color(avg_confidence)
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value confidence-{conf_color}-text">{avg_confidence:.0%}</p>
            <p class="metric-label">Confianza prom.</p>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        display_model = ml_model_name if ml_model_name else model_option
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="font-size:1rem">{display_model}</p>
            <p class="metric-label">Modelo</p>
        </div>
        """, unsafe_allow_html=True)
    if m5:
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{total_objects}</p>
                <p class="metric-label">Objetos</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Category navigation ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📸 Clasificación por ambiente")

    if label_list:
        nav_html = '<div style="margin-bottom:1rem">'
        for label in label_list:
            count = len(grouped[label])
            nav_html += f'<span class="category-badge">{label} <span class="count">{count}</span></span> '
        nav_html += '</div>'
        st.markdown(nav_html, unsafe_allow_html=True)

    # ── Image gallery ───────────────────────────────────────────────────────────
    for label in label_list:
        items = grouped[label]

        # Category header
        category_objects = {}
        if model_option == "ML + YOLO Objects":
            for item in items:
                for det in item.get("detections", []):
                    cls = det["class_name"]
                    category_objects[cls] = category_objects.get(cls, 0) + 1

        st.markdown(f"""
        <div style="margin-top:1.5rem; margin-bottom:0.8rem; padding-bottom:0.5rem; border-bottom:2px solid #e5e7eb;">
            <span class="category-badge">{label} <span class="count">{len(items)} fotos</span></span>
        </div>
        """, unsafe_allow_html=True)

        # Object tags for category
        if model_option == "ML + YOLO Objects" and category_objects:
            tags_html = ""
            for obj, count in sorted(category_objects.items(), key=lambda x: -x[1])[:6]:
                tags_html += f'<span class="object-tag">{obj} ({count}x)</span> '
            if len(category_objects) > 6:
                tags_html += f'<span class="object-tag">+{len(category_objects)-6} más</span>'
            st.markdown(tags_html, unsafe_allow_html=True)

        # Image grid
        cols = st.columns(min(4, len(items)))
        for idx, item in enumerate(items):
            with cols[idx % len(cols)]:
                st.image(str(item["image_path"]), use_container_width=True)

                # Confidence bar
                render_confidence_bar(item["probability"])

                # Enhanced description (YOLO)
                if model_option == "ML + YOLO Objects" and item.get("enhanced_description"):
                    st.caption(item["enhanced_description"])

                # Probabilities expander
                if item.get("top_3"):
                    with st.expander("Ver probabilidades"):
                        render_top3(item["top_3"])

                # YOLO detections expander
                if model_option == "ML + YOLO Objects" and item.get("detections"):
                    unique_objs = list(set([d["class_name"] for d in item["detections"]]))
                    with st.expander(f"🪑 {len(unique_objs)} objetos detectados"):
                        obj_counts = {}
                        for det in item["detections"]:
                            cls = det["class_name"]
                            obj_counts[cls] = obj_counts.get(cls, 0) + 1
                        for obj_name, count in sorted(obj_counts.items(), key=lambda x: -x[1]):
                            st.write(f"- **{obj_name}** ({count}x)")

    # ── Distribution table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Distribución por clase")

    summary_data = {
        "Clase": list(grouped.keys()),
        "Cantidad": [len(items) for items in grouped.values()],
        "Confianza prom.": [
            f"{sum(i['probability'] for i in items) / len(items):.0%}"
            for items in grouped.values()
        ]
    }
    st.dataframe(summary_data, use_container_width=True, hide_index=True)

    # ── YOLO summary ───────────────────────────────────────────────────────────
    if model_option == "ML + YOLO Objects":
        st.markdown("#### 🪑 Resumen de objetos detectados")

        all_objects: dict[str, int] = {}
        for r in results:
            for cls, count in r.get("yolo_summary", {}).get("by_class", {}).items():
                all_objects[cls] = all_objects.get(cls, 0) + count

        if all_objects:
            obj_data = {
                "Objeto": list(all_objects.keys()),
                "Cantidad": list(all_objects.values()),
            }
            st.dataframe(obj_data, use_container_width=True, hide_index=True)
        else:
            st.info("No se detectaron objetos en las imágenes.")


if __name__ == "__main__":
    main()
