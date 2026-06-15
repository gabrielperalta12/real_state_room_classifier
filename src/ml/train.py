"""
Entrena clasificadores ML sobre embeddings CLIP.

Uso:
    python -m src.ml.train --input_dir data/embeddings --output_dir models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

from ..config import DEFAULT_OUTPUT_DIR, DEFAULT_MODEL_DIR
from ..labels import DISPLAY_LABELS
from ..utils import ensure_dir, validate_dir


def build_candidate_models(random_state: int = 42, use_gpu: bool = False) -> dict[str, Pipeline]:
    """Retorna candidatos de clasificadores para embeddings CLIP."""
    tree_method = "hist"
    device = "cpu" if not use_gpu else "cuda"
    
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state
            )),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=random_state
            )),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", XGBClassifier(
                n_estimators=200,
                eval_metric="mlogloss",
                random_state=random_state,
                tree_method=tree_method,
                device=device,
            )),
        ]),
    }


def tune_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    n_iter: int = 50,
    cv: int = 3,
    use_gpu: bool = False,
) -> Pipeline:
    """
    Tunea XGBoost con RandomizedSearchCV y regularización.

    Args:
        X_train: Embeddings de entrenamiento.
        y_train: Labels codificados.
        X_test: Embeddings de test.
        y_test: Labels de test.
        random_state: Seed para reproducibilidad.
        n_iter: Número de combinaciones de parámetros a probar.
        cv: Número de folds para cross-validation.
        use_gpu: Si True, usa GPU para entrenamiento.

    Returns:
        Pipeline con el mejor modelo encontrado.
    """
    print(f"\nTuning XGBoost with RandomizedSearchCV + Regularization...")
    print(f"  n_iter: {n_iter}, cv: {cv}, GPU: {use_gpu}")
    print("=" * 60)

    tree_method = "hist"
    device = "cpu" if not use_gpu else "cuda"

    # Espacio de búsqueda con regularización
    param_distributions = {
        "classifier__n_estimators": randint(100, 500),
        "classifier__max_depth": randint(3, 10),
        "classifier__learning_rate": uniform(0.01, 0.3),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__min_child_weight": randint(1, 10),
        # Regularización L1 y L2
        "classifier__reg_alpha": uniform(0, 1),
        "classifier__reg_lambda": uniform(0, 2),
        # Regularización adicional
        "classifier__gamma": uniform(0, 0.5),
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            eval_metric="mlogloss",
            random_state=random_state,
            tree_method=tree_method,
            device=device,
        )),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        random_state=random_state,
        n_jobs=-1,
        verbose=2,
    )

    print(f"\nStarting tuning...")
    print("-" * 60)
    
    search.fit(X_train, y_train)
    
    print("-" * 60)
    print(f"\nBest parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    # Evaluar en test
    best_pipeline = search.best_estimator_
    test_score = best_pipeline.score(X_test, y_test)
    print(f"Test F1: {test_score:.4f}")

    return best_pipeline


def train_classifier(
    input_dir: str | Path = DEFAULT_OUTPUT_DIR / "embeddings",
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_xgboost: bool = False,
    n_iter: int = 50,
    cv: int = 3,
    use_gpu: bool = False,
) -> dict:
    """
    Entrena múltiples clasificadores y guarda el mejor.

    Soporta dos modos:
      1. Directorio con train/: Usa embeddings pre-divididos (sin data leakage)
      2. Directorio simple: Hace train_test_split interno

    Args:
        input_dir: Directorio de embeddings.
        output_dir: Directorio de salida para modelos.
        test_size: Fracción de test split (solo si no hay train/ pre-dividido).
        random_state: Seed para reproducibilidad.
        tune_xgboost: Si True, tunea XGBoost con RandomizedSearchCV.
        n_iter: Número de combinaciones de parámetros para tuning.
        cv: Número de folds para cross-validation.
        use_gpu: Si True, usa GPU para XGBoost.

    Returns:
        Diccionario con métricas y ruta del modelo guardado.
    """
    input_path = validate_dir(input_dir)
    output_path = ensure_dir(output_dir)

    # Verificar si hay train/test pre-dividido
    train_dir = input_path / "train"
    test_dir = input_path / "test"
    
    if train_dir.is_dir() and test_dir.is_dir():
        # Modo train/test pre-dividido
        print(f"\nUsing pre-divided train/test split")
        X_train = np.load(train_dir / "X_embeddings.npy")
        y_train = np.load(train_dir / "y_labels.npy", allow_pickle=True)
        X_test = np.load(test_dir / "X_embeddings.npy")
        y_test = np.load(test_dir / "y_labels.npy", allow_pickle=True)
        
        print(f"Train: {X_train.shape[0]} images")
        print(f"Test: {X_test.shape[0]} images")
    else:
        # Modo original: split interno
        print(f"\nNo pre-divided split found. Using internal train_test_split")
        X = np.load(input_path / "X_embeddings.npy")
        y = np.load(input_path / "y_labels.npy", allow_pickle=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"Train: {X_train.shape[0]} images")
        print(f"Test: {X_test.shape[0]} images")

    print(f"Classes: {len(set(y_train))}")
    print(f"GPU: {use_gpu}")

    # Encode labels to integers for XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    class_names = label_encoder.classes_

    # Entrenar candidatos
    candidates = build_candidate_models(random_state, use_gpu)
    results = []
    best_f1 = -1.0
    best_model = None
    best_name = None

    for name, pipeline in candidates.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train_encoded)
        y_pred = pipeline.predict(X_test)

        # Convert back to string labels for classification report
        y_test_labels = label_encoder.inverse_transform(y_test_encoded)
        y_pred_labels = label_encoder.inverse_transform(y_pred)

        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)
        f1_weighted = report["weighted avg"]["f1-score"]

        results.append({
            "model": name,
            "accuracy": round(accuracy, 4),
            "f1_weighted": round(f1_weighted, 4),
            "precision_weighted": round(report["weighted avg"]["precision"], 4),
            "recall_weighted": round(report["weighted avg"]["recall"], 4),
        })

        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")

        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_model = pipeline
            best_name = name

    # XGBoost tuning (opcional)
    if tune_xgboost:
        xgb_tuned = tune_xgboost_model(X_train, y_train_encoded, X_test, y_test_encoded, random_state, n_iter, cv, use_gpu)
        y_pred_xgb = xgb_tuned.predict(X_test)

        y_test_labels = label_encoder.inverse_transform(y_test_encoded)
        y_pred_xgb_labels = label_encoder.inverse_transform(y_pred_xgb)

        accuracy_xgb = accuracy_score(y_test_labels, y_pred_xgb_labels)
        report_xgb = classification_report(y_test_labels, y_pred_xgb_labels, output_dict=True, zero_division=0)
        f1_xgb = report_xgb["weighted avg"]["f1-score"]

        results.append({
            "model": "xgboost_tuned",
            "accuracy": round(accuracy_xgb, 4),
            "f1_weighted": round(f1_xgb, 4),
            "precision_weighted": round(report_xgb["weighted avg"]["precision"], 4),
            "recall_weighted": round(report_xgb["weighted avg"]["recall"], 4),
        })

        print(f"\nXGBoost Tuned - Accuracy: {accuracy_xgb:.1%}, F1: {f1_xgb:.4f}")

        # Auto-replace if tuned XGBoost is better
        if f1_xgb > best_f1:
            print(f"\nXGBoost tuned ({f1_xgb:.4f}) > {best_name} ({best_f1:.4f}). Replacing best model.")
            best_f1 = f1_xgb
            best_model = xgb_tuned
            best_name = "xgboost_tuned"
        else:
            print(f"\nXGBoost tuned ({f1_xgb:.4f}) <= {best_name} ({best_f1:.4f}). Keeping current best.")

    # Guardar mejor modelo
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "classes": sorted(set(y_train)),
        "label_encoder": label_encoder,
    }
    model_path = output_path / "best_classifier.joblib"
    joblib.dump(artifact, model_path)

    # Guardar comparación
    comparison_df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    comparison_path = output_path / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # Reporte detallado del mejor modelo
    y_pred_best = best_model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    y_pred_labels = label_encoder.inverse_transform(y_pred_best)
    report_path = output_path / "classification_report.csv"
    report_df = pd.DataFrame(
        classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)
    ).transpose()
    report_df.to_csv(report_path)

    print("\n" + "=" * 60)
    print("MEJORES RESULTADOS")
    print("=" * 60)
    print(f"Mejor modelo: {best_name}")
    print(f"F1 (weighted): {best_f1:.4f}")
    print(f"\nModelo guardado en: {model_path}")
    print(f"Comparación en: {comparison_path}")
    print(f"Reporte en: {report_path}")

    return {
        "best_model": best_name,
        "best_f1": best_f1,
        "model_path": model_path,
        "comparison": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML classifiers on CLIP embeddings.")
    parser.add_argument("--input_dir", default=str(DEFAULT_OUTPUT_DIR / "embeddings"), help="Embeddings directory.")
    parser.add_argument("--output_dir", default=str(DEFAULT_MODEL_DIR), help="Output directory for models.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--tune_xgboost", action="store_true", help="Enable XGBoost hyperparameter tuning with RandomizedSearchCV.")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of parameter combinations to try (for tuning).")
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds (for tuning).")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for XGBoost training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_classifier(
        args.input_dir,
        args.output_dir,
        args.test_size,
        args.random_state,
        args.tune_xgboost,
        args.n_iter,
        args.cv,
        args.use_gpu,
    )


if __name__ == "__main__":
    main()
