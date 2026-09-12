"""Information category classification models for crisis reports."""

from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def _format_labels(labels: Union[List[Any], np.ndarray, pd.Series]) -> np.ndarray:
    """Ensure labels are clean 1D array, extracting first label if list-like."""
    formatted = []
    has_multilabel = False

    for item in labels:
        if isinstance(item, (list, tuple, set)):
            has_multilabel = True
            if len(item) > 0:
                formatted.append(str(item[0]).strip())
            else:
                formatted.append("Unknown")
        elif pd.isna(item):
            formatted.append("Unknown")
        else:
            str_item = str(item).strip()
            # If string is stringified list e.g. "['Flood', 'Rescue']"
            if str_item.startswith("[") and str_item.endswith("]"):
                has_multilabel = True
                cleaned_items = [x.strip(" '\"") for x in str_item[1:-1].split(",") if x.strip(" '\"")]
                formatted.append(cleaned_items[0] if cleaned_items else "Unknown")
            else:
                formatted.append(str_item)

    if has_multilabel:
        print("[train_category_classifier] Warning: Multi-label structure detected. "
              "Defaulting to single-label classification using the primary (first) label per report.")

    return np.array(formatted)


def train_category_classifier(
    embeddings: np.ndarray,
    labels: Union[List[Any], np.ndarray, pd.Series],
    model_type: str = "logreg"
) -> Pipeline:
    """Train category classification pipeline with StandardScaler + classifier.

    Args:
        embeddings: 2D numpy array of text embeddings.
        labels: 1D array or list of category labels.
        model_type: 'logreg' (LogisticRegression) or 'svm' (LinearSVC).

    Returns:
        Fitted sklearn Pipeline.
    """
    y = _format_labels(labels)
    X = np.asarray(embeddings, dtype=np.float32)

    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    elif model_type == "svm":
        clf = LinearSVC(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'logreg' or 'svm'.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

    print(f"[train_category_classifier] Training {model_type.upper()} on {len(X)} samples across {len(set(y))} classes...")
    pipeline.fit(X, y)
    return pipeline


def evaluate_classifier(model: Pipeline, X_test: np.ndarray, y_test: Any) -> Dict[str, Any]:
    """Evaluate classifier performance on test split.

    Returns:
        Dict with 'macro_f1' and full 'classification_report' string.
    """
    y_true = _format_labels(y_test)
    X = np.asarray(X_test, dtype=np.float32)
    y_pred = model.predict(X)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "macro_f1": float(macro_f1),
        "classification_report": report
    }


def save_model(model: Any, path: str) -> None:
    """Save fitted model pipeline to disk using joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    print(f"[save_model] Model saved to '{path}'")


def load_model(path: str) -> Any:
    """Load model pipeline from disk using joblib."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(p)


def predict_category(model: Any, embedding: np.ndarray) -> str:
    """Predict category for a single embedding vector."""
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    preds = model.predict(emb)
    return str(preds[0])


if __name__ == "__main__":
    # Test classifier
    np.random.seed(42)
    X = np.random.randn(20, 16).astype(np.float32)
    y = ["Rescue", "Medical", "Infrastructure", "Rescue"] * 5

    model = train_category_classifier(X, y, model_type="logreg")
    eval_res = evaluate_classifier(model, X, y)
    print(f"Sample Macro-F1: {eval_res['macro_f1']:.4f}")
    pred = predict_category(model, X[0])
    print("Predicted category for first sample:", pred)
