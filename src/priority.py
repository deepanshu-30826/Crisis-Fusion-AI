"""Priority urgency ranking models and continuous scoring for crisis reports."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import ndcg_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Mapping of priority labels to continuous urgency relevance scores [0.0, 1.0]
PRIORITY_LABEL_MAP: Dict[str, float] = {
    "critical": 1.0,
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
    "lowest": 0.0,
    "none": 0.0
}


def _convert_priority_labels(labels: Union[List[Any], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert priority labels (strings or numbers) to continuous numeric values."""
    scores = []
    for item in labels:
        if isinstance(item, (int, float)) and not np.isnan(item):
            scores.append(float(item))
        elif isinstance(item, str):
            clean = item.strip().lower()
            if clean in PRIORITY_LABEL_MAP:
                scores.append(PRIORITY_LABEL_MAP[clean])
            else:
                try:
                    scores.append(float(clean))
                except ValueError:
                    scores.append(0.25)
        else:
            scores.append(0.0)
    return np.array(scores, dtype=np.float32)


def _prepare_features(embeddings: np.ndarray, extra_features: Optional[np.ndarray] = None) -> np.ndarray:
    """Combine dense embeddings with optional extra features."""
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if extra_features is not None:
        extra = np.asarray(extra_features, dtype=np.float32)
        if extra.ndim == 1:
            extra = extra.reshape(1, -1) if len(X) == 1 else extra.reshape(len(X), -1)
        X = np.hstack([X, extra])
    return X


def train_priority_model(
    embeddings: np.ndarray,
    priority_labels: Union[List[Any], np.ndarray, pd.Series],
    model_type: str = "ridge",
    extra_features: Optional[np.ndarray] = None
) -> Pipeline:
    """Train priority ranking model using Ridge regression or ordinal-style Logistic Regression.

    Args:
        embeddings: 2D array of embeddings.
        priority_labels: Priority indicators (continuous or categorical).
        model_type: 'ridge' or 'logreg'.
        extra_features: Optional extra numerical features (e.g. category one-hot).

    Returns:
        Fitted sklearn Pipeline.
    """
    X = _prepare_features(embeddings, extra_features)
    y_continuous = _convert_priority_labels(priority_labels)

    if model_type == "ridge":
        model = Ridge(alpha=1.0)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])
        pipeline.fit(X, y_continuous)
    elif model_type == "logreg":
        # Discretize into ordinal bins for LogisticRegression
        bins = np.array([0.0, 0.33, 0.66, 1.01])
        y_discrete = np.digitize(y_continuous, bins) - 1
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        pipeline.fit(X, y_discrete)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'ridge' or 'logreg'.")

    # Store metadata on pipeline
    pipeline._model_type = model_type
    print(f"[train_priority_model] Trained {model_type.upper()} priority model on {len(X)} reports.")
    return pipeline


def predict_priority_score(
    model: Pipeline,
    embedding: np.ndarray,
    extra_features: Optional[np.ndarray] = None
) -> float:
    """Predict a continuous priority score for ranking.

    Returns:
        float score suitable for ranking via NDCG.
    """
    X = _prepare_features(embedding, extra_features)
    model_type = getattr(model, "_model_type", "ridge")

    if model_type == "ridge":
        score = float(model.predict(X)[0])
    else:
        # Expected value from class probabilities: sum(prob_i * weight_i)
        probs = model.predict_proba(X)[0]
        weights = np.linspace(0.0, 1.0, len(probs))
        score = float(np.dot(probs, weights))

    return round(score, 4)


def evaluate_priority_ndcg(
    true_relevance: Union[List[float], np.ndarray],
    predicted_scores: Union[List[float], np.ndarray],
    k: Optional[int] = None
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG) for ranked items.

    Args:
        true_relevance: 1D array of ground truth urgency values.
        predicted_scores: 1D array of predicted continuous ranking scores.
        k: Optional rank cutoff.

    Returns:
        float NDCG score in [0.0, 1.0].
    """
    y_true = np.asarray(true_relevance, dtype=np.float32)
    y_pred = np.asarray(predicted_scores, dtype=np.float32)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: true ({len(y_true)}) != pred ({len(y_pred)})")

    n = len(y_true)
    if n < 2:
        return 1.0

    # If all true relevance scores are identical, ranking is trivial
    if np.all(y_true == y_true[0]):
        return 1.0

    # Reshape to (1, n_samples) for single-query ranking
    y_true_2d = y_true.reshape(1, -1)
    y_pred_2d = y_pred.reshape(1, -1)

    # Scikit-learn ndcg_score requires non-negative true relevance
    min_val = np.min(y_true_2d)
    if min_val < 0:
        y_true_2d = y_true_2d - min_val

    try:
        score = ndcg_score(y_true_2d, y_pred_2d, k=k)
        return float(round(score, 4))
    except Exception as e:
        print(f"[evaluate_priority_ndcg] Warning: {e}")
        return 0.0


def save_priority_model(model: Pipeline, path: str) -> None:
    """Save fitted priority model to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    print(f"[save_priority_model] Saved priority model to '{path}'")


def load_priority_model(path: str) -> Pipeline:
    """Load priority model from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Priority model not found at '{path}'")
    return joblib.load(p)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(20, 16).astype(np.float32)
    y = ["Critical", "High", "Low", "Medium"] * 5

    m_ridge = train_priority_model(X, y, model_type="ridge")
    preds = [predict_priority_score(m_ridge, X[i]) for i in range(len(X))]
    ndcg = evaluate_priority_ndcg(_convert_priority_labels(y), preds, k=5)
    print("Ridge Priority NDCG@5:", ndcg)
