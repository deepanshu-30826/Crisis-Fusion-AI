"""Crisis report semantic clustering and pairwise F1 benchmarking."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    metric: str = "euclidean"
) -> np.ndarray:
    """Cluster normalized embeddings using HDBSCAN.

    Uses sklearn.cluster.HDBSCAN or standalone hdbscan.
    Returns:
        1D array of cluster labels aligned to embeddings (-1 for noise).
    """
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
        labels = clusterer.fit_predict(embeddings)
        return np.asarray(labels, dtype=int)
    except (ImportError, AttributeError):
        pass

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
        labels = clusterer.fit_predict(embeddings)
        return np.asarray(labels, dtype=int)
    except Exception as e:
        print(f"[cluster_hdbscan] HDBSCAN not available ({e}). Falling back to Agglomerative.")
        return cluster_agglomerative(embeddings, distance_threshold=0.35, metric="cosine")


def cluster_agglomerative(
    embeddings: np.ndarray,
    distance_threshold: float = 0.35,
    metric: str = "cosine",
    linkage: str = "average"
) -> np.ndarray:
    """Cluster embeddings using Agglomerative Clustering with distance threshold.

    Returns:
        1D array of cluster labels aligned to embeddings.
    """
    if len(embeddings) == 0:
        return np.array([], dtype=int)

    # If only 1 sample, AgglomerativeClustering cannot fit; return label 0
    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric=metric,
        linkage=linkage
    )
    labels = clusterer.fit_predict(embeddings)
    return np.asarray(labels, dtype=int)


def pairwise_clustering_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute pairwise F1 between true and predicted clustering assignments.

    For all pairs of items (i, j):
    - Same-cluster pair if labels match and are non-noise (label >= 0)
    - Precision = correctly_grouped_pairs / all_predicted_same_pairs
    - Recall = correctly_grouped_pairs / all_true_same_pairs
    - F1 = harmonic mean

    Handles zero-division edge cases cleanly by returning 0.0.
    """
    y_true = np.asarray(true_labels)
    y_pred = np.asarray(pred_labels)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: true ({len(y_true)}) != pred ({len(y_pred)})")

    n = len(y_true)
    if n < 2:
        return 0.0

    # Efficient O(N) calculation via contingency counts:
    # Build mapping of (pred, true) counts
    # Noise labels (< 0) are treated as singletons (no pairs form between two noise points)
    pred_counts: Dict[Any, int] = {}
    true_counts: Dict[Any, int] = {}
    joint_counts: Dict[tuple, int] = {}

    for i in range(n):
        p = y_pred[i]
        t = y_true[i]

        # Only group if not noise (convention: label == -1 denotes unclustered/noise)
        if p != -1:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        if t != -1:
            true_counts[t] = true_counts.get(t, 0) + 1
        if p != -1 and t != -1:
            pair_key = (p, t)
            joint_counts[pair_key] = joint_counts.get(pair_key, 0) + 1

    # Number of predicted same-cluster pairs: sum(n_c * (n_c - 1) / 2)
    pred_pairs = sum((c * (c - 1)) // 2 for c in pred_counts.values())
    true_pairs = sum((c * (c - 1)) // 2 for c in true_counts.values())
    correct_pairs = sum((c * (c - 1)) // 2 for c in joint_counts.values())

    if pred_pairs == 0 or true_pairs == 0 or correct_pairs == 0:
        return 0.0

    precision = correct_pairs / pred_pairs
    recall = correct_pairs / true_pairs

    if precision + recall == 0:
        return 0.0

    f1 = 2.0 * (precision * recall) / (precision + recall)
    return float(f1)


def compare_clustering_methods(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    param_grid_hdbscan: Optional[List[Dict[str, Any]]] = None,
    param_grid_agglomerative: Optional[List[Dict[str, Any]]] = None
) -> pd.DataFrame:
    """Compare HDBSCAN and Agglomerative clustering over hyperparameter grids.

    Returns:
        DataFrame of (method, params, f1) sorted by f1 descending.
    """
    if param_grid_hdbscan is None:
        param_grid_hdbscan = [
            {"min_cluster_size": 2, "metric": "euclidean"},
            {"min_cluster_size": 3, "metric": "euclidean"},
            {"min_cluster_size": 5, "metric": "euclidean"}
        ]

    if param_grid_agglomerative is None:
        param_grid_agglomerative = [
            {"distance_threshold": 0.25, "metric": "cosine", "linkage": "average"},
            {"distance_threshold": 0.35, "metric": "cosine", "linkage": "average"},
            {"distance_threshold": 0.45, "metric": "cosine", "linkage": "average"},
            {"distance_threshold": 0.35, "metric": "cosine", "linkage": "complete"}
        ]

    results: List[Dict[str, Any]] = []

    # Evaluate HDBSCAN grid
    for params in param_grid_hdbscan:
        try:
            preds = cluster_hdbscan(embeddings, **params)
            f1 = pairwise_clustering_f1(true_labels, preds)
            results.append({
                "method": "hdbscan",
                "params": str(params),
                "f1": round(f1, 4),
                "num_clusters": len(set(preds) - {-1})
            })
        except Exception as e:
            print(f"[compare_clustering_methods] HDBSCAN error with {params}: {e}")

    # Evaluate Agglomerative grid
    for params in param_grid_agglomerative:
        try:
            preds = cluster_agglomerative(embeddings, **params)
            f1 = pairwise_clustering_f1(true_labels, preds)
            results.append({
                "method": "agglomerative",
                "params": str(params),
                "f1": round(f1, 4),
                "num_clusters": len(set(preds) - {-1})
            })
        except Exception as e:
            print(f"[compare_clustering_methods] Agglomerative error with {params}: {e}")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by="f1", ascending=False).reset_index(drop=True)
    return df_results


if __name__ == "__main__":
    # Sanity check with synthetic embeddings
    np.random.seed(42)
    # 3 clusters of 4 items each
    true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    test_embs = np.random.randn(12, 16)
    # add cluster shifts
    test_embs[:4] += 5.0
    test_embs[4:8] += -5.0
    test_embs[8:] += 10.0
    # normalize
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)

    agg_preds = cluster_agglomerative(test_embs, distance_threshold=0.5)
    f1 = pairwise_clustering_f1(true_labels, agg_preds)
    print("Agglomerative Labels:", agg_preds)
    print("Pairwise Clustering F1:", f1)

    # Edge cases
    assert pairwise_clustering_f1(np.array([0, 1]), np.array([0, 1])) == 0.0  # no pairs in true or pred
    assert pairwise_clustering_f1(np.array([0, 0]), np.array([0, 0])) == 1.0  # perfect 1 pair
    print("Pairwise F1 edge tests passed!")
