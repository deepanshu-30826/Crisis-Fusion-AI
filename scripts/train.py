import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from src.data_loader import load_config
from src.clustering import cluster_agglomerative, compare_clustering_methods
from src.classifier import train_category_classifier, evaluate_classifier, save_model
from src.priority import train_priority_model, evaluate_priority_ndcg, save_priority_model, _convert_priority_labels


def run_training(config_path: str = "config.yaml"):
    config = load_config(config_path)
    paths = config.get("paths", {})
    models_cfg = config.get("models", {})
    cluster_cfg = config.get("clustering", {})

    reports_path = paths.get("processed_reports_path", "data/processed/filtered_reports.csv")
    embs_path = paths.get("embeddings_path", "data/embeddings/embeddings.npy")

    print(f"[run_training] Loading data from '{reports_path}' and '{embs_path}'...")
    df = pd.read_csv(reports_path)
    embeddings = np.load(embs_path)

    if len(df) != len(embeddings):
        raise ValueError(f"Length mismatch: {len(df)} reports vs {len(embeddings)} embeddings")

    models_dir = Path(paths.get("cluster_profiles_path", "models/cluster_profiles.json")).parent
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clustering Profiling
    print("=== Step 1: Semantic Clustering & Profile Generation ===")
    dist_thresh = float(cluster_cfg.get("distance_threshold", 0.35))
    metric = cluster_cfg.get("metric", "cosine")
    linkage = cluster_cfg.get("linkage", "average")

    cluster_labels = cluster_agglomerative(
        embeddings,
        distance_threshold=dist_thresh,
        metric=metric,
        linkage=linkage
    )
    df["assigned_cluster"] = cluster_labels
    num_clusters = len(set(cluster_labels) - {-1})
    print(f"Discovered {num_clusters} distinct semantic clusters.")

    # Compute centroids for each cluster
    centroids = {}
    for c_lbl in set(cluster_labels):
        if c_lbl == -1:
            continue
        mask = (cluster_labels == c_lbl)
        cluster_embs = embeddings[mask]
        centroid = np.mean(cluster_embs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        cid = f"C_{c_lbl + 1:03d}"
        centroids[cid] = [float(x) for x in centroid]

    profiles_path = paths.get("cluster_profiles_path", "models/cluster_profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(centroids, f, indent=2)
    print(f"Saved {len(centroids)} cluster centroids to '{profiles_path}'.")

    # 2. Train Category Classifier
    print("=== Step 2: Training Category Classifier ===")
    clf_type = models_cfg.get("classifier_type", "logreg")
    cat_model = train_category_classifier(embeddings, df["categories"], model_type=clf_type)
    cat_eval = evaluate_classifier(cat_model, embeddings, df["categories"])
    print(f"Classifier Training Macro-F1: {cat_eval['macro_f1']:.4f}")

    cat_model_path = paths.get("category_model_path", "models/category_model.joblib")
    save_model(cat_model, cat_model_path)

    # 3. Train Priority Urgency Model
    print("=== Step 3: Training Priority Urgency Model ===")
    prio_type = models_cfg.get("priority_model_type", "ridge")
    priority_model = train_priority_model(embeddings, df["priority"], model_type=prio_type)

    prio_scores = [
        float(priority_model.predict(emb.reshape(1, -1))[0])
        for emb in embeddings
    ]
    ndcg = evaluate_priority_ndcg(_convert_priority_labels(df["priority"]), prio_scores, k=5)
    print(f"Priority Ranking NDCG@5: {ndcg:.4f}")

    prio_model_path = paths.get("priority_model_path", "models/priority_model.joblib")
    save_priority_model(priority_model, prio_model_path)

    print("=== Model Training Completed Successfully! ===")


if __name__ == "__main__":
    run_training()
