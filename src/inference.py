"""End-to-end inference pipeline for crisis report fusion and triage."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.preprocessing import clean_text
from src.embeddings import embed_texts
from src.classifier import predict_category
from src.priority import predict_priority_score
from src.retrieval import retrieve_top_k, get_evidence_ids


def assign_cluster(
    embedding: np.ndarray,
    cluster_profiles: Dict[str, Any],
    similarity_threshold: float = 0.65
) -> str:
    """Find the nearest cluster centroid, assigning to existing or creating a new singleton cluster."""
    # cluster_profiles can be either a dict of {cluster_id: centroid_vector}
    # or a wrapper dict {"centroids": {...}, "threshold": ...}
    centroids_dict = cluster_profiles.get("centroids", cluster_profiles) if isinstance(cluster_profiles, dict) else {}

    best_cid: Optional[str] = None
    best_sim = -1.0

    for cid, centroid in centroids_dict.items():
        if cid == "threshold" or cid == "counter":
            continue
        c_vec = np.asarray(centroid, dtype=np.float32)
        sim = float(np.dot(embedding, c_vec))
        if sim > best_sim:
            best_sim = sim
            best_cid = str(cid)

    if best_cid is not None and best_sim >= similarity_threshold:
        return best_cid
    else:
        # Create a new singleton cluster ID
        new_id_num = len(centroids_dict) + 1
        new_cid = f"C_{new_id_num:03d}"
        centroids_dict[new_cid] = embedding.copy()
        if "centroids" in cluster_profiles:
            cluster_profiles["centroids"] = centroids_dict
        return new_cid


def predict_report(
    report_text: str,
    cluster_profiles: Dict[str, Any],
    category_model: Any,
    priority_model: Any,
    faiss_index: Any,
    id_lookup: Union[List[str], np.ndarray],
    embedding_model: Any,
    similarity_threshold: float = 0.65,
    top_k: int = 3
) -> Dict[str, Any]:
    """Execute end-to-end inference for a single crisis report.

    Pipeline:
    1. Preprocess text (meaning-preserving normalization)
    2. Compute dense L2-normalized embedding
    3. Assign to nearest cluster centroid or create new singleton cluster
    4. Predict actionable information category
    5. Predict continuous priority urgency score
    6. Retrieve top-k nearest matching evidence tweet IDs

    Output schema exactly matches:
    {
      "cluster_id": "C_017",
      "information_category": "Affected Population",
      "priority_score": 0.91,
      "evidence_ids": ["tweet_182", "tweet_311", "tweet_492"]
    }
    """
    # 1. Preprocess
    cleaned = clean_text(report_text)

    # 2. Embed
    emb = embed_texts([cleaned], embedding_model, normalize=True)[0]

    # 3. Cluster assignment
    cid = assign_cluster(emb, cluster_profiles, similarity_threshold=similarity_threshold)

    # 4. Predict Category
    category = predict_category(category_model, emb)

    # 5. Predict Priority
    score = predict_priority_score(priority_model, emb)
    # Ensure score is formatted nicely as float [0.0, 1.0]
    score_clipped = float(np.clip(score, 0.0, 1.0))
    priority_val = round(score_clipped, 2)

    # 6. Retrieve Top-K Evidence IDs
    indices, _ = retrieve_top_k(faiss_index, emb, k=top_k)
    evidence_ids = get_evidence_ids(indices, id_lookup)

    return {
        "cluster_id": str(cid),
        "information_category": str(category),
        "priority_score": priority_val,
        "evidence_ids": [str(e) for e in evidence_ids]
    }


def predict_batch(
    reports: List[Union[Dict[str, Any], str]],
    cluster_profiles: Dict[str, Any],
    category_model: Any,
    priority_model: Any,
    faiss_index: Any,
    id_lookup: Union[List[str], np.ndarray],
    embedding_model: Any,
    output_path: str = "outputs/predictions.jsonl",
    similarity_threshold: float = 0.65,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Run predict_report over multiple reports and save line-delimited JSONL."""
    results: List[Dict[str, Any]] = []

    for item in reports:
        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("full_text")
                or item.get("clean_text")
                or item.get("report_text")
                or ""
            )
        else:
            text = str(item)

        pred = predict_report(
            report_text=text,
            cluster_profiles=cluster_profiles,
            category_model=category_model,
            priority_model=priority_model,
            faiss_index=faiss_index,
            id_lookup=id_lookup,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            top_k=top_k
        )
        results.append(pred)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[predict_batch] Wrote {len(results)} predictions to '{output_path}'")
    return results
