"""Evaluation suite for Crisis Report Fusion and Priority Ranking."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.clustering import pairwise_clustering_f1
from src.classifier import _format_labels
from src.priority import _convert_priority_labels, evaluate_priority_ndcg


def evidence_id_completeness(
    predicted_evidence_ids: List[List[str]],
    true_evidence_ids: List[List[str]]
) -> float:
    """Compute recall-style evidence ID completeness.

    Fraction of true evidence IDs that appear in the predicted list,
    averaged across all evaluated items.
    """
    if not predicted_evidence_ids or not true_evidence_ids:
        return 0.0

    recalls: List[float] = []

    for pred_list, true_list in zip(predicted_evidence_ids, true_evidence_ids):
        # Format as string sets
        true_set = {str(x).strip() for x in (true_list if isinstance(true_list, list) else [true_list]) if str(x).strip()}
        pred_set = {str(x).strip() for x in (pred_list if isinstance(pred_list, list) else [pred_list]) if str(x).strip()}

        if not true_set:
            # If no ground truth evidence was required, completeness is neutral
            recalls.append(1.0)
            continue

        matched = len(true_set.intersection(pred_set))
        recalls.append(matched / len(true_set))

    return float(np.mean(recalls)) if recalls else 0.0


def _load_data_records(path: str) -> List[Dict[str, Any]]:
    """Load JSONL, JSON, or CSV records into list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix == ".jsonl":
        records = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else list(data.values())
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def run_full_evaluation(
    predictions_path: str = "outputs/predictions.jsonl",
    ground_truth_path: str = "data/processed/filtered_reports.csv"
) -> Dict[str, Any]:
    """Load predictions and ground truth, compute all four scored metrics, and output summary.

    Challenge weights:
    - Clustering Pairwise F1: 35%
    - Category Macro-F1: 25%
    - Priority NDCG: 20%
    - Evidence Completeness: 15%
    (Runtime/reproducibility: 5% scored manually)

    Returns:
        Dictionary containing metric values, weighted scores, and total.
    """
    print(f"[run_full_evaluation] Loading predictions from: {predictions_path}")
    preds = _load_data_records(predictions_path)

    print(f"[run_full_evaluation] Loading ground truth from: {ground_truth_path}")
    truth = _load_data_records(ground_truth_path)

    n_eval = min(len(preds), len(truth))
    if n_eval == 0:
        print("Warning: No records to evaluate.")
        return {}

    preds = preds[:n_eval]
    truth = truth[:n_eval]

    # 1. Clustering
    pred_clusters = [str(r.get("cluster_id", i)) for i, r in enumerate(preds)]
    
    # Identify true cluster key if present
    true_cluster_key = None
    for candidate in ["cluster_id", "source_id", "event_id"]:
        if any(candidate in r and r[candidate] is not None for r in truth):
            true_cluster_key = candidate
            break

    if true_cluster_key:
        true_clusters = [str(r.get(true_cluster_key)) for r in truth]
    else:
        # If no explicit cluster ID is given in ground truth, fall back to categories as proxy grouping
        print("[run_full_evaluation] Note: No 'cluster_id' or 'source_id' in ground truth; using 'categories' as event proxy.")
        true_clusters = [str(r.get("categories") or r.get("category") or i) for i, r in enumerate(truth)]
    # Map cluster string identifiers to integers for pairwise_clustering_f1
    cluster_map_true = {k: idx for idx, k in enumerate(set(true_clusters))}
    cluster_map_pred = {k: idx for idx, k in enumerate(set(pred_clusters))}
    true_cluster_ints = np.array([cluster_map_true[c] for c in true_clusters])
    pred_cluster_ints = np.array([cluster_map_pred[c] for c in pred_clusters])

    clustering_f1 = pairwise_clustering_f1(true_cluster_ints, pred_cluster_ints)

    # 2. Information Category Macro-F1
    pred_cats = [str(r.get("information_category", "")) for r in preds]
    true_raw_cats = [r.get("categories") or r.get("category") or "" for r in truth]
    true_cats = _format_labels(true_raw_cats)
    category_macro_f1 = float(f1_score(true_cats, pred_cats, average="macro", zero_division=0))

    # 3. Priority NDCG
    pred_priorities = [float(r.get("priority_score", 0.0)) for r in preds]
    true_raw_priorities = [r.get("priority") or 0.0 for r in truth]
    true_priorities = _convert_priority_labels(true_raw_priorities)
    priority_ndcg = evaluate_priority_ndcg(true_priorities, pred_priorities)

    # 4. Evidence Completeness
    pred_evs = [r.get("evidence_ids", []) for r in preds]
    true_evs = []
    for r in truth:
        if "evidence_ids" in r and isinstance(r["evidence_ids"], list):
            true_evs.append(r["evidence_ids"])
        elif "tweet_id" in r:
            true_evs.append([str(r["tweet_id"])])
        else:
            true_evs.append([])
    evidence_completeness = evidence_id_completeness(pred_evs, true_evs)

    # Weighted Scoring (out of 95 total automated points)
    w_clustering = 35.0
    w_category = 25.0
    w_priority = 20.0
    w_evidence = 15.0

    pts_clustering = clustering_f1 * w_clustering
    pts_category = category_macro_f1 * w_category
    pts_priority = priority_ndcg * w_priority
    pts_evidence = evidence_completeness * w_evidence
    total_points = pts_clustering + pts_category + pts_priority + pts_evidence
    normalized_100 = (total_points / 95.0) * 100.0

    results = {
        "num_evaluated_reports": n_eval,
        "metrics": {
            "clustering_pairwise_f1": round(clustering_f1, 4),
            "category_macro_f1": round(category_macro_f1, 4),
            "priority_ndcg": round(priority_ndcg, 4),
            "evidence_completeness": round(evidence_completeness, 4)
        },
        "points": {
            "clustering_points": round(pts_clustering, 2),
            "category_points": round(pts_category, 2),
            "priority_points": round(pts_priority, 2),
            "evidence_points": round(pts_evidence, 2),
            "total_score_95": round(total_points, 2),
            "normalized_score_100": round(normalized_100, 2)
        }
    }

    # Print clean benchmark summary table
    print("\n" + "=" * 76)
    print("  CRISIS REPORT FUSION & PRIORITY RANKING — EVALUATION SCORECARD")
    print("=" * 76)
    print(f"  {'Metric':<26} {'Raw Score':<12} {'Max Weight':<12} {'Awarded Points':<14}")
    print("-" * 76)
    print(f"  {'Pairwise Clustering F1':<26} {clustering_f1:<12.4f} {w_clustering:<12.1f} {pts_clustering:<14.2f}")
    print(f"  {'Category Macro-F1':<26} {category_macro_f1:<12.4f} {w_category:<12.1f} {pts_category:<14.2f}")
    print(f"  {'Priority NDCG':<26} {priority_ndcg:<12.4f} {w_priority:<12.1f} {pts_priority:<14.2f}")
    print(f"  {'Evidence Completeness':<26} {evidence_completeness:<12.4f} {w_evidence:<12.1f} {pts_evidence:<14.2f}")
    print("-" * 76)
    print(f"  {'TOTAL AUTOMATED SCORE':<26} {'-':<12} {'95.0':<12} {total_points:<14.2f} / 95.0")
    print(f"  {'NORMALIZED SCORE (100)':<26} {'-':<12} {'100.0':<12} {normalized_100:<14.2f} / 100.0")
    print("=" * 76 + "\n")

    return results


if __name__ == "__main__":
    import sys
    pred_f = sys.argv[1] if len(sys.argv) > 1 else "outputs/predictions.jsonl"
    truth_f = sys.argv[2] if len(sys.argv) > 2 else "data/processed/filtered_reports.csv"
    if Path(pred_f).exists() and Path(truth_f).exists():
        run_full_evaluation(pred_f, truth_f)
    else:
        print(f"Evaluation files not found ({pred_f}, {truth_f}). Run pipeline first.")
