import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_config
from src.embeddings import load_embedding_model
from src.retrieval import load_index
from src.inference import predict_batch
from src.evaluation import run_full_evaluation


def run_prediction(
    input_path: str = "data/processed/filtered_reports.csv",
    config_path: str = "config.yaml"
):
    config = load_config(config_path)
    paths = config.get("paths", {})
    models_cfg = config.get("models", {})
    retrieval_cfg = config.get("retrieval", {})

    print("=== Step 1: Loading Pipeline Artifacts ===")
    emb_model = load_embedding_model(models_cfg.get("embedding_model_name"))
    cat_model = joblib.load(paths.get("category_model_path", "models/category_model.joblib"))
    prio_model = joblib.load(paths.get("priority_model_path", "models/priority_model.joblib"))
    faiss_index = load_index(paths.get("faiss_index_path", "models/faiss.index"))
    id_lookup = np.load(paths.get("id_lookup_path", "data/embeddings/id_lookup.npy"), allow_pickle=True)

    with open(paths.get("cluster_profiles_path", "models/cluster_profiles.json"), "r", encoding="utf-8") as f:
        cluster_profiles = json.load(f)

    print(f"=== Step 2: Running Batch Predictions on '{input_path}' ===")
    df = pd.read_csv(input_path)
    reports = df.to_dict(orient="records")

    predictions = predict_batch(
        reports=reports,
        cluster_profiles=cluster_profiles,
        category_model=cat_model,
        priority_model=prio_model,
        faiss_index=faiss_index,
        id_lookup=id_lookup,
        embedding_model=emb_model,
        output_path=paths.get("predictions_path", "outputs/predictions.jsonl"),
        similarity_threshold=float(retrieval_cfg.get("similarity_threshold", 0.65)),
        top_k=int(retrieval_cfg.get("top_k", 3))
    )

    print(f"Sample prediction (Item 0):\n{json.dumps(predictions[0], indent=2)}")

    print("=== Step 3: Running Comprehensive 100-Point Evaluation ===")
    eval_results = run_full_evaluation(
        predictions_path=paths.get("predictions_path", "outputs/predictions.jsonl"),
        ground_truth_path=input_path
    )

    eval_out = paths.get("evaluation_report_path", "outputs/evaluation_report.json")
    Path(eval_out).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to '{eval_out}'.")


if __name__ == "__main__":
    import sys
    in_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed/filtered_reports.csv"
    run_prediction(input_path=in_file)
