import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loader import build_filtered_dataset, load_config
from src.preprocessing import preprocess_dataframe
from src.variant_generator import generate_variants
from src.embeddings import embed_dataframe


def run_preparation(config_path: str = "config.yaml"):
    config = load_config(config_path)
    paths = config.get("paths", {})
    models_cfg = config.get("models", {})

    print("=== Step 1: Ingesting Raw Data & Applying Label Filter ===")
    df_filtered = build_filtered_dataset(
        tweets_glob=paths.get("raw_data_glob", "data/raw/*.json.gz"),
        labels_path=paths.get("raw_labels_path", "data/raw/labels.json"),
        output_path=paths.get("processed_reports_path", "data/processed/filtered_reports.csv"),
        config_path=config_path
    )

    if df_filtered.empty:
        print("Warning: Filtered dataset is empty. Ensure raw data exists.")
        return

    print("=== Step 2: Preprocessing Text (Meaning-Preserving) ===")
    df_clean = preprocess_dataframe(df_filtered, text_col="text")
    df_clean.to_csv(paths.get("processed_reports_path"), index=False)
    print(f"Updated '{paths.get('processed_reports_path')}' with 'clean_text' column.")

    print("=== Step 3: Generating Test Variants for Robustness Testing ===")
    df_variants = generate_variants(df_clean, id_col="tweet_id", text_col="clean_text")
    var_path = paths.get("processed_variants_path", "data/processed/variants.csv")
    Path(var_path).parent.mkdir(parents=True, exist_ok=True)
    df_variants.to_csv(var_path, index=False)
    print(f"Generated {len(df_variants)} variants saved to '{var_path}'.")

    print("=== Step 4: Generating Dense L2-Normalized Embeddings ===")
    emb_model_name = models_cfg.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    emb_path = paths.get("embeddings_path", "data/embeddings/embeddings.npy")
    df_clean, embeddings = embed_dataframe(
        df_clean,
        text_col="clean_text",
        model_name=emb_model_name,
        output_path=emb_path
    )

    # Save ID lookup array aligned with embeddings
    id_lookup_path = paths.get("id_lookup_path", "data/embeddings/id_lookup.npy")
    tweet_ids = np.array(df_clean["tweet_id"].astype(str).tolist(), dtype=object)
    np.save(id_lookup_path, tweet_ids)
    print(f"Saved {len(tweet_ids)} ID lookups to '{id_lookup_path}'.")

    print("=== Data Preparation Completed Successfully! ===")


if __name__ == "__main__":
    run_preparation()
