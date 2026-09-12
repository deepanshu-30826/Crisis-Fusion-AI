import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data_loader import load_config
from src.retrieval import build_faiss_index, save_index


def run_build_index(config_path: str = "config.yaml"):
    config = load_config(config_path)
    paths = config.get("paths", {})

    embs_path = paths.get("embeddings_path", "data/embeddings/embeddings.npy")
    index_path = paths.get("faiss_index_path", "models/faiss.index")

    print(f"[run_build_index] Loading embeddings from '{embs_path}'...")
    embeddings = np.load(embs_path)

    print(f"[run_build_index] Building index over {len(embeddings)} vectors...")
    index = build_faiss_index(embeddings)

    save_index(index, index_path)
    print("=== Index Construction Completed Successfully! ===")


if __name__ == "__main__":
    run_build_index()
