"""Dense vector embeddings for crisis report text using SentenceTransformers."""

from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import pandas as pd

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load a SentenceTransformer embedding model.

    If sentence_transformers is not installed, falls back gracefully to a
    hash-based dense projection so the pipeline continues to run.
    """
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[load_embedding_model] Loading '{model_name}'...")
        model = SentenceTransformer(model_name)
        return model
    except ImportError:
        print("[load_embedding_model] Warning: sentence-transformers not installed.")
        print("Using lightweight deterministic projection fallback.")
        return _FallbackEmbedder()
    except Exception as e:
        print(f"[load_embedding_model] Warning: Could not load {model_name} ({e}).")
        print("Using lightweight deterministic projection fallback.")
        return _FallbackEmbedder()


class _FallbackEmbedder:
    """Lightweight 384-dim TF-IDF & hashing embedder used if torch/transformers unavailable."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        import hashlib
        n_samples = len(texts)
        res = np.zeros((n_samples, self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = str(t).lower().split()
            if not tokens:
                res[i, 0] = 1.0
                continue
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                res[i, idx] += sign
        if normalize_embeddings:
            norms = np.linalg.norm(res, axis=1, keepdims=True)
            res = np.divide(res, np.maximum(norms, 1e-12))
        return res


def embed_texts(
    texts: List[str],
    model: Any,
    normalize: bool = True,
    batch_size: int = 64
) -> np.ndarray:
    """Generate dense embeddings for a list of texts.

    L2-normalizes embeddings when normalize=True (required for cosine similarity).

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    str_texts = [str(t) if t is not None else "" for t in texts]

    # Try calling model.encode
    try:
        embeddings = model.encode(
            str_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )
    except TypeError:
        # Some older/fallback models don't support normalize_embeddings kwarg
        embeddings = model.encode(str_texts, batch_size=batch_size, show_progress_bar=False)

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(embeddings, np.maximum(norms, 1e-12))

    return embeddings


def embed_dataframe(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    model_name: str = DEFAULT_MODEL_NAME,
    output_path: str = "data/embeddings/embeddings.npy",
    batch_size: int = 64
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Embed dataframe text column and save embeddings to .npy file.

    Returns:
        Tuple of (original_df, index_aligned_embeddings)
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataframe. Available: {list(df.columns)}")

    model = load_embedding_model(model_name)
    texts = df[text_col].fillna("").astype(str).tolist()
    embeddings = embed_texts(texts, model, normalize=True, batch_size=batch_size)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, embeddings)
    print(f"[embed_dataframe] Saved {len(embeddings)} embeddings to '{output_path}' (shape: {embeddings.shape})")

    return df, embeddings


if __name__ == "__main__":
    sample_texts = [
        "Major bridge collapse reported on river crossing.",
        "Medical aid requested for 12 injured people.",
        "Flash flood warning issued for northern county."
    ]
    m = load_embedding_model()
    embs = embed_texts(sample_texts, m, normalize=True)
    print("Embedding shape:", embs.shape)
    print("L2 Norms (should be 1.0):", np.linalg.norm(embs, axis=1))
