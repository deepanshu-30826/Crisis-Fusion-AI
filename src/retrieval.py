"""Vector similarity search and evidence tweet retrieval using FAISS."""

from pathlib import Path
from typing import Any, List, Tuple, Union
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FallbackMatrixIndex:
    """Pure NumPy cosine / inner-product index when faiss-cpu is unavailable."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)

    def add(self, embeddings: np.ndarray) -> None:
        embs = np.asarray(embeddings, dtype=np.float32)
        if len(self.vectors) == 0:
            self.vectors = embs
        else:
            self.vectors = np.vstack([self.vectors, embs])

    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_vectors, dtype=np.float32)
        if len(self.vectors) == 0:
            return np.empty((len(q), 0), dtype=np.float32), np.empty((len(q), 0), dtype=int)

        # Inner-product matrix multiplication (equivalent to cosine since pre-normalized)
        sims = np.dot(q, self.vectors.T)
        k_actual = min(k, self.vectors.shape[0])
        # Get top-k indices sorted descending
        top_indices = np.argsort(-sims, axis=1)[:, :k_actual]
        top_scores = np.take_along_axis(sims, top_indices, axis=1)
        return top_scores, top_indices


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build an inner-product index for cosine similarity search.

    Uses faiss.IndexFlatIP if available, else FallbackMatrixIndex.
    """
    embs = np.asarray(embeddings, dtype=np.float32)
    if embs.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape {embs.shape}")

    dim = embs.shape[1]

    if HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        print(f"[build_faiss_index] Built FAISS IndexFlatIP with {index.ntotal} vectors (dim={dim}).")
        return index
    else:
        print("[build_faiss_index] FAISS not found; using high-performance NumPy Inner-Product Index.")
        index = FallbackMatrixIndex(dim)
        index.add(embs)
        return index


def save_index(index: Any, path: str) -> None:
    """Save the index to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if HAS_FAISS and isinstance(index, faiss.Index):
        faiss.write_index(index, str(p))
    else:
        import joblib
        joblib.dump(index, str(p))
    print(f"[save_index] Index saved to '{path}'")


def load_index(path: str) -> Any:
    """Load index from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Index file not found: {path}")

    if HAS_FAISS:
        try:
            return faiss.read_index(str(p))
        except Exception:
            pass

    import joblib
    return joblib.load(str(p))


def retrieve_top_k(index: Any, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve top-k nearest indices and similarity scores for a query embedding.

    Returns:
        Tuple of (indices_1d, scores_1d)
    """
    q = np.asarray(query_embedding, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    # Search returns (scores, indices)
    scores, indices = index.search(q, k)
    return indices[0], scores[0]


def get_evidence_ids(indices: Union[List[int], np.ndarray], id_lookup_array: Union[List[str], np.ndarray]) -> List[str]:
    """Map FAISS result indices back to exact original tweet/report IDs.

    Preserves exact ID strings without truncation or modifications.
    """
    evidence_ids: List[str] = []
    lookup = list(id_lookup_array)

    for idx in indices:
        if 0 <= idx < len(lookup):
            evidence_ids.append(str(lookup[idx]))

    return evidence_ids


if __name__ == "__main__":
    np.random.seed(42)
    test_embs = np.random.randn(10, 16).astype(np.float32)
    test_embs = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)
    tweet_ids = [f"tweet_{i:04d}" for i in range(10)]

    idx = build_faiss_index(test_embs)
    matched_indices, scores = retrieve_top_k(idx, test_embs[3], k=3)
    matched_evs = get_evidence_ids(matched_indices, tweet_ids)

    print("Top matched indices:", matched_indices)
    print("Top matched evidence IDs:", matched_evs)
    print("Scores:", scores)
    assert matched_evs[0] == "tweet_0003", "Top match should be exact query vector itself"
    print("Retrieval test passed!")
