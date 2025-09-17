from __future__ import annotations
"""
EmbeddingFeaturizer: sklearn-compatible transformer that yields sentence embeddings.
Stores only the model name for serialization; loads the model lazily at transform time.
"""
from typing import Optional, List
import numpy as np


_EMBEDDERS = {}


def _get_embedder(name: str):
    from sentence_transformers import SentenceTransformer
    global _EMBEDDERS
    if name not in _EMBEDDERS:
        _EMBEDDERS[name] = SentenceTransformer(name)
    return _EMBEDDERS[name]


class EmbeddingFeaturizer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize

    def fit(self, X: List[str], y=None):
        return self

    def transform(self, X: List[str]):
        emb = _get_embedder(self.model_name).encode(list(X), normalize_embeddings=self.normalize)
        return np.asarray(emb, dtype=np.float32)

    # Help joblib/pickle avoid serializing heavy models
    def __getstate__(self):
        return {"model_name": self.model_name, "normalize": self.normalize}

    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self.normalize = state.get("normalize", True)

