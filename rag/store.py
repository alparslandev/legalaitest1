"""Basit yerel vektor deposu: numpy matrisi + JSON ustveri.
Kucuk/orta olcek icin yeterli ve sağlam. 10M belgeye cikinca
buradaki search/save fonksiyonlari Qdrant/pgvector'a tasinir."""
import json

import numpy as np

from . import config

_VECTORS_PATH = config.STORAGE_DIR / "vectors.npy"
_META_PATH = config.STORAGE_DIR / "chunks.json"


def save(vectors: np.ndarray, metadata: list[dict]) -> None:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_VECTORS_PATH, vectors)
    _META_PATH.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def exists() -> bool:
    return _VECTORS_PATH.exists() and _META_PATH.exists()


def load() -> tuple[np.ndarray, list[dict]]:
    vectors = np.load(_VECTORS_PATH)
    metadata = json.loads(_META_PATH.read_text(encoding="utf-8"))
    return vectors, metadata


def search(query_vec: np.ndarray, top_k: int) -> list[dict]:
    vectors, metadata = load()
    # vektorler normalize edildigi icin ic carpim = cosine benzerligi
    scores = vectors @ query_vec
    order = np.argsort(-scores)[:top_k]
    results: list[dict] = []
    for idx in order:
        item = dict(metadata[int(idx)])
        item["score"] = float(scores[int(idx)])
        results.append(item)
    return results
