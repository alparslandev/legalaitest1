"""Yerel embedding modeli sarmalayicisi (arama icin metni vektore cevirir)."""
import os

# Model zaten yerelde onbellekte; HF Hub'a baglanmadan onbellekten yukle.
# Bu, "unauthenticated requests to HF Hub" uyarisini susturur ve yuklemeyi hizlandirir.
# YENI bir EMBEDDING_MODEL indirmen gerekirse: terminalde "HF_HUB_OFFLINE=0 python ingest.py"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Ilk cagrida model indirilir (~2GB), sonraki cagrilarda onbellekten gelir.
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    vecs = _model().encode(
        texts,
        normalize_embeddings=True,           # cosine benzerligi icin
        show_progress_bar=len(texts) > 8,
        batch_size=16,
    )
    return np.asarray(vecs, dtype=np.float32)
