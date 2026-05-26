"""Belgeleri indeksle: documents/ -> parcala -> embed -> storage/

Kullanim:
    python ingest.py
"""
from rag import config, documents, embeddings, store


def main() -> None:
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    items = list(documents.iter_document_chunks(config.DOCUMENTS_DIR))
    if not items:
        print(f"'{config.DOCUMENTS_DIR}' icinde belge yok.")
        print("Desteklenen turler: .txt, .md, .docx -- dosyalari oraya koyup tekrar calistir.")
        return

    print(f"{len(items)} parca bulundu. Embedding modeli yukleniyor "
          f"(ilk calistirmada ~2GB model indirilir)...")
    texts = [it["text"] for it in items]
    vectors = embeddings.embed(texts)
    store.save(vectors, items)

    n_docs = len({it["source"] for it in items})
    print(f"Tamam: {n_docs} belge, {len(items)} parca indekslendi -> {config.STORAGE_DIR}")


if __name__ == "__main__":
    main()
