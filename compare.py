"""Ayni soruyu TUM modellere sorup cevaplari alt alta gosterir.

Kullanim:
    python compare.py                 # senden soru ister, 'q' diyene kadar calisir
    python compare.py "soru metni"    # tek soru sorup cikar

Not: 16GB RAM'de iki yerel model ayni anda belege sigmaz. Bu yuzden yerel
modeller SIRAYLA calisir ve her biri cevaptan sonra bellekten birakilir
(keep_alive=0). Haiku bulutta oldugu icin RAM kullanmaz.
Kiyaslanacak modelleri asagidaki MODELS listesinden degistirebilirsin.
"""
import sys

from rag import config, embeddings, store, llm

# (gosterim adi, backend, model)
MODELS = [
    ("Claude Haiku   (bulut)", "anthropic", config.LLM_MODEL),
    ("gemma4:e4b     (yerel)", "ollama", "gemma4:e4b"),
    ("deepseek-r1:8b (yerel)", "ollama", "deepseek-r1:8b"),
]


def compare(question: str) -> None:
    if not store.exists():
        print("Once indeksleme yap:  python ingest.py")
        return

    qvec = embeddings.embed([question])[0]
    hits = store.search(qvec, config.TOP_K)
    relevant = [h for h in hits if h["score"] >= config.MIN_SIMILARITY]

    print("\n" + "=" * 72)
    print("SORU:", question)
    if relevant:
        print("Getirilen kaynaklar:", ", ".join(sorted({h["source"] for h in relevant})))
    else:
        print("Getirilen kaynak yok (esik alti).")
    print("=" * 72)

    # Tum modeller AYNI parcalari gorur -> adil kiyas.
    for adi, backend, model in MODELS:
        print(f"\n### {adi}")
        sys.stdout.flush()
        try:
            ka = 0 if backend == "ollama" else None  # yerel modeli cevaptan sonra birak
            cevap = llm.answer(question, relevant, backend=backend, model=model, keep_alive=ka)
        except SystemExit as e:
            cevap = f"[atlandi] {e}"
        except Exception as e:  # noqa: BLE001 - kiyasta bir model patlasa digerleri devam etsin
            cevap = f"[hata] {e}"
        print(cevap)
    print()


def main() -> None:
    if len(sys.argv) > 1:
        compare(" ".join(sys.argv[1:]))
        return

    print("Tum modelleri kiyasla. Soru yaz, cikmak icin 'q'.")
    while True:
        try:
            q = input("\nSoru> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit", ""}:
            break
        compare(q)


if __name__ == "__main__":
    main()
