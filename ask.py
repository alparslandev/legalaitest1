"""Terminalden hukuki soru-cevap.

Kullanim:
    python ask.py "soru metni"     # tek soru
    python ask.py                   # etkilesimli mod (cikis: q)
"""
import sys

from rag import config, embeddings, store


def answer_question(question: str) -> None:
    if not store.exists():
        print("Once indeksleme yap:  python ingest.py")
        return

    qvec = embeddings.embed([question])[0]
    hits = store.search(qvec, config.TOP_K)
    relevant = [h for h in hits if h["score"] >= config.MIN_SIMILARITY]

    from rag import llm  # gec import: agir model yuklenmeden once anahtar/akis kontrolu
    response = llm.answer(question, relevant)

    print("\n" + response + "\n")
    if relevant:
        print("-- Kullanilan kaynaklar --")
        seen: set[str] = set()
        for h in relevant:
            if h["source"] not in seen:
                print(f"  - {h['source']} (skor: {h['score']:.2f})")
                seen.add(h["source"])


def main() -> None:
    if len(sys.argv) > 1:
        answer_question(" ".join(sys.argv[1:]))
        return

    print("Hukuki belge asistani. Soru yaz, cikmak icin 'q'.")
    while True:
        try:
            q = input("\nSoru> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit", ""}:
            break
        answer_question(q)


if __name__ == "__main__":
    main()
