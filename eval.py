"""Sadece Haiku ile dogruluk testi: tests/sorular.txt icindeki sorulari calistirir,
beklenen cevapla kiyaslar ve PASS/FAIL ozeti verir.

Kullanim:
    python eval.py

Boylece "Haiku dogru cevap veriyor mu?" sorusunu goz karari degil, tekrarlanabilir
bir test olarak takip edersin. FAIL ciktiginda 'kaynaklar' satirina bak:
  - dogru belge getirilmis ama cevap yanlissa  -> model/prompt sorunu
  - dogru belge hic getirilmemisse              -> retrieval sorunu (TOP_K / chunking)
"""
from pathlib import Path

from rag import config, embeddings, store, llm

TESTS = Path(__file__).resolve().parent / "tests" / "sorular.txt"


def _load_cases() -> list[tuple[str, str | None]]:
    cases: list[tuple[str, str | None]] = []
    for raw in TESTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            q, beklenen = line.split("|", 1)
            cases.append((q.strip(), beklenen.strip()))
        else:
            cases.append((line, None))
    return cases


def _ask_haiku(question: str) -> tuple[str, list[str]]:
    qv = embeddings.embed([question])[0]
    hits = store.search(qv, config.TOP_K)
    relevant = [h for h in hits if h["score"] >= config.MIN_SIMILARITY]
    kaynaklar = sorted({h["source"] for h in relevant})
    cevap = llm.answer(question, relevant, backend="anthropic", model=config.LLM_MODEL)
    return cevap, kaynaklar


def main() -> None:
    if not store.exists():
        print("Once indeksleme yap:  python ingest.py")
        return

    cases = _load_cases()
    gecti = kaldi = kontrolsuz = 0

    for q, beklenen in cases:
        cevap, kaynaklar = _ask_haiku(q)
        if beklenen is None:
            durum = "—"
            kontrolsuz += 1
        elif beklenen.upper() == "BULAMADI":
            ok = "bulamad" in cevap.lower()
            durum = "PASS" if ok else "FAIL"
            gecti += ok
            kaldi += not ok
        else:
            ok = beklenen.lower() in cevap.lower()
            durum = "PASS" if ok else "FAIL"
            gecti += ok
            kaldi += not ok

        print("=" * 72)
        print(f"[{durum}] {q}")
        print(f"kaynaklar: {', '.join(kaynaklar) or '(yok)'}")
        print(cevap.strip())
        print()

    print("=" * 72)
    print(f"SONUC: {gecti} PASS, {kaldi} FAIL, {kontrolsuz} kontrolsuz / toplam {len(cases)}")


if __name__ == "__main__":
    main()
