"""Kaynaga zorlayan, halusinasyonu engelleyen cevap uretimi.
Iki motor destekler: yerel Ollama (DeepSeek) ve bulut Claude Haiku.
Hangisinin kullanilacagi rag/config.py icindeki LLM_BACKEND ile secilir."""
import os
import re

import httpx

from . import config

SYSTEM_PROMPT = """Sen bir hukuki belge asistanisin. Gorevin, asagida "BELGELER" \
basligi altinda sana verilen metinlere dayanarak soruyu yanitlamaktir.

Kurallar:
1. Yalnizca verilen belgelerdeki bilgileri kullan. Genel hukuk bilginden, ezberinden \
veya belgelerde hic gecmeyen bilgilerden ASLA yararlanma.
2. Belgelerdeki ifadelere dayanan, dogrudan ve acikca desteklenen cikarimlari \
yapabilirsin (ornegin belgedeki bir kurali sorudaki duruma uygulamak). Boyle bir \
cikarim yaparken hangi ifadeye dayandigini kaynagiyla goster; belge disi bilgi veya \
tahmin EKLEME.
3. Verdigin her bilginin ardindan (Kaynak: <dosya adi>) biciminde kaynagini belirt.
4. Cevap, belgelerden ne aktararak ne de dogrudan cikarimla elde edilemiyorsa, baska \
hicbir sey eklemeden aynen sunu yaz: "Bu konuda saglanan belgelerde bilgi bulamadim."
5. Acik ve duzgun bir Turkce ile yanitla."""

NO_INFO = "Bu konuda saglanan belgelerde bilgi bulamadim."


def build_context(chunks: list[dict]) -> str:
    blocks = [f"[Kaynak: {c['source']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(blocks)


def answer(question: str, chunks: list[dict], *, backend: str | None = None,
           model: str | None = None, keep_alive=None) -> str:
    if not chunks:
        return NO_INFO
    user_msg = f"BELGELER:\n{build_context(chunks)}\n\nSORU: {question}"
    backend = backend or config.LLM_BACKEND
    if backend == "ollama":
        return _answer_ollama(user_msg, model or config.OLLAMA_MODEL, keep_alive)
    if backend == "anthropic":
        return _answer_anthropic(user_msg, model or config.LLM_MODEL)
    raise SystemExit(f"Bilinmeyen LLM_BACKEND: {backend!r} (ollama|anthropic)")


# --- Yerel motor: Ollama (DeepSeek) ---

def _answer_ollama(user_msg: str, model: str, keep_alive=None) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": config.LLM_TEMPERATURE},
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive  # 0 => cevaptan sonra modeli bellekten birak
    try:
        resp = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=600)
    except httpx.ConnectError:
        raise SystemExit(
            "Ollama'ya baglanilamadi. Sunucu calisiyor mu?\n"
            "  Terminalde:  ollama serve   (ayri bir pencerede acik kalsin)"
        )
    if resp.status_code == 404:
        raise SystemExit(
            f"'{model}' modeli yuklu degil.\n"
            f"  Indir:  ollama pull {model}"
        )
    resp.raise_for_status()
    text = resp.json()["message"]["content"]
    # DeepSeek-R1 gibi modeller cevabi <think>...</think> ile baslatir; onu temizle.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# --- Bulut motor: Claude Haiku ---

def _answer_anthropic(user_msg: str, model: str) -> str:
    import anthropic
    from anthropic import Anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or key.startswith("sk-ant-buraya"):
        raise SystemExit(
            "\nANTHROPIC_API_KEY tanimli degil.\n"
            "  1) https://console.anthropic.com adresinden bir API anahtari al\n"
            "  2) Proje kokundeki .env dosyasina yaz: ANTHROPIC_API_KEY=sk-ant-...\n"
        )
    try:
        resp = Anthropic(api_key=key).messages.create(
            model=model,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.AuthenticationError:
        raise SystemExit("API anahtari gecersiz. .env icindeki ANTHROPIC_API_KEY'i kontrol et.")
    except anthropic.RateLimitError:
        raise SystemExit("Hiz limitine takildin. Biraz bekleyip tekrar dene.")
    except anthropic.APIStatusError as e:
        msg = str(getattr(e, "message", e))
        if "credit balance" in msg.lower():
            raise SystemExit(
                "Anthropic hesabinda kredi yok.\n"
                "  https://console.anthropic.com -> Plans & Billing -> kredi ekle."
            )
        raise SystemExit(f"Anthropic API hatasi: {msg}")
    return "".join(block.text for block in resp.content if block.type == "text")
