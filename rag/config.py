"""Merkezi ayarlar. Model, yol ve parametreleri buradan degistirebilirsin."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # .env icindeki ANTHROPIC_API_KEY'i ortama yukler

DOCUMENTS_DIR = ROOT / "documents"   # belgeleri buraya koy (.txt, .md, .docx)
STORAGE_DIR = ROOT / "storage"       # olusan indeks burada saklanir

# --- Embedding: aramayi saglar, metni vektore cevirir (yerel, bedava) ---
EMBEDDING_MODEL = "BAAI/bge-m3"      # cok dilli; Turkce + uzun hukuki metin icin guclu

# --- Cevap ureten model: motor secimi ---
# "ollama"    -> yerel, bedava (Ollama uzerinden DeepSeek). Mac'te calisir.
# "anthropic" -> bulut Claude Haiku (console.anthropic.com'da kredi gerektirir).
LLM_BACKEND = "anthropic"

# Bulut Haiku (LLM_BACKEND="anthropic" ise kullanilir)
LLM_MODEL = "claude-haiku-4-5-20251001"

# Yerel Ollama (LLM_BACKEND="ollama" ise kullanilir)
OLLAMA_MODEL = "deepseek-r1:8b"      # Turkce icin alternatif: "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434"

LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.0                # 0 = en tutarli, en az "yaratici" => halusinasyon riski en dusuk

# --- Parcalama (chunking) ---
CHUNK_TARGET_CHARS = 1200            # bir parcanin hedef boyutu
CHUNK_OVERLAP_CHARS = 200            # parcalar arasi orpusme (baglam kopmasin)

# --- Arama ---
TOP_K = 8                            # soru basina getirilecek parca sayisi (recall icin artirildi)
MIN_SIMILARITY = 0.35                # bu skorun altindaki parcalar "alakasiz" sayilir
