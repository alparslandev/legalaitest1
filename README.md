# Hukuki Belge Asistanı — Halüsinasyonsuz, Kaynaklı Soru-Cevap (RAG)

Kendi hukuki belge arşivin üzerinde, **yalnızca o belgelere dayanarak** cevap veren
bir soru-cevap sistemi. Model bilgi uydurmaz: her cevabın altında hangi belgeden
geldiğini gösterir, belgelerde cevap yoksa **"Bu konuda sağlanan belgelerde bilgi
bulamadım."** der.

> Terminal üzerinden çalışır. Cevap üretimi için bulut **Claude Haiku** ya da
> tamamen yerel/bedava bir model (**Ollama** üzerinden DeepSeek, Gemma vb.) kullanabilirsin.

---

## İçindekiler
- [Bu proje ne işe yarar?](#bu-proje-ne-işe-yarar)
- [Neden "eğitim" değil, RAG?](#neden-eğitim-değil-rag)
- [Nasıl çalışır?](#nasıl-çalışır)
- [Halüsinasyonu nasıl engelliyor?](#halüsinasyonu-nasıl-engelliyor)
- [Kurulum](#kurulum)
- [Hızlı demo](#hızlı-demo-kurgusal-örneklerle)
- [Kendi belgelerinle kullanım](#kendi-belgelerinle-kullanım)
- [Komutlar](#komutlar)
- [Modeller (iki motor)](#modeller-iki-motor)
- [Yapılandırma](#yapılandırma-ragconfigpy)
- [Doğruluğu ölçmek](#doğruluğu-ölçmek-evalpy)
- [Dosya yapısı](#dosya-yapısı)
- [Gizlilik](#gizlilik)
- [Ölçeklenme](#ölçeklenme)
- [Sınırlamalar (dürüst notlar)](#sınırlamalar-dürüst-notlar)

---

## Bu proje ne işe yarar?

Büyük dil modellerine doğrudan hukuki soru sorduğunda, model emin bir tonla
**uydurabilir** (halüsinasyon) — yanlış madde numarası, var olmayan içtihat, vb.
Hukukta bu kabul edilemez.

Bu sistem bu sorunu şöyle çözer: soru geldiğinde önce **senin belge arşivinden**
ilgili bölümleri bulur, sonra modele "yalnızca bu metne dayan, kaynağını göster,
metinde yoksa bilmiyorum de" talimatıyla iletir. Sonuç:

- Cevaplar **senin belgelerinden** gelir, modelin genel ezberinden değil
- Her cevap **kaynaklı**: hangi dosya/madde olduğu görünür
- Belgede yoksa **uydurmaz**, açıkça "bulamadım" der
- Belge ekleme/çıkarma saniyeler sürer (yeniden eğitim yok)

Tipik kullanım: mevzuat, içtihat, yönetmelik, sözleşme arşivleri üzerinde
güvenilir, kaynak gösteren bir danışman.

---

## Neden "eğitim" değil, RAG?

Sezgisel olarak "AI'yı belgelerimle eğiteyim" mantıklı gelir; ama teknik gerçek farklıdır:

| | Fine-tuning (eğitim) | **RAG (bu proje)** |
|---|---|---|
| Ne öğretir | Üslup, format, davranış | — (öğretmez) |
| Bilgi kaynağı | Modelin ağırlıklarına gömülür (bulanık) | Soru anında belgeden **getirilir** (net) |
| Halüsinasyon | **Azaltmaz, çoğu zaman artırır** | Kaynağa zorlayarak **engeller** |
| Kaynak gösterme | Yapamaz | Her cevapta gösterir |
| Belge güncelleme | Yeniden eğitim (pahalı, yavaş) | `ingest.py` (saniyeler) |

Halüsinasyonsuzluğun anahtarı **eğitim değil**, "getir + kaynağa zorla + yoksa sustur"
disiplinidir. Bu proje tam olarak bunu uygular.

---

## Nasıl çalışır?

İki aşama var: **(1) İndeksleme** (bir kez, belge ekledikçe tekrar) ve
**(2) Soru-cevap** (her soruda).

```
1) İNDEKSLEME  (ingest.py)
   documents/*.txt,*.md,*.docx,*.pdf
        │  oku
        ▼
   gürültü temizliği  (URL, sayfa no, tarih başlığı atılır)
        │
        ▼
   cümle bütünlüğünde parçalama (chunk)   ~1200 karakter, örtüşmeli
        │
        ▼
   yerel embedding modeli (BAAI/bge-m3)  → her parça 1024 boyutlu vektör
        │
        ▼
   storage/  (vectors.npy + chunks.json)


2) SORU-CEVAP  (ask.py)
   soru
    │  aynı modelle vektöre çevir
    ▼
   arama: en benzer parçaları getir (cosine, ilk TOP_K)
    │
    ▼
   modele gönder:  SİSTEM TALİMATI + [getirilen parçalar] + SORU
    │             "sadece bu metne dayan, kaynak göster, yoksa bulamadım de"
    ▼
   cevap + kullanılan kaynaklar
```

**Bileşenler:**
- **Embedding (arama):** `BAAI/bge-m3` — çok dilli, Türkçe + uzun hukuki metinde güçlü.
  Yerelde çalışır, **bedava**, ikinci bir API anahtarı istemez.
- **Vektör veritabanı:** Harici bir vektör DB (Pinecone, Qdrant, Chroma, pgvector, Milvus)
  **kullanılmadı**. Bunun yerine `rag/store.py` içinde **yerleşik, hafif bir depo** var:
  vektörler `storage/vectors.npy` (NumPy dizisi) + üstveri `storage/chunks.json`; arama
  **cosine benzerliği** ile yapılır (`numpy` ile, normalize edilmiş vektörlerde nokta çarpımı).
  Küçük/orta veri için hızlı ve bağımlılıksızdır; büyük ölçekte gerçek bir vektör DB'ye
  geçilir (bkz. [Ölçeklenme](#ölçeklenme)).
- **Cevap üreten model:** bulut Haiku ya da yerel Ollama (config'den seçilir).

---

## Halüsinasyonu nasıl engelliyor?

Dört katmanlı güvence:

1. **Sıkı sistem talimatı** (`rag/llm.py`): "Yalnızca verilen belgeleri kullan;
   genel hukuk bilginden yararlanma; her bilgiyi `(Kaynak: dosya)` ile göster;
   cevap yoksa aynen 'Bu konuda sağlanan belgelerde bilgi bulamadım.' yaz."
2. **Belge-içi çıkarıma izin, belge-dışı bilgiye yasak:** model belgedeki bir
   kuralı sorudaki duruma uygulayabilir, ama belgede geçmeyen bilgi ekleyemez.
3. **`temperature = 0`:** en tutarlı, en az "yaratıcı" üretim.
4. **Kaynak şeffaflığı:** cevabın altında kullanılan belgeler listelenir; doğrulayabilirsin.

Gerçek bir testte model, **bildiği** bir genel-hukuk cevabını (İş Kanunu ihbar
süreleri) bile, belgede yazmadığı için **kullanmadı** ve "bulamadım" dedi —
istenen davranış tam olarak budur.

---

## Kurulum

Gereksinimler: Python 3.12+, (yerel model için) [Ollama](https://ollama.com).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Haiku (bulut) kullanacaksan:
cp .env.example .env        # .env içine: ANTHROPIC_API_KEY=sk-ant-...
```

İlk `ingest.py` çalıştırmasında embedding modeli (`bge-m3`, ~2 GB) bir kez iner,
sonra önbellekten gelir.

---

## Hızlı demo (örnek belgelerle)

`examples/` içinde iki tür örnek var: **kurgusal** belgeler (kira yönetmeliği,
çalışan el kitabı) ve **kamuya açık** gerçek belgeler (Yargıtay kararları, KVKK
Kurul Kararı — **telifli içerik yoktur; mahkeme kararları da anonimleştirilmiştir**).
Sistemi denemek için:

```bash
mkdir -p documents && cp examples/*.txt documents/
python ingest.py
python ask.py "konut kirasında depozito en fazla ne kadar olabilir?"
python eval.py             # tests/sorular.txt ile otomatik doğruluk testi
```

Beklenen: depozito sorusu "üç aylık" diye kaynaklı yanıtlanır; `eval.py` 5/5 PASS verir.

---

## Kendi belgelerinle kullanım

```bash
# 1) Belgelerini documents/ içine koy (.txt, .md, .docx, .pdf)
# 2) İndeksle (her belge ekleme/değişiminde tekrar):
python ingest.py
# 3) Sor:
python ask.py "kendi sorun"
python ask.py              # etkileşimli mod: arka arkaya sor, çıkış için q
```

> Açık duran `ask.py`/`compare.py` oturumu, her soruda indeksi diskten yeniden
> okur — yeni `ingest.py` sonrası **oturumu kapatmana gerek yoktur**.

---

## Komutlar

| Komut | Ne yapar |
|---|---|
| `python ingest.py` | `documents/`'ı temizler, parçalar, vektörler ve `storage/`'a yazar |
| `python ask.py "soru"` | Tek soru sorar (config'deki motorla) |
| `python ask.py` | Etkileşimli soru-cevap döngüsü (çıkış: `q`) |
| `python compare.py [soru]` | **Tüm modelleri** aynı soruda yan yana kıyaslar |
| `python eval.py` | `tests/sorular.txt` sorularını çalıştırır, PASS/FAIL özeti verir |

---

## Modeller (iki motor)

`rag/config.py` içindeki `LLM_BACKEND` tek satırla değiştirilir:

```python
LLM_BACKEND = "anthropic"   # Bulut Claude Haiku — en yüksek kalite, API kredisi harcar
LLM_BACKEND = "ollama"      # Yerel/bedava — veri makineden çıkmaz
#   ollama seçilirse:  OLLAMA_MODEL = "deepseek-r1:8b" | "gemma4:e4b" | ...
```

- **Haiku (bulut):** en tutarlı ve nüanslı; `ANTHROPIC_API_KEY` + kredi gerekir.
- **Ollama (yerel):** ücretsiz, gizli (veri dışarı çıkmaz); kaliteyi modele göre
  Haiku'ya yaklaştırabilir. Kıyaslamak için `compare.py` kullan.

---

## Yapılandırma (`rag/config.py`)

| Ayar | Varsayılan | Açıklama |
|---|---|---|
| `LLM_BACKEND` | `"anthropic"` | Cevap motoru: `anthropic` \| `ollama` |
| `LLM_MODEL` | `claude-haiku-4-5-...` | Bulut model kimliği |
| `OLLAMA_MODEL` | `deepseek-r1:8b` | Yerel model adı |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Arama (embedding) modeli |
| `CHUNK_TARGET_CHARS` | `1200` | Parça hedef boyutu |
| `CHUNK_OVERLAP_CHARS` | `200` | Parçalar arası örtüşme |
| `TOP_K` | `8` | Soru başına getirilen parça sayısı (recall) |
| `MIN_SIMILARITY` | `0.35` | Bu skorun altındaki parçalar "alakasız" sayılır |
| `LLM_TEMPERATURE` | `0.0` | 0 = en tutarlı, en az halüsinasyon |

İpucu: model "bulamadım" derken cevap aslında belgedeyse, bu çoğunlukla bir
**arama (recall)** sorunudur — `TOP_K`'yı artır (örn. 12). Yeniden indekslemeye
gerek yoktur; `TOP_K`/`MIN_SIMILARITY` anında etkilidir.

---

## Doğruluğu ölçmek (`eval.py`)

`tests/sorular.txt` içine soru + beklenen cevap çiftleri yazıp doğruluğu
**göz kararı değil, ölçülebilir** takip edersin:

```
soru | beklenen anahtar ifade     # cevapta bu geçmeli -> PASS
soru | BULAMADI                    # cevap "bulamadım" olmalı -> PASS
soru                               # sadece göster, otomatik kontrol yok
```

`python eval.py` her soruyu çalıştırır, `kaynaklar` satırını ve PASS/FAIL'i gösterir.
**FAIL teşhisi:**
- Doğru belge gelmiş ama cevap yanlış → model/talimat meselesi
- Doğru belge hiç gelmemiş → arama meselesi (`TOP_K`/chunking)

---

## Dosya yapısı

```
ingest.py            indeksleme (belge → parça → vektör → storage/)
ask.py               soru-cevap (tek model)
compare.py           tüm modelleri yan yana kıyas
eval.py              tests/sorular.txt ile doğruluk testi
rag/
  config.py          tüm ayarlar
  documents.py       belge okuma + gürültü temizliği + cümle-bazlı parçalama
  embeddings.py      yerel embedding modeli (bge-m3)
  store.py           numpy vektör deposu + cosine arama
  llm.py             Haiku/Ollama çağrısı + halüsinasyon engelleyen sistem talimatı
examples/            örnek belgeler: kurgusal + kamuya açık içtihat/karar (yayınlanır)
tests/sorular.txt    doğruluk testi soru/beklenen çiftleri
documents/           SENİN belgelerin — .gitignore'da, yayınlanmaz
storage/             üretilen indeks — .gitignore'da
```

### Python dosyaları: hangisini çalıştırıyorsun?

10 Python dosyası var ama **günlük kullanımda yalnızca 2-4'ünü sen çalıştırırsın**;
kalanı arka planda çalışan "motor"dur. Bu, tek dev dosya yerine her parçanın tek bir
iş yaptığı standart bir bölümlemedir (bakımı kolaylaştırır).

**Çalıştırdıkların — komutlar (kök dizinde):**

| Dosya | Komut | İş |
|---|---|---|
| `ingest.py` | `python ingest.py` | Belgeleri indeksle |
| `ask.py` | `python ask.py` | Soru-cevap (tek model) |
| `compare.py` | `python compare.py` | Modelleri yan yana kıyasla |
| `eval.py` | `python eval.py` | Doğruluk testi |

**Motor — `rag/` paketi (elle açmazsın; yukarıdaki komutlar bunları kullanır):**

| Dosya | İş |
|---|---|
| `rag/config.py` | Ayarlar — değişiklik yapacağın tek motor dosyası |
| `rag/documents.py` | Belge okuma (txt/docx/pdf) + temizleme + parçalama |
| `rag/embeddings.py` | Metni vektöre çeviren yerel model (bge-m3) |
| `rag/store.py` | Vektör deposu + arama (en benzer parçayı bul) |
| `rag/llm.py` | Modele sorma + halüsinasyon engelleyen sistem talimatı |
| `rag/__init__.py` | Boş — klasörü Python "paketi" yapar |

Bir şey değiştirmek istersen tek yere bakarsın: **model →** `rag/config.py`,
**PDF/okuma →** `rag/documents.py`, **cevap kuralları →** `rag/llm.py`.

---

## Gizlilik

- `documents/` ve `storage/` `.gitignore`'dadır → **kendi belgelerin ve indeksin
  asla yayınlanmaz**. Depo yalnızca `examples/` içindeki örneklerle gelir (kurgusal belgeler + kamuya açık içtihat/karar).
- `.env` (API anahtarı) `.gitignore`'dadır.
- **Bulut Haiku** kullanırken belge metni Anthropic'e gönderilir. Belgelerin gizliyse
  `LLM_BACKEND = "ollama"` ile tamamen yerel çalış — veri makineden hiç çıkmaz.

---

## Ölçeklenme

Başlangıçta vektörler basit bir numpy dosyasında tutulur — küçük/orta veri (binlerce
parça) için hızlı ve yeterli. Çok büyük korpusa (milyonlarca belge) çıkarken yalnızca
`rag/store.py` Qdrant/pgvector/Milvus gibi gerçek bir vektör veritabanıyla değiştirilir;
indeksleme ve soru-cevap akışının geri kalanı **aynı kalır**.

---

## Sınırlamalar (dürüst notlar)

- Sistem ancak **getirebildiği** belgeler kadar iyidir: doğru parça ilk `TOP_K`'ya
  giremezse model haklı olarak "bulamadım" der. Kalitenin asıl kaldıracı arama
  (chunking + `TOP_K` + embedding), modelin kendisi değil.
- Cevaplar üretildikleri belgelere bağlıdır; **belgeler eski/yanlışsa cevap da öyle
  olur**. Sistem belgeyi doğrulamaz, sadakatle aktarır.
- Yerel modellerin kalitesi Haiku'nun altında olabilir; kritik kullanımda
  `compare.py` ile kıyaslayıp karar ver.
- **Metin-tabanlı** PDF'ler (`.pdf`) doğrudan desteklenir (`pypdfium2` ile). Ancak
  **taranmış/görüntü PDF'lerden metin çıkmaz** — bunlar için önce OCR gerekir
  (sistem böyle bir dosyada "metin çıkarılamadı" uyarısı verir).
```
