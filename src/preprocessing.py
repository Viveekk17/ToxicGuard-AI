"""CO1 - Text extraction, preprocessing, cleaning and vectorisation of unstructured data."""
from __future__ import annotations

import html
import io
import re
import unicodedata
from dataclasses import dataclass, field

from .lexicon import HINGLISH_NORMALIZATION

# ------------------------------------------------------------------ regex patterns
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\d{10}\b|\+?\d{1,3}[\s-]\d{3,5}[\s-]\d{4,6}")
ELONGATION_RE = re.compile(r"(\w)\1{2,}")          # "stupiddddd" -> "stupidd"
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1+")       # "!!!!" -> "!"
WHITESPACE_RE = re.compile(r"\s+")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# Word tokens: Latin/digits/apostrophes, or Devanagari runs (incl. vowel signs & nukta).
TOKEN_RE = re.compile(r"[ऀ-ॿ]+|[A-Za-z0-9']+")

STOPWORDS_EN = set("""a an the and or but if while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there when where
why how all any both each few more most other some such no nor not only own same so than too very s t can will
just don should now i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she
her hers herself it its itself they them their theirs themselves what which who whom this that these those am is
are was were be been being have has had having do does did doing would could ought i'm you're he's she's it's
we're they're i've you've we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll
isn't aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't
cannot couldn't mustn't let's that's who's what's here's there's when's where's why's how's also get got""".split())
STOPWORDS_HI = set("""का की के को में से है हैं था थी थे और या पर भी तो ही यह वह ये वो इस उस एक कि जो कर
करना किया गया हो होता होती रहा रही रहे नहीं ना लिए साथ तक अब जब तब कुछ सब हम आप मैं मुझे मेरा
hai hain ho tha thi the aur ya par bhi to hi yeh ye woh wo is us ek ki ka ke ko mein me se kar karna kiya
gaya hota hoti raha rahi rahe nahi na liye sath tak ab jab tab kuch sab hum main mujhe mera meri mere""".split())
STOPWORDS = STOPWORDS_EN | STOPWORDS_HI


@dataclass
class PreprocessResult:
    original: str
    model_text: str                 # light cleaning -> fed to BERT (keeps casing & meaning)
    normalized_text: str            # aggressive normalisation -> TF-IDF / LDA
    tokens: list[str]
    tokens_no_stop: list[str]
    steps: list[dict] = field(default_factory=list)   # [{step, output}] for the UI audit trail
    extracted: dict = field(default_factory=dict)      # urls / mentions / hashtags / emails / phones


# ------------------------------------------------------------------ extraction
def extract_text_from_upload(name: str, data: bytes) -> list[str]:
    """Extract a list of comments from an uploaded TXT / CSV / PDF / JSON file."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 2]
    if lower.endswith(".csv") or lower.endswith(".xlsx"):
        import pandas as pd

        df = pd.read_csv(io.BytesIO(data)) if lower.endswith(".csv") else pd.read_excel(io.BytesIO(data))
        preferred = [c for c in df.columns if str(c).lower() in {"comment", "comment_text", "text", "comments", "message", "content"}]
        col = preferred[0] if preferred else df.select_dtypes(include="object").columns[0]
        return [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    if lower.endswith(".json"):
        import json

        obj = json.loads(data.decode("utf-8", errors="ignore"))
        if isinstance(obj, list):
            return [str(o.get("text", o) if isinstance(o, dict) else o) for o in obj]
        return [str(obj)]
    text = data.decode("utf-8", errors="ignore")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def extract_entities_regex(text: str) -> dict:
    return {
        "urls": URL_RE.findall(text),
        "emails": EMAIL_RE.findall(text),
        "mentions": MENTION_RE.findall(text),
        "hashtags": ["#" + h for h in HASHTAG_RE.findall(text)],
        "phones": [p.strip() for p in PHONE_RE.findall(text)],
    }


# ------------------------------------------------------------------ cleaning
def clean_for_model(text: str) -> str:
    """Light, meaning-preserving cleaning used for the transformer input."""
    t = html.unescape(text)
    t = unicodedata.normalize("NFC", t)
    t = HTML_TAG_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = MENTION_RE.sub("@user", t)
    t = HASHTAG_RE.sub(r"\1", t)
    t = ELONGATION_RE.sub(r"\1\1", t)
    t = REPEAT_PUNCT_RE.sub(r"\1", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t or text.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def preprocess(text: str, language: str = "en") -> PreprocessResult:
    """Full, step-by-step preprocessing pipeline (every step is recorded for the UI)."""
    steps: list[dict] = [{"step": "0. Raw input", "output": text}]

    t = html.unescape(text)
    t = HTML_TAG_RE.sub(" ", t)
    steps.append({"step": "1. HTML unescape & tag removal", "output": t})

    t = unicodedata.normalize("NFC", t)
    steps.append({"step": "2. Unicode NFC normalisation (Devanagari safe)", "output": t})

    extracted = extract_entities_regex(t)
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = MENTION_RE.sub(" @user ", t)
    t = HASHTAG_RE.sub(r"\1", t)
    steps.append({"step": "3. URL / e-mail removal, @mention masking, #hashtag split", "output": WHITESPACE_RE.sub(" ", t).strip()})

    t = ELONGATION_RE.sub(r"\1\1", t)
    t = REPEAT_PUNCT_RE.sub(r"\1", t)
    steps.append({"step": "4. Character elongation & punctuation squashing", "output": WHITESPACE_RE.sub(" ", t).strip()})

    model_text = WHITESPACE_RE.sub(" ", t).strip() or text.strip()

    t = t.lower()
    steps.append({"step": "5. Case folding", "output": WHITESPACE_RE.sub(" ", t).strip()})

    tokens = tokenize(t)
    steps.append({"step": "6. Tokenisation (Latin + Devanagari aware)", "output": " | ".join(tokens)})

    if language in ("hinglish", "mixed"):
        tokens = [HINGLISH_NORMALIZATION.get(tok, tok) for tok in tokens]
        steps.append({"step": "7. Hinglish spelling normalisation", "output": " | ".join(tokens)})

    tokens_no_stop = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1 and not tok.isdigit()]
    steps.append({"step": f"{len(steps)}. Stop-word removal (English + Hindi + Hinglish)", "output": " | ".join(tokens_no_stop)})

    return PreprocessResult(
        original=text,
        model_text=model_text,
        normalized_text=" ".join(tokens_no_stop),
        tokens=tokens,
        tokens_no_stop=tokens_no_stop,
        steps=steps,
        extracted=extracted,
    )


# ------------------------------------------------------------------ vectorisation
def tfidf_vector(normalized_text: str, vectorizer=None, top_k: int = 15) -> list[dict]:
    """Return the top TF-IDF weighted terms of a document.

    Uses the corpus-level vectorizer fitted in the training notebook (``artifacts/tfidf.joblib``)
    when available; otherwise falls back to a term-frequency vector of the single document.
    """
    if not normalized_text.strip():
        return []
    if vectorizer is not None:
        vec = vectorizer.transform([normalized_text])
        feats = vectorizer.get_feature_names_out()
        row = vec.tocoo()
        pairs = sorted(zip(row.col, row.data), key=lambda x: -x[1])[:top_k]
        return [{"term": feats[i], "tfidf": round(float(w), 4)} for i, w in pairs]

    from collections import Counter

    counts = Counter(normalized_text.split())
    total = sum(counts.values())
    return [{"term": w, "tf": round(c / total, 4)} for w, c in counts.most_common(top_k)]
