"""CO3 - Multilingual NLP: language / script identification (English, Hindi, Hinglish)
and translation through a cloud AI service (Google Cloud Translate via deep-translator)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .lexicon import HINGLISH_MARKERS

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-z']+")

LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish (code-mixed)", "mixed": "Mixed script", "other": "Other"}


@dataclass
class LanguageInfo:
    code: str                 # en | hi | hinglish | mixed | other
    name: str
    script: str               # Latin | Devanagari | Mixed
    devanagari_ratio: float
    hinglish_ratio: float     # share of Latin words that are Hinglish markers
    hinglish_markers: list[str]
    detector: str             # which rule decided


def detect_language(text: str) -> LanguageInfo:
    """Rule-based + statistical language identification tuned for Indian social-media text.

    1. Script analysis (Devanagari vs Latin character ratio).
    2. Hinglish lexical markers (romanised Hindi function words).
    3. Fallback statistical detector (``langdetect``) for other Latin text.
    """
    dev = len(DEVANAGARI_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    letters = dev + lat
    dev_ratio = dev / letters if letters else 0.0

    words = [w.lower() for w in WORD_RE.findall(text)]
    markers = [w for w in words if w in HINGLISH_MARKERS]
    hing_ratio = len(markers) / len(words) if words else 0.0

    if letters == 0:
        return LanguageInfo("other", LANG_NAMES["other"], "None", 0.0, 0.0, [], "no letters")
    if dev_ratio >= 0.6:
        return LanguageInfo("hi", LANG_NAMES["hi"], "Devanagari", dev_ratio, hing_ratio, markers, "script ratio")
    if dev_ratio >= 0.15:
        return LanguageInfo("mixed", LANG_NAMES["mixed"], "Mixed", dev_ratio, hing_ratio, markers, "script ratio")

    # Latin script: Hinglish vs English
    if (len(set(markers)) >= 2 and hing_ratio >= 0.2) or (len(words) <= 4 and len(set(markers)) >= 1 and hing_ratio >= 0.34):
        return LanguageInfo("hinglish", LANG_NAMES["hinglish"], "Latin", dev_ratio, hing_ratio, markers, "Hinglish lexical markers")

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        code = detect(text)
        if code == "hi":
            return LanguageInfo("hinglish", LANG_NAMES["hinglish"], "Latin", dev_ratio, hing_ratio, markers, "langdetect")
        if code != "en" and hing_ratio >= 0.15:
            return LanguageInfo("hinglish", LANG_NAMES["hinglish"], "Latin", dev_ratio, hing_ratio, markers, "langdetect + markers")
    except Exception:
        pass
    return LanguageInfo("en", LANG_NAMES["en"], "Latin", dev_ratio, hing_ratio, markers, "default / langdetect")


def _google_cloud_v2(text: str, key: str) -> str | None:
    """Official Google Cloud Translation API (v2) - used when GOOGLE_TRANSLATE_API_KEY is set."""
    import requests

    r = requests.post("https://translation.googleapis.com/language/translate/v2",
                      params={"key": key}, data={"q": text, "target": "en", "format": "text"}, timeout=10)
    r.raise_for_status()
    return r.json()["data"]["translations"][0]["translatedText"]


def _google_public(text: str) -> str | None:
    """Google Translate public endpoint (no key). Auto-detects romanised Hindi ("hi-Latn")."""
    import requests

    r = requests.get("https://translate.googleapis.com/translate_a/single",
                     params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text}, timeout=10)
    r.raise_for_status()
    return "".join(seg[0] for seg in r.json()[0] if seg and seg[0])


def _deep_translator(text: str) -> str | None:
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source="auto", target="en").translate(text)


@lru_cache(maxsize=1024)
def translate_to_english(text: str) -> tuple[str, bool]:
    """Translate Hindi / Hinglish text to English with Google's cloud translation service.

    Tries, in order: official Cloud Translation API (if a key is configured) -> public Google
    endpoint -> deep-translator. Returns ``(translation, ok)``; on failure the input is returned.
    """
    from .config import _secret

    text = text[:4500]
    providers = []
    key = _secret("GOOGLE_TRANSLATE_API_KEY")
    if key:
        providers.append(lambda t: _google_cloud_v2(t, key))
    providers += [_google_public, _deep_translator]
    for fn in providers:
        try:
            out = fn(text)
            if out and out.strip():
                return out.strip(), True
        except Exception:
            continue
    return text, False
