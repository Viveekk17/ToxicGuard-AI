"""End-to-end analysis pipeline tying CO1-CO4 together."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .analysis import information_extraction, load_tfidf, sentiment, topic_of
from .classifier import ToxicityClassifier
from .config import LABELS
from .explainer import Explanation, explain, natural_language_rationale
from .lexicon import ABUSE_LEXICON
from .multilingual import LanguageInfo, detect_language, translate_to_english
from .preprocessing import PreprocessResult, preprocess, tfidf_vector, tokenize


@dataclass
class Settings:
    translation_ensemble: bool = True
    use_ig: bool = True
    use_occlusion: bool = True
    use_lime: bool = False
    ig_steps: int = 24
    lime_samples: int = 200


@dataclass
class AnalysisResult:
    text: str
    language: LanguageInfo
    prep: PreprocessResult
    translation: str
    translated_ok: bool
    probs: dict
    probs_original: dict
    probs_translated: dict | None
    positives: list[str]
    is_toxic: bool
    severity: str
    target_label: str
    explanation: Explanation
    ensemble_rule: str | None
    explained_text: str
    rationale: list[str]
    sentiment: dict
    info: dict
    tfidf: list[dict]
    topic: dict | None
    subwords: list[dict]
    timings: dict = field(default_factory=dict)


def ensemble_rule(tokens: list[str]) -> str:
    """Lexicon-gated translation ensemble.

    If the original comment or its translation contains a known abusive term, keep the MAX of the original and translation
    scores (do not let a soft translation hide real abuse). Otherwise AVERAGE them: romanised Hinglish
    can trigger false alarms in mBERT that a clean English translation cancels out.
    """
    return "max" if any(t.lower() in ABUSE_LEXICON for t in tokens) else "mean"


def combine(p_orig: np.ndarray, p_tr: np.ndarray, rule: str) -> np.ndarray:
    return np.maximum(p_orig, p_tr) if rule == "max" else (p_orig + p_tr) / 2


def severity_level(probs: dict, positives: list[str]) -> str:
    m = max(probs.values())
    if not positives:
        return "Borderline" if m >= 0.3 else "Safe"
    if "severe_toxic" in positives or "threat" in positives or m >= 0.9:
        return "Severe"
    if m >= 0.75 or len(positives) >= 3:
        return "High"
    return "Moderate"


def analyze(clf: ToxicityClassifier, text: str, settings: Settings) -> AnalysisResult:
    t = {}
    t0 = time.perf_counter()

    lang = detect_language(text)
    prep = preprocess(text, lang.code)
    t["preprocessing"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    if lang.code == "en":
        translation, ok = prep.model_text, True
    else:
        translation, ok = translate_to_english(prep.model_text)
    t["translation"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    texts = [prep.model_text]
    use_tr = settings.translation_ensemble and lang.code != "en" and ok and translation != prep.model_text
    if use_tr:
        texts.append(translation)
    p = clf.predict_proba(texts)
    p_orig = p[0]
    p_tr = p[1] if use_tr else None
    rule = ensemble_rule(prep.tokens + tokenize(translation)) if p_tr is not None else None
    final = combine(p_orig, p_tr, rule) if p_tr is not None else p_orig
    probs = {lab: float(v) for lab, v in zip(LABELS, final)}
    t["classification"] = time.perf_counter() - t2

    positives = [lab for lab in LABELS if probs[lab] >= clf.thresholds.get(lab, 0.5)]
    # Any sub-category implies the comment is toxic overall.
    if positives and "toxic" not in positives:
        positives.insert(0, "toxic")
    is_toxic = bool(positives)

    # Explain the strongest *specific* category if present, else "toxic".
    specific = [lab for lab in positives if lab != "toxic"]
    target = max(specific, key=lambda lab: probs[lab]) if specific else "toxic"

    # Explain on whichever input (original / translation) drove the decision for that label.
    li = LABELS.index(target)
    explain_text = translation if (p_tr is not None and p_tr[li] > p_orig[li] + 0.15) else prep.model_text

    t3 = time.perf_counter()
    exp = explain(clf, explain_text, target, settings.use_ig, settings.use_occlusion,
                  settings.use_lime, settings.ig_steps, settings.lime_samples)
    t["explainability"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    senti = sentiment(translation)
    info = information_extraction(text, translation)
    norm_en = preprocess(translation, "en").normalized_text if lang.code != "en" else prep.normalized_text
    tfidf = tfidf_vector(norm_en, load_tfidf())
    topic = topic_of(norm_en)
    t["linguistic_analysis"] = time.perf_counter() - t4

    return AnalysisResult(
        text=text, language=lang, prep=prep, translation=translation, translated_ok=ok,
        probs=probs,
        probs_original={lab: float(v) for lab, v in zip(LABELS, p_orig)},
        probs_translated={lab: float(v) for lab, v in zip(LABELS, p_tr)} if p_tr is not None else None,
        ensemble_rule=rule,
        positives=positives, is_toxic=is_toxic, severity=severity_level(probs, positives),
        target_label=target, explanation=exp, explained_text=explain_text,
        rationale=natural_language_rationale(exp, probs, positives),
        sentiment=senti, info=info, tfidf=tfidf, topic=topic,
        subwords=clf.subword_view(prep.model_text), timings=t,
    )


def quick_classify(clf: ToxicityClassifier, texts: list[str], translation_ensemble: bool = True) -> list[dict]:
    """Fast batch path (no XAI) used for file uploads."""
    rows = []
    langs = [detect_language(x) for x in texts]
    cleaned = [preprocess(x, lg.code).model_text for x, lg in zip(texts, langs)]
    p = clf.predict_proba(cleaned)
    for i, (x, lg) in enumerate(zip(texts, langs)):
        probs = p[i]
        tr = x
        if translation_ensemble and lg.code != "en":
            tr, ok = translate_to_english(cleaned[i])
            if ok and tr != cleaned[i]:
                probs = combine(probs, clf.predict_proba([tr])[0], ensemble_rule(preprocess(x, lg.code).tokens + tokenize(tr)))
        pos = [lab for lab, v in zip(LABELS, probs) if v >= clf.thresholds.get(lab, 0.5)]
        if pos and "toxic" not in pos:
            pos.insert(0, "toxic")
        rows.append({"comment": x, "language": lg.name, "translation": tr,
                     **{lab: float(v) for lab, v in zip(LABELS, probs)},
                     "is_toxic": bool(pos), "categories": ", ".join(pos) or "-"})
    return rows
