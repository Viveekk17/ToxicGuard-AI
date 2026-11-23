"""CO2 - Sentiment analysis & information extraction.  CO3 - Topic modelling."""
from __future__ import annotations

import re
from functools import lru_cache

import joblib
import numpy as np

from .config import ARTIFACTS_DIR
from .lexicon import ABUSE_LEXICON, IDENTITY_TERMS, SECOND_PERSON
from .preprocessing import extract_entities_regex

WORD_RE = re.compile(r"[ऀ-ॿ]+|[A-Za-z']+")


# ================================================================== Sentiment (VADER)
@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def sentiment(text_en: str) -> dict:
    """Lexicon + rule-based sentiment (VADER) on the English (translated) text.

    VADER handles negation, intensifiers, capitalisation and punctuation emphasis, which are
    frequent in social-media comments. Labels follow the standard compound-score thresholds.
    """
    s = _vader().polarity_scores(text_en)
    c = s["compound"]
    label = "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral"
    intensity = "Strong" if abs(c) >= 0.6 else "Moderate" if abs(c) >= 0.3 else "Mild"
    return {"label": label, "intensity": intensity, "compound": c, "positive": s["pos"], "neutral": s["neu"], "negative": s["neg"]}


# ================================================================== Information extraction
@lru_cache(maxsize=1)
def _spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def information_extraction(original: str, text_en: str) -> dict:
    """Extract who/what the comment is about.

    * Named entities (spaCy statistical NER on the English text, if installed)
    * Identity-group mentions (multilingual gazetteer)  -> identity-hate evidence
    * Target type (individual via 2nd-person pronouns / @mentions, or a group)
    * Abusive terms (multilingual lexicon) with category
    * Structured artefacts (URLs, e-mails, @mentions, #hashtags, phone numbers) via regex
    """
    words_orig = [w.lower() for w in WORD_RE.findall(original)]
    words_en = [w.lower() for w in WORD_RE.findall(text_en)]
    all_words = set(words_orig) | set(words_en)

    identities = {}
    for group, terms in IDENTITY_TERMS.items():
        hits = sorted({t for t in terms if t in all_words})
        if hits:
            identities[group] = hits

    regex_ents = extract_entities_regex(original)
    second_person = sorted({w for w in all_words if w in SECOND_PERSON})
    if second_person or regex_ents["mentions"]:
        target = "Individual (directly addressed)"
    elif identities:
        target = "Group / identity (" + ", ".join(g.replace("_", " ") for g in identities) + ")"
    else:
        target = "No explicit target"

    abusive = []
    seen = set()
    for w in words_orig + words_en:
        if w in ABUSE_LEXICON and w not in seen:
            seen.add(w)
            abusive.append({"term": w, "category": ABUSE_LEXICON[w]})

    entities = []
    nlp = _spacy()
    if nlp is not None and text_en.strip():
        doc = nlp(text_en)
        entities = [{"text": e.text, "label": e.label_, "description": _spacy_label(e.label_)} for e in doc.ents]

    return {
        "target": target,
        "second_person_pronouns": second_person,
        "identity_mentions": identities,
        "abusive_terms": abusive,
        "named_entities": entities,
        "ner_available": nlp is not None,
        **regex_ents,
    }


def _spacy_label(label: str) -> str:
    try:
        import spacy

        return spacy.explain(label) or label
    except Exception:
        return label


# ================================================================== Topic modelling (LDA)
@lru_cache(maxsize=1)
def load_topic_model():
    """Corpus-level LDA trained in the notebook (artifacts/topic_model.joblib)."""
    p = ARTIFACTS_DIR / "topic_model.joblib"
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


@lru_cache(maxsize=1)
def load_tfidf():
    p = ARTIFACTS_DIR / "tfidf.joblib"
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


def topic_of(normalized_en: str) -> dict | None:
    tm = load_topic_model()
    if tm is None or not normalized_en.strip():
        return None
    vec, lda, labels = tm["vectorizer"], tm["lda"], tm["topic_labels"]
    dist = lda.transform(vec.transform([normalized_en]))[0]
    order = np.argsort(-dist)
    feats = vec.get_feature_names_out()
    topics = []
    for i in order:
        top_words = [feats[j] for j in np.argsort(-lda.components_[i])[:8]]
        topics.append({"topic": labels[i], "weight": float(dist[i]), "keywords": top_words})
    return {"dominant": topics[0], "distribution": topics}


def fit_batch_topics(docs: list[str], n_topics: int = 4, n_words: int = 8) -> dict | None:
    """Fit an LDA model on an uploaded batch of comments (unsupervised topic discovery)."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    docs = [d for d in docs if d.strip()]
    if len(docs) < 5:
        return None
    vec = CountVectorizer(max_df=0.9, min_df=1 if len(docs) < 50 else 2, max_features=3000)
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return None
    n_topics = max(2, min(n_topics, X.shape[0] - 1, X.shape[1]))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    doc_topic = lda.fit_transform(X)
    feats = vec.get_feature_names_out()
    topics = [{"topic": f"Topic {i+1}", "keywords": [feats[j] for j in np.argsort(-comp)[:n_words]]}
              for i, comp in enumerate(lda.components_)]
    return {"topics": topics, "doc_topic": doc_topic.argmax(axis=1).tolist(), "perplexity": float(lda.perplexity(X))}
