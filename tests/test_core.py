"""Lightweight unit tests (no model download needed).  Run: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import fit_batch_topics, information_extraction, sentiment
from src.llm import OUTPUT_SCHEMA, build_prompt, offline_analysis
from src.multilingual import detect_language
from src.preprocessing import clean_for_model, extract_text_from_upload, preprocess


def test_language_detection():
    assert detect_language("You are a wonderful person").code == "en"
    assert detect_language("तू एक नंबर का बेवकूफ है").code == "hi"
    assert detect_language("Tu pagal hai kya yaar, chup kar").code == "hinglish"


def test_cleaning():
    t = clean_for_model("Check https://x.com @rahul #fail stupiddddd!!!!")
    assert "https" not in t and "@user" in t and "stupidd" in t and "!!" not in t
    r = preprocess("Tu nhi samjhega yaar", "hinglish")
    assert "nahi" in r.tokens and len(r.steps) >= 7


def test_extraction_from_csv():
    rows = extract_text_from_upload("c.csv", b"id,comment\n1,hello there\n2,you idiot\n")
    assert rows == ["hello there", "you idiot"]


def test_sentiment_and_ie():
    assert sentiment("I love this, great work!")["label"] == "Positive"
    info = information_extraction("All immigrants are criminals", "All immigrants are criminals")
    assert "nationality_ethnicity" in info["identity_mentions"]
    assert information_extraction("you idiot", "you idiot")["target"].startswith("Individual")


def test_topics():
    docs = ["cricket match india win", "cricket bat ball", "movie actor film", "film song actor", "match score cricket", "actor movie hit"]
    assert fit_batch_topics(docs, 2)["topics"]


def test_prompts_and_offline():
    sys_p, user_p = build_prompt("xai_grounded", "Tu pagal hai", "Hinglish", "You are crazy", {"toxic": 0.9}, [("pagal", 1.0)])
    assert "classifier_evidence" in user_p and "Examples" in sys_p
    out = offline_analysis("you idiot", ["toxic", "insult"], {"toxic": 0.9, "insult": 0.8}, [("idiot", 1.0)], "Individual", "en")
    assert set(OUTPUT_SCHEMA["required"]) <= set(out)


def test_ensemble_rule():
    import numpy as np
    from src.pipeline import combine, ensemble_rule

    assert ensemble_rule(["yaar", "movie", "acchi"]) == "mean"
    assert ensemble_rule(["tu", "bewakoof", "hai"]) == "max"
    assert np.allclose(combine(np.array([0.8]), np.array([0.0]), "mean"), [0.4])
    assert np.allclose(combine(np.array([0.8]), np.array([0.0]), "max"), [0.8])
