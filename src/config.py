"""Central configuration for ToxicGuardAI.

Values can be overridden with environment variables or Streamlit secrets
(``.streamlit/secrets.toml`` locally, "Secrets" panel on Streamlit Cloud).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

APP_NAME = "ToxicGuardAI"
APP_TAGLINE = "Explainable multilingual toxic comment detection with BERT + XAI + GenAI"

# Jigsaw label schema used by both the fine-tuned model and the fallback model.
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_DISPLAY = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate",
}
LABEL_DESCRIPTION = {
    "toxic": "Rude, disrespectful or unreasonable language likely to make someone leave a discussion.",
    "severe_toxic": "Extremely hateful, aggressive or vulgar language.",
    "obscene": "Profanity, sexually explicit or vulgar content.",
    "threat": "Intent or wish to inflict pain, injury or violence on someone.",
    "insult": "Insulting, inflammatory or demeaning language directed at a person.",
    "identity_hate": "Hate directed at a protected identity (religion, caste, race, gender, orientation, nationality...).",
}

# Model trained in notebooks/ToxicGuardAI_Training.ipynb and pushed to the Hugging Face Hub.
# Until you train your own, the app falls back to a public Jigsaw BERT model.
FALLBACK_MODEL_ID = "unitary/toxic-bert"

CLAUDE_MODEL = "claude-opus-5-5"
# Default Gemini model (override with GEMINI_MODEL in secrets). If it is unavailable for a key,
# the app falls back through GEMINI_FALLBACK_MODELS automatically.
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]


def _secret(name: str, default: str | None = None) -> str | None:
    """Read a value from env vars first, then Streamlit secrets (if available)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def model_id() -> str:
    local = ARTIFACTS_DIR / "model"
    if (local / "config.json").exists():
        return str(local)
    return _secret("MODEL_ID", FALLBACK_MODEL_ID) or FALLBACK_MODEL_ID


def gemini_api_key() -> str | None:
    return _secret("GEMINI_API_KEY")


def gemini_model() -> str:
    return _secret("GEMINI_MODEL", GEMINI_DEFAULT_MODEL) or GEMINI_DEFAULT_MODEL


def anthropic_api_key() -> str | None:
    return _secret("ANTHROPIC_API_KEY")


def hf_token() -> str | None:
    return _secret("HF_TOKEN")
