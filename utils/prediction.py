import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_model"

with open(BASE_DIR / "model" / "labels.json", "r", encoding="utf-8") as file:
    LABELS = json.load(file)

with open(BASE_DIR / "model" / "thresholds.json", "r", encoding="utf-8") as file:
    THRESHOLDS = json.load(file)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.to(DEVICE)
model.eval()

MAX_LENGTH = 256


def _validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a valid comment.")
    return text.strip()


def _probability_batch(texts):
    """Return six sigmoid probabilities for each input string."""
    texts = [_validate_text(text) for text in texts]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits
        probs = torch.sigmoid(logits)

    return probs.detach().cpu().numpy()


def predict_comment(text):
    """Predict all six multilabel toxicity categories for one comment."""
    text = _validate_text(text)
    probabilities = _probability_batch([text])[0]

    results = []
    for label, probability in zip(LABELS, probabilities):
        threshold = float(THRESHOLDS[label])
        results.append({
            "category": label,
            "probability": float(probability),
            "threshold": threshold,
            "detected": bool(probability >= threshold),
        })
    return results


def _make_category_predictor(category):
    """SHAP calls this function repeatedly with perturbed versions of a comment."""
    if category not in LABELS:
        raise ValueError(f"Unknown category: {category}")
    category_index = LABELS.index(category)

    def predict_for_shap(texts):
        # SHAP may supply a scalar string or a batch of strings.
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(text) for text in texts]
        return _probability_batch(texts)[:, category_index].reshape(-1, 1)

    return predict_for_shap


@lru_cache(maxsize=len(LABELS))
def _get_shap_explainer(category):
    """Create one cached SHAP text explainer per toxicity category."""
    import shap

    masker = shap.maskers.Text(tokenizer)
    predictor = _make_category_predictor(category)
    return shap.Explainer(
        predictor,
        masker,
        algorithm="partition",
        output_names=[category],
    )


def explain_comment(text, category, max_evals=120):
    """
    Explain one category's probability using SHAP token attributions.

    Positive SHAP values push the selected category probability upward;
    negative values push it downward. These are model attributions, not
    proof that a word is inherently harmful or causal.
    """
    text = _validate_text(text)
    if category not in LABELS:
        raise ValueError(f"Unknown category: {category}")

    explainer = _get_shap_explainer(category)
    explanation = explainer([text], max_evals=max_evals)

    # Text explanations generally have shape (1, token_count, 1).
    token_data = np.asarray(explanation.data[0], dtype=object).reshape(-1)
    shap_values = np.asarray(explanation.values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 0]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]
    else:
        shap_values = shap_values.reshape(-1)

    # Keep token/value pairs aligned if a SHAP version returns a special token.
    count = min(len(token_data), len(shap_values))
    tokens = [str(token) for token in token_data[:count]]
    values = [float(value) for value in shap_values[:count]]

    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    base_value = float(base_values[0]) if base_values.size else 0.0
    reconstructed = base_value + float(np.sum(values))

    return {
        "method": "SHAP Text (Partition)",
        "category": category,
        "base": base_value,
        "output": reconstructed,
        "tokens": tokens,
        "values": values,
    }
