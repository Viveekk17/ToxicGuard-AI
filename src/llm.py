"""CO4 - Generative AI / LLM layer (Google Gemini or Anthropic Claude) with explicit prompt engineering.

Prompting strategies offered in the UI ("Prompt Lab"):
  * zero_shot     - role + task + output schema only
  * few_shot      - adds labelled English / Hindi / Hinglish demonstrations
  * xai_grounded  - few-shot + the BERT scores and XAI attributions injected as evidence
                    (the LLM must justify, agree or disagree with the classifier) [default]

Every strategy uses structured outputs (JSON schema) so the response is machine-readable.
Without an API key, an offline template explanation is produced so the app always works.
"""
from __future__ import annotations

import json
import re

from .config import CLAUDE_MODEL, LABEL_DISPLAY, LABELS, anthropic_api_key, gemini_api_key, gemini_model, GEMINI_FALLBACK_MODELS
from .lexicon import ABUSE_LEXICON

CATEGORY_ENUM = LABELS + ["none"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["toxic", "borderline", "non_toxic"]},
        "categories": {"type": "array", "items": {"type": "string", "enum": CATEGORY_ENUM}},
        "severity": {"type": "string", "enum": ["none", "low", "moderate", "high", "severe"]},
        "agrees_with_classifier": {"type": "boolean"},
        "target_type": {"type": "string", "enum": ["individual", "group", "self", "none", "unclear"]},
        "target_description": {"type": "string"},
        "intent": {"type": "string"},
        "explanation": {"type": "string"},
        "key_phrases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"phrase": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["phrase", "reason"],
                "additionalProperties": False,
            },
        },
        "cultural_context": {"type": "string"},
        "detoxified_rewrite": {"type": "string"},
        "moderation_action": {"type": "string", "enum": ["allow", "warn", "hide", "remove", "escalate"]},
    },
    "required": ["verdict", "categories", "severity", "agrees_with_classifier", "target_type",
                 "target_description", "intent", "explanation", "key_phrases", "cultural_context",
                 "detoxified_rewrite", "moderation_action"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are ToxicGuardAI's senior content-moderation analyst. You review user comments \
written in English, Hindi (Devanagari) or Hinglish (romanised, code-mixed Hindi-English) and explain \
toxicity decisions to students, moderators and the comment's author.

Category definitions (Jigsaw taxonomy):
- toxic: rude, disrespectful or unreasonable language
- severe_toxic: extremely hateful, aggressive or vulgar language
- obscene: profanity, sexual or vulgar content
- threat: intent or wish to harm someone
- insult: demeaning language aimed at a person
- identity_hate: hate aimed at a protected identity (religion, caste, gender, race, orientation, nationality, disability)

How to work:
1. Read the comment in its original language first; account for Hindi/Hinglish slang, sarcasm, \
reclaimed or friendly banter ("pagal hai kya yaar" between friends is usually mild), and quoted speech.
2. Decide the verdict independently, then compare with any classifier evidence you are given and \
set agrees_with_classifier honestly. Disagree when the context justifies it.
3. key_phrases must be exact substrings of the original comment.
4. explanation: 2-4 plain sentences a non-expert can follow; reference the key phrases.
5. cultural_context: explain any Hindi/Hinglish slang in English; write "Not applicable" for plain English.
6. detoxified_rewrite: keep the author's underlying point and language (Hindi stays Hindi, Hinglish \
stays Hinglish) but remove every insult, slur, profanity and threat. For a non-toxic comment return it unchanged.
7. Use "none" alone in categories when the comment is non-toxic."""

FEW_SHOT = """Examples of correct analyses:

Comment: "You are an absolute idiot, nobody wants you here."
-> verdict toxic; categories [toxic, insult]; severity moderate; target individual; key_phrases \
["absolute idiot", "nobody wants you here"]; rewrite "I strongly disagree with you, and I think this \
contribution doesn't fit here."; action hide.

Comment: "Tu pagal hai kya? Dimag kharab hai tera, nikal yahan se"
-> verdict toxic; categories [toxic, insult]; severity moderate; target individual; cultural_context \
"'pagal' = crazy, 'dimag kharab' = your mind is broken, 'nikal yahan se' = get out of here"; rewrite \
"Tum aisa kyun soch rahe ho? Mujhe tumhari baat samajh nahi aayi."; action hide.

Comment: "इस फिल्म की कहानी बहुत अच्छी थी, सबको देखनी चाहिए"
-> verdict non_toxic; categories [none]; severity none; target none; rewrite unchanged; action allow.

Comment: "Main tujhe jaan se maar dunga"
-> verdict toxic; categories [toxic, threat]; severity severe; target individual; cultural_context \
"'jaan se maar dunga' = I will kill you"; action escalate."""


def build_prompt(strategy: str, comment: str, language: str, translation: str | None,
                 probs: dict | None = None, top_words: list[tuple[str, float]] | None = None) -> tuple[str, str]:
    system = SYSTEM_PROMPT
    if strategy in ("few_shot", "xai_grounded"):
        system = SYSTEM_PROMPT + "\n\n" + FEW_SHOT

    parts = [f"<comment language=\"{language}\">\n{comment}\n</comment>"]
    if translation and translation.strip() and translation.strip() != comment.strip():
        parts.append(f"<machine_translation source=\"Google Translate\">\n{translation}\n</machine_translation>")
    if strategy == "xai_grounded" and probs:
        score_lines = "\n".join(f"- {LABEL_DISPLAY[k]}: {v:.1%}" for k, v in probs.items())
        parts.append(f"<classifier_evidence model=\"fine-tuned multilingual BERT\">\nCategory probabilities:\n{score_lines}")
        if top_words:
            parts.append("Most influential words (Integrated Gradients + Occlusion consensus, 1.0 = strongest):\n"
                         + "\n".join(f"- \"{w}\": {s:+.2f}" for w, s in top_words))
        parts.append("</classifier_evidence>")
    parts.append("Analyse the comment and return the JSON object.")
    return system, "\n\n".join(parts)


def llm_provider() -> str | None:
    """Which LLM to use: Gemini if GEMINI_API_KEY is set, else Claude if ANTHROPIC_API_KEY is set, else None."""
    if gemini_api_key():
        return "gemini"
    if anthropic_api_key():
        return "claude"
    return None


def llm_available() -> bool:
    return llm_provider() is not None


def llm_label() -> str:
    """Human-readable name of the active LLM, for the UI."""
    p = llm_provider()
    return {"gemini": f"Google Gemini ({gemini_model()})", "claude": f"Anthropic Claude ({CLAUDE_MODEL})"}.get(p, "offline template")


def analyze_with_llm(strategy: str, comment: str, language: str, translation: str | None,
                     probs: dict | None = None, top_words: list[tuple[str, float]] | None = None) -> dict:
    """Call the configured LLM with structured (JSON-schema) output. Returns the parsed JSON + metadata."""
    if llm_provider() == "gemini":
        return _analyze_with_gemini(strategy, comment, language, translation, probs, top_words)
    return _analyze_with_claude(strategy, comment, language, translation, probs, top_words)


def _analyze_with_gemini(strategy, comment, language, translation, probs=None, top_words=None) -> dict:
    from google import genai
    from google.genai import errors, types

    system, user = build_prompt(strategy, comment, language, translation, probs, top_words)
    client = genai.Client(api_key=gemini_api_key(), http_options=types.HttpOptions(timeout=25_000))  # 25 s per call
    candidates = [gemini_model()] + [m for m in GEMINI_FALLBACK_MODELS if m != gemini_model()]

    def call(config):
        nonlocal model
        last = None
        import time

        deadline = time.time() + 60   # overall budget so the UI never hangs during a demo
        for m in candidates:          # configured model first; fall back if missing (404), busy or timed out (5xx)
            if time.time() > deadline:
                break
            model = m
            try:
                return client.models.generate_content(model=m, contents=user, config=config)
            except errors.ClientError as e:
                if e.code != 404:
                    raise
                last = e
            except errors.ServerError as e:
                last = e
            except Exception as e:  # client-side timeout / network
                last = e
        raise last

    model = candidates[0]
    try:
        try:
            resp = call(types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                                    response_json_schema=OUTPUT_SCHEMA))
        except errors.ClientError as e:
            if e.code != 400:
                raise
            # Older models may reject the schema: fall back to JSON mode with the schema in the prompt.
            resp = client.models.generate_content(
                model=model, contents=user + "\n\nReturn ONLY a JSON object matching this JSON schema:\n" + json.dumps(OUTPUT_SCHEMA),
                config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json"))
    except errors.ClientError as e:
        if e.code in (401, 403) or "API key" in str(e):
            return {"error": "Invalid GEMINI_API_KEY."}
        if e.code == 429:
            return {"error": "Gemini free-tier rate limit reached - wait a minute and try again."}
        if e.code == 404:
            return {"error": f"Gemini model '{model}' not found - set GEMINI_MODEL in secrets to a current model."}
        return {"error": f"Gemini API error ({e.code}): {e.message}"}
    except errors.ServerError as e:
        return {"error": f"Gemini is busy right now (HTTP {e.code}) - showing the offline explanation; try again in a minute."}
    except Exception as e:  # network etc.
        return {"error": f"Gemini did not respond in time ({type(e).__name__}) - showing the offline explanation; try again in a minute."}

    text = resp.text or ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Could not parse Gemini output (the response may have been blocked).", "raw": text}
    missing = [k for k in OUTPUT_SCHEMA["required"] if k not in data]
    if missing:
        return {"error": f"Gemini output is missing fields: {', '.join(missing)}", "raw": text}
    usage = getattr(resp, "usage_metadata", None)
    data["_meta"] = {
        "model": getattr(resp, "model_version", None) or model,
        "provider": "Gemini",
        "strategy": strategy,
        "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        "system_prompt": system,
        "user_prompt": user,
        "source": "llm",
    }
    return data


def _analyze_with_claude(strategy, comment, language, translation, probs=None, top_words=None) -> dict:
    import anthropic

    system, user = build_prompt(strategy, comment, language, translation, probs, top_words)
    client = anthropic.Anthropic(api_key=anthropic_api_key(), timeout=90.0, max_retries=2)
    try:
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        return {"error": "Invalid ANTHROPIC_API_KEY."}
    except anthropic.RateLimitError:
        return {"error": "Claude API rate limit reached - try again in a minute."}
    except anthropic.APIStatusError as e:
        return {"error": f"Claude API error ({e.status_code}): {e.message}"}
    except anthropic.APIConnectionError:
        return {"error": "Could not reach the Claude API (network)."}

    if response.stop_reason == "refusal":
        return {"error": "The model declined to analyse this comment."}
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Could not parse model output.", "raw": text}
    data["_meta"] = {
        "model": response.model,
        "strategy": strategy,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "system_prompt": system,
        "user_prompt": user,
        "provider": "Claude",
        "source": "llm",
    }
    return data


# ------------------------------------------------------------------ offline fallback
def offline_analysis(comment: str, positives: list[str], probs: dict, top_words: list[tuple[str, float]],
                     target: str, language: str) -> dict:
    """Template-based explanation + rule-based rewrite used when no API key is configured."""
    max_p = max(probs.values()) if probs else 0
    verdict = "toxic" if positives else "borderline" if max_p > 0.3 else "non_toxic"
    severity = ("severe" if ("severe_toxic" in positives or "threat" in positives) else
                "high" if max_p > 0.85 else "moderate" if positives else "low" if verdict == "borderline" else "none")
    phrases = [{"phrase": w, "reason": f"High attribution ({s:+.2f}) in the BERT explanation"
                + (f"; lexicon category: {LABEL_DISPLAY[ABUSE_LEXICON[w.lower().strip('.,!?')]]}" if w.lower().strip(".,!?") in ABUSE_LEXICON else "")}
               for w, s in top_words[:4] if s > 0.1]
    if verdict == "non_toxic":
        expl = "The classifier found no category above its threshold and no word carried strong toxic evidence."
    else:
        cats = ", ".join(LABEL_DISPLAY[p] for p in positives) or "borderline toxicity"
        expl = (f"The comment was classified as {cats}. The decision is driven mainly by "
                + ", ".join(f"'{p['phrase']}'" for p in phrases[:3]) + "." if phrases else
                f"The comment was classified as {cats} based on its overall context.")

    rewrite = comment
    if verdict != "non_toxic":
        for w, s in top_words:
            bare = w.lower().strip(".,!?")
            if s > 0.5 or bare in ABUSE_LEXICON:
                rewrite = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", "", rewrite)
        rewrite = re.sub(r"\s+", " ", rewrite).strip() or "[comment removed]"

    return {
        "verdict": verdict,
        "categories": positives or ["none"],
        "severity": severity,
        "agrees_with_classifier": True,
        "target_type": "individual" if target.startswith("Individual") else "group" if target.startswith("Group") else "none",
        "target_description": target,
        "intent": "Derived from classifier evidence (offline mode).",
        "explanation": expl,
        "key_phrases": phrases,
        "cultural_context": "Not available in offline mode." if language != "en" else "Not applicable",
        "detoxified_rewrite": rewrite,
        "moderation_action": {"severe": "escalate", "high": "remove", "moderate": "hide", "low": "warn"}.get(severity, "allow"),
        "_meta": {"source": "offline", "strategy": "template"},
    }
