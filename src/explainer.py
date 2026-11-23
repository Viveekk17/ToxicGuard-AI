"""Explainable AI (XAI) for the BERT toxicity classifier.

Three complementary, model-faithful attribution methods are computed at *word* level:

* **Integrated Gradients** (Sundararajan et al., 2017) - gradient-based, satisfies the
  completeness axiom (sum of attributions == f(x) - f(baseline)); we report the residual.
* **Occlusion / leave-one-out** - perturbation-based: drop in probability when a word is removed.
* **LIME** (Ribeiro et al., 2016) - local surrogate linear model on random word masks.

They are fused into a **consensus** ranking, and the explanation is quantitatively checked with
the ERASER faithfulness metrics (DeYoung et al., 2020): *comprehensiveness* and *sufficiency*.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from .classifier import MAX_LEN, ToxicityClassifier
from .config import LABEL_DISPLAY, LABELS
from .lexicon import ABUSE_LEXICON

WORD_RE = re.compile(r"\S+")
STRIP_PUNCT = re.compile(r"^[^\wऀ-ॿ]+|[^\wऀ-ॿ]+$")


@dataclass
class WordAttribution:
    word: str
    start: int
    end: int
    ig: float = 0.0
    occlusion: float = 0.0
    lime: float | None = None
    consensus: float = 0.0
    lexicon_category: str | None = None


@dataclass
class Explanation:
    target_label: str
    target_prob: float
    words: list[WordAttribution]
    ig_delta: float | None = None             # completeness residual of IG
    comprehensiveness: float | None = None
    sufficiency: float | None = None
    top_k: int = 0
    methods: list[str] = field(default_factory=list)

    def top_words(self, n: int = 8, positive_only: bool = True) -> list[WordAttribution]:
        ws = sorted(self.words, key=lambda w: -w.consensus)
        if positive_only:
            ws = [w for w in ws if w.consensus > 0.05]
        return ws[:n]


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def _normalize(scores: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(scores)) if len(scores) else 0
    return scores / m if m > 1e-12 else scores


def _bare(word: str) -> str:
    return STRIP_PUNCT.sub("", word).lower()


# ------------------------------------------------------------------ Integrated Gradients
def integrated_gradients(clf: ToxicityClassifier, text: str, label: str, steps: int = 24, chunk: int = 8):
    """Word-level Integrated Gradients w.r.t. the target label's logit. Returns (scores, delta)."""
    tok, model = clf.tokenizer, clf.model
    label_idx = clf.label_index[LABELS.index(label)]
    words = _word_spans(text)

    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"]
    extra = {k: v for k, v in enc.items() if k not in ("input_ids",)}

    special = torch.tensor(
        tok.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True), dtype=torch.bool
    )
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    base_ids = input_ids.clone()
    base_ids[0, ~special] = pad_id

    emb = model.get_input_embeddings()
    with torch.no_grad():
        x = emb(input_ids)
        b = emb(base_ids)

    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps   # midpoint Riemann sum
    total_grad = torch.zeros_like(x[0])
    for i in range(0, steps, chunk):
        a = alphas[i : i + chunk]
        k = len(a)
        interp = (b + a[:, None, None] * (x - b)).detach().requires_grad_(True)
        kw = {key: val.expand(k, -1) for key, val in extra.items()}
        logits = model(inputs_embeds=interp, **kw).logits[:, label_idx]
        grad = torch.autograd.grad(logits.sum(), interp)[0]
        total_grad += grad.sum(0)
    attr = ((x[0] - b[0]) * (total_grad / steps)).sum(-1).detach()

    with torch.no_grad():
        fx = model(inputs_embeds=x, **extra).logits[0, label_idx]
        fb = model(inputs_embeds=b, **extra).logits[0, label_idx]
    delta = float(attr.sum() - (fx - fb))

    scores = np.zeros(len(words))
    for t_i, (s, e) in enumerate(offsets):
        if special[t_i] or e <= s:
            continue
        for w_i, (_, ws, we) in enumerate(words):
            if ws <= s < we:
                scores[w_i] += float(attr[t_i])
                break
    return scores, delta


# ------------------------------------------------------------------ Occlusion
def occlusion(clf: ToxicityClassifier, text: str, label: str, max_words: int = 80) -> np.ndarray:
    words = _word_spans(text)[:max_words]
    li = LABELS.index(label)
    base = clf.predict_proba([text])[0, li]
    variants = [" ".join(w for j, (w, _, _) in enumerate(words) if j != i) for i in range(len(words))]
    if not variants:
        return np.zeros(0)
    probs = clf.predict_proba(variants)[:, li]
    scores = base - probs
    full = np.zeros(len(_word_spans(text)))
    full[: len(scores)] = scores
    return full


# ------------------------------------------------------------------ LIME
def lime_scores(clf: ToxicityClassifier, text: str, label: str, num_samples: int = 200) -> np.ndarray | None:
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception:
        return None
    li = LABELS.index(label)
    words = _word_spans(text)

    def fn(samples):
        p = clf.predict_proba(list(samples), batch_size=32)[:, li]
        return np.column_stack([1 - p, p])

    explainer = LimeTextExplainer(class_names=["clean", label], split_expression=r"\s+", bow=False, random_state=42)
    exp = explainer.explain_instance(text, fn, num_features=len(words), num_samples=num_samples, labels=(1,))
    # With bow=False LIME features are positional: map by index in the whitespace split.
    weights = dict(exp.as_map()[1])
    return np.array([weights.get(i, 0.0) for i in range(len(words))])


# ------------------------------------------------------------------ Faithfulness (ERASER)
def faithfulness(clf: ToxicityClassifier, text: str, label: str, ranked_idx: list[int], k: int) -> tuple[float, float]:
    words = [w for w, _, _ in _word_spans(text)]
    li = LABELS.index(label)
    top = set(ranked_idx[:k])
    without = " ".join(w for i, w in enumerate(words) if i not in top) or "."
    only = " ".join(w for i, w in enumerate(words) if i in top) or "."
    p_full, p_without, p_only = clf.predict_proba([text, without, only])[:, li]
    return float(p_full - p_without), float(p_full - p_only)


# ------------------------------------------------------------------ main entry
def explain(
    clf: ToxicityClassifier,
    text: str,
    label: str,
    use_ig: bool = True,
    use_occlusion: bool = True,
    use_lime: bool = False,
    ig_steps: int = 24,
    lime_samples: int = 200,
) -> Explanation:
    spans = _word_spans(text)
    words = [WordAttribution(w, s, e, lexicon_category=ABUSE_LEXICON.get(_bare(w))) for w, s, e in spans]
    prob = float(clf.predict_proba([text])[0, LABELS.index(label)])
    exp = Explanation(target_label=label, target_prob=prob, words=words)
    if not words:
        return exp

    normed = []
    if use_ig:
        ig, delta = integrated_gradients(clf, text, label, steps=ig_steps)
        exp.ig_delta = delta
        exp.methods.append("Integrated Gradients")
        for w, v in zip(words, ig):
            w.ig = float(v)
        normed.append(_normalize(ig))
    if use_occlusion:
        occ = occlusion(clf, text, label)
        exp.methods.append("Occlusion")
        for w, v in zip(words, occ):
            w.occlusion = float(v)
        normed.append(_normalize(occ))
    if use_lime:
        lm = lime_scores(clf, text, label, num_samples=lime_samples)
        if lm is not None:
            exp.methods.append("LIME")
            for w, v in zip(words, lm):
                w.lime = float(v)
            normed.append(_normalize(lm))

    if normed:
        cons = np.mean(np.vstack(normed), axis=0)
        for w, v in zip(words, cons):
            w.consensus = float(v)

    ranked = list(np.argsort([-w.consensus for w in words]))
    k = max(1, min(3, int(np.ceil(0.2 * len(words)))))
    exp.top_k = k
    exp.comprehensiveness, exp.sufficiency = faithfulness(clf, text, label, ranked, k)
    return exp


def natural_language_rationale(exp: Explanation, probs: dict, positives: list[str]) -> list[str]:
    """Deterministic, evidence-based rationale sentences built only from XAI outputs."""
    lines = []
    label = LABEL_DISPLAY[exp.target_label]
    top = exp.top_words(5)
    if not positives:
        lines.append(f"No category crossed its decision threshold; the highest score is **{label}** at {probs.get(exp.target_label, exp.target_prob):.1%}.")
        if top:
            lines.append("Words that pushed the score up the most (still not enough to flag): "
                         + ", ".join(f"**{w.word}**" for w in top[:3]) + ".")
        else:
            lines.append("No word produced a meaningful toxic signal - the comment reads as non-toxic to the model.")
        return lines

    lines.append(f"The model flags the comment as **{', '.join(LABEL_DISPLAY[p] for p in positives)}**; "
                 f"the explanation targets **{label}** ({probs.get(exp.target_label, exp.target_prob):.1%}).")
    if top:
        total = sum(max(w.consensus, 0) for w in exp.words) or 1
        main = top[0]
        lines.append(f"The single most influential word is **“{main.word}”**, carrying about "
                     f"{max(main.consensus, 0) / total:.0%} of the positive evidence"
                     + (f"; it is a known *{LABEL_DISPLAY[main.lexicon_category]}* term in our multilingual lexicon." if main.lexicon_category else "."))
        if len(top) > 1:
            lines.append("Other contributing words: " + ", ".join(f"**{w.word}**" for w in top[1:]) + ".")
    if exp.comprehensiveness is not None:
        drop = exp.comprehensiveness * 100
        lines.append(f"Without the top {exp.top_k} word(s) the {label} score changes by **{drop:+.0f} points**, so "
                     + ("these words clearly drive the decision." if drop >= 30 else
                        "the decision depends on the whole sentence, not just a few words." if drop < 10 else
                        "these words matter, together with the wider context."))
    neg = [w for w in sorted(exp.words, key=lambda w: w.consensus) if w.consensus < -0.15][:3]
    if neg:
        lines.append("Words that *reduced* the toxicity score: " + ", ".join(f"*{w.word}*" for w in neg) + ".")
    return lines
