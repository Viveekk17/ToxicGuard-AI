"""Generate the ToxicGuardAI project poster (16:9, three columns) from the real training results.

Usage (from the project root, with the virtual environment active):
    python poster/make_poster.py              # results + live XAI example from your trained model
    python poster/make_poster.py --no-xai     # skip loading the model (faster)

Reads   artifacts/metrics.json  (produced by the Colab notebook)
Writes  poster/ToxicGuardAI_Poster.html  -> open in Chrome/Edge -> Ctrl+P -> Save as PDF
        (Layout: Landscape · Margins: None · Background graphics: ON)
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ============================================================== EDIT THESE DETAILS
INSTITUTE = "MIT ACADEMY OF ENGINEERING"
INSTITUTE_SUB = ""          # optional second line, e.g. city
DEPARTMENT = "CSE(AIML)"
TEAM = [("SINGH VIVEK SUNILKUMAR", "202401110015"), ("VINAY DHAKE", "202401110037")]
MENTOR = "Prof. SHRUTI GADEWAR"
COURSE = "Natural Language Processing"
APP_URL = ""          # e.g. "toxicguardai.streamlit.app" (shown in the footer if set)
GITHUB_URL = ""       # e.g. "github.com/<user>/ToxicGuardAI"
# ==============================================================================

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_NAME = {"toxic": "Toxic", "severe_toxic": "Severe toxic", "obscene": "Obscene", "threat": "Threat",
              "insult": "Insult", "identity_hate": "Identity hate"}
LANG_NAME = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
NAVY, BLUE, ORANGE, GREEN, RED, GREY = "#0B2A5B", "#2563EB", "#F59E0B", "#16A34A", "#DC2626", "#94A3B8"

XAI_EXAMPLES = ["Tu pagal hai kya? Chup kar bewakoof, teri aukaat nahi hai."]
# To show an English example instead, use e.g.: ["You are a pathetic idiot and nobody wants you here."]


# ------------------------------------------------------------------ helpers
def esc(x) -> str:
    return html.escape(str(x))


def pct(x, d=1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def num(x, d=3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def get(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def section(title: str, body: str, cls: str = "") -> str:
    return f'<section class="box {cls}"><h2>{esc(title)}</h2><div class="bd">{body}</div></section>'


def table(headers: list[str], rows: list[list], highlight_last: bool = False, cls: str = "") -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = []
    for i, r in enumerate(rows):
        hl = ' class="hl"' if highlight_last and i == len(rows) - 1 else ""
        trs.append(f"<tr{hl}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def flow(steps: list[str], cls: str = "") -> str:
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<div class="step">{s}</div>')
        if i < len(steps) - 1:
            parts.append('<div class="arr">&#8594;</div>')
    return f'<div class="flow {cls}">{"".join(parts)}</div>'


# ------------------------------------------------------------------ SVG charts
def grouped_bars(groups: list[str], series: list[tuple[str, str, list]], w=560, h=170, y_min=0.0) -> str:
    """Vertical grouped bar chart. series = [(name, color, values)]."""
    pad_l, pad_b, pad_t = 34, 34, 26
    pw, ph = w - pad_l - 8, h - pad_b - pad_t
    n, k = len(groups), len(series)
    gw = pw / n
    bw = min(34, (gw - 16) / k)
    span = 1 - y_min
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" class="chart">']
    for t in [y_min + span * i / 4 for i in range(5)]:
        y = pad_t + ph - (t - y_min) / span * ph
        out.append(f'<line x1="{pad_l}" x2="{w-8}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E2E8F0"/>'
                   f'<text x="{pad_l-5}" y="{y+4:.1f}" text-anchor="end" class="ax">{t:.2f}</text>')
    for gi, g in enumerate(groups):
        x0 = pad_l + gi * gw + (gw - bw * k) / 2
        for si, (_, color, vals) in enumerate(series):
            v = vals[gi]
            if v is None:
                continue
            bh = max(0, (v - y_min) / span * ph)
            x, y = x0 + si * bw, pad_t + ph - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-3:.1f}" height="{bh:.1f}" rx="2" fill="{color}"/>'
                       f'<text x="{x + (bw-3)/2:.1f}" y="{y-3:.1f}" text-anchor="middle" class="val">{v:.2f}</text>')
        out.append(f'<text x="{pad_l + gi*gw + gw/2:.1f}" y="{h-pad_b+16}" text-anchor="middle" class="lab">{esc(g)}</text>')
    lx = pad_l
    for name, color, _ in series:
        out.append(f'<rect x="{lx}" y="4" width="11" height="11" rx="2" fill="{color}"/><text x="{lx+15}" y="13" class="lab">{esc(name)}</text>')
        lx += 22 + 7.2 * len(name)
    out.append("</svg>")
    return "".join(out)


def confusion_svg(cm: list[list[int]], w=250, h=190) -> str:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    mx = max(tn, fp, fn, tp) or 1
    cells = [(tn, 0, 0), (fp, 1, 0), (fn, 0, 1), (tp, 1, 1)]
    ox, oy, cs = 78, 34, 72
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" class="chart">',
           f'<text x="{ox+cs}" y="14" text-anchor="middle" class="lab">Predicted</text>',
           f'<text x="{ox+cs/2}" y="{oy-5}" text-anchor="middle" class="ax">Non-toxic</text>',
           f'<text x="{ox+1.5*cs}" y="{oy-5}" text-anchor="middle" class="ax">Toxic</text>',
           f'<text x="{ox-6}" y="{oy+cs/2+4}" text-anchor="end" class="ax">Non-toxic</text>',
           f'<text x="{ox-6}" y="{oy+1.5*cs+4}" text-anchor="end" class="ax">Toxic</text>',
           f'<text x="12" y="{oy+cs}" text-anchor="middle" class="lab" transform="rotate(-90 12 {oy+cs})">Actual</text>']
    for v, cx, cy in cells:
        a = 0.15 + 0.85 * v / mx
        txt = "#fff" if a > 0.55 else NAVY
        out.append(f'<rect x="{ox+cx*cs}" y="{oy+cy*cs}" width="{cs-2}" height="{cs-2}" rx="4" fill="rgba(37,99,235,{a:.2f})"/>'
                   f'<text x="{ox+cx*cs+cs/2-1}" y="{oy+cy*cs+cs/2+6}" text-anchor="middle" font-size="17" font-weight="700" fill="{txt}">{v:,}</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ live XAI example
def xai_examples(enabled: bool) -> str:
    if not enabled:
        return ""
    try:
        import tomllib

        from src.classifier import ToxicityClassifier
        from src.config import LABEL_DISPLAY, model_id
        from src.pipeline import Settings, analyze

        sec = ROOT / ".streamlit" / "secrets.toml"
        mid = model_id()
        if sec.exists() and "MODEL_ID" in (cfg := tomllib.loads(sec.read_text(encoding="utf-8"))):
            mid = cfg["MODEL_ID"]
        print(f"Loading {mid} for the XAI example…")
        clf = ToxicityClassifier(mid)
    except Exception as e:  # noqa: BLE001
        print(f"(XAI example skipped: {e})")
        return ""

    blocks = []
    for text in XAI_EXAMPLES:
        r = analyze(clf, text, Settings(use_lime=False))
        exp, t = r.explanation, r.explained_text
        if t != r.prep.model_text or exp.target_label != "toxic":
            # Poster: always explain the overall "toxic" score on the ORIGINAL (Hinglish/Hindi) words.
            from src.explainer import explain
            t = r.prep.model_text
            exp = explain(clf, t, "toxic", use_lime=False)
        words = exp.words
        m = max((abs(w.consensus) for w in words), default=1) or 1
        spans, cur = [], 0
        for w in words:
            spans.append(esc(t[cur:w.start]))
            a = min(abs(w.consensus) / m, 1) * 0.85
            rgb = "220,38,38" if w.consensus > 0 else "22,163,74"
            spans.append(f'<span style="background:rgba({rgb},{a:.2f});padding:1px 3px;border-radius:3px">{esc(t[w.start:w.end])}</span>')
            cur = w.end
        spans.append(esc(t[cur:]))
        cats = ", ".join(LABEL_DISPLAY[p] for p in r.positives) or "Non-toxic"
        tr = f'<div class="tr">Translation: {esc(r.translation)}</div>' if r.language.code != "en" else ""
        comp = exp.comprehensiveness
        blocks.append(f'<div class="xai"><div class="xh"><b>{esc(r.language.name)}</b> · P(toxic) {pct(r.probs["toxic"])} · '
                      f'{esc(cats)} · comprehensiveness {pct(comp) if comp is not None else "—"}</div>'
                      f'<div class="xt">{"".join(spans)}</div>{tr}</div>')
    return "".join(blocks)


# ------------------------------------------------------------------ build
def build(m: dict, xai_html: str) -> str:
    ov, bl, pl = m.get("overall", {}), m.get("baseline_tfidf", {}), m.get("per_label", {})
    plang, tr = m.get("per_language", {}), m.get("training", {})
    ds = m.get("dataset", {})

    # ---------------- column 1
    abstract = ("<p><b>ToxicGuardAI</b> detects, categorises and <b>explains</b> toxic comments in <b>English, Hindi and "
                "Hinglish</b>. A fine-tuned <b>multilingual BERT</b> predicts 6 toxicity categories, <b>XAI</b> shows which words "
                "drove each decision, and an <b>LLM</b> writes a plain-language explanation and a polite rewrite — in a deployed web app.</p>")
    problem = ("<ul><li>Manual moderation does not scale to millions of comments.</li>"
               "<li>Most detectors are <b>English-only</b>; Hindi and Hinglish slang is missed.</li>"
               "<li>Deep models are <b>black boxes</b> — no reason is given for a flag.</li>"
               "<li><b>Gap:</b> multilingual + fine-grained + <i>faithfully explained</i> detection.</li></ul>")
    objectives = ("<ol><li>Fine-tune mBERT for 6-category toxicity in English, Hindi, Hinglish.</li>"
                  "<li>Explain every decision at word level and measure faithfulness.</li>"
                  "<li>Add sentiment, information extraction, topics and cloud translation.</li>"
                  "<li>Integrate an LLM and deploy a working web app.</li></ol>")
    lit = table(["Approach", "Limitation"], [
        ["Keyword / lexicon filters", "Miss context, sarcasm, spelling variants"],
        ["TF-IDF + linear models", "No semantics; poor on code-mixed text"],
        ["BERT (Devlin et al., 2019)", "Strong, but English-centric and opaque"],
        ["Perspective API / Detoxify", "Black-box scores, weak on Hinglish"],
    ], cls="lit")

    srcs = ds.get("by_source_lang")
    if srcs:
        rows = [[("Jigsaw Toxic Comments" if r["source"] == "jigsaw" else "TextDetox"), LANG_NAME.get(r["lang"], r["lang"]),
                 f'{int(r["n"]):,}', pct(r["toxic_rate"], 0)] for r in srcs]
    else:
        rows = [["Jigsaw Toxic Comments", "English", "≈46,000", "≈35%"], ["TextDetox", "English", "5,000", "≈50%"],
                ["TextDetox", "Hindi", "5,000", "≈50%"], ["TextDetox", "Hinglish", "4,363", "≈50%"]]
    split = (f'Total <b>{ds["total"]:,}</b> · train <b>{ds["train"]:,}</b> / val <b>{ds["val"]:,}</b> / test <b>{ds["test"]:,}</b>'
             if ds.get("total") else "Split 76.5% train / 8.5% val / 15% test")
    dataset = (table(["Source", "Language", "Comments", "Toxic"], rows) +
               f'<p class="small">{split} (stratified). Jigsaw: 6 labels · TextDetox: binary label.</p>')
    prep = (flow(["Cleaning (HTML, URL, @user)", "Tokenisation", "Hinglish normalise", "Stop-words (EN+HI)", "TF-IDF + mBERT embeddings"], "wrap") +
            '<p class="small">e.g. <i>“stupiddddd!!!”</i> → <i>“stupidd!”</i> · <i>“nhi”</i> → <i>“nahi”</i></p>')

    col1 = (section("1. Project Overview", abstract) + section("2. Problem Statement", problem) +
            section("3. Objectives", objectives) + section("4. Literature Review", lit) +
            section("5. Dataset", dataset) + section("6. NLP Preprocessing", prep))

    # ---------------- column 2
    method = (flow(["Comment", "Language ID", "Preprocess", "Google Translate", "mBERT ×6 heads", "XAI", "Sentiment · IE · LDA",
                    "LLM explanation"]) +
              '<div class="two"><ul>'
              "<li><b>Classifier:</b> bert-base-multilingual-cased, 6 sigmoid outputs, <b>masked BCE</b> loss (unknown labels ignored) with pos_weight</li>"
              "<li><b>Thresholds:</b> tuned per category on validation F1</li>"
              "<li><b>Ensemble:</b> original + English translation (max if abusive term, else mean)</li></ul><ul>"
              "<li><b>XAI:</b> Integrated Gradients + Occlusion + LIME → consensus, ERASER faithfulness</li>"
              "<li><b>Sentiment:</b> VADER · <b>IE:</b> spaCy NER + gazetteer · <b>Topics:</b> LDA</li>"
              "<li><b>LLM:</b> zero / few-shot / XAI-grounded prompts, JSON output</li></ul></div>")
    arch = ('<div class="arch">'
            '<div class="lay"><div class="lt">Input</div><div class="li">Text box · CSV / Excel / TXT / PDF upload</div></div>'
            '<div class="arr">&#8594;</div>'
            '<div class="lay"><div class="lt">Processing</div><div class="li">Language ID · Cleaning · Tokenisation · Cloud translation</div></div>'
            '<div class="arr">&#8594;</div>'
            '<div class="lay main"><div class="lt">Model</div><div class="li">Fine-tuned mBERT (6 labels) + XAI engine · VADER · NER · LDA · LLM</div></div>'
            '<div class="arr">&#8594;</div>'
            '<div class="lay"><div class="lt">Output</div><div class="li">Verdict &amp; severity · Category scores · Word heat-map · Explanation · Rewrite</div></div>'
            "</div>")
    impl = ("<ul><li><b>Language:</b> Python 3.12</li><li><b>Models:</b> mBERT, TF-IDF + LR baseline, VADER, spaCy, LDA, LLM</li>"
            "<li><b>Frameworks:</b> PyTorch, HF Transformers, scikit-learn, Streamlit</li>"
            "<li><b>Tools:</b> Colab, Hugging Face Hub, GitHub, Streamlit Cloud</li></ul>")
    setup = (f'<ul><li><b>Hardware:</b> {esc(tr.get("gpu", "NVIDIA T4 GPU"))} (Colab) for training · CPU for inference</li>'
             f'<li><b>Training:</b> {tr.get("epochs", 2)} epochs · lr {tr.get("learning_rate", 2e-5)} · batch {tr.get("batch_size", 32)} · '
             f'max length {tr.get("max_len", 128)} · fp16'
             + (f' · {tr["train_runtime_min"]} min' if tr.get("train_runtime_min") else "") + "</li>"
             "<li><b>Metrics:</b> Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, faithfulness</li></ul>")

    comp_rows = [
        ["TF-IDF + Logistic Regression", pct(bl.get("accuracy")), pct(bl.get("precision")), pct(bl.get("recall")),
         pct(bl.get("f1_toxic")), num(bl.get("macro_f1")), num(bl.get("macro_auc"))],
        ["<b>Fine-tuned mBERT (ours)</b>", f'<b>{pct(ov.get("binary_accuracy"))}</b>', f'<b>{pct(ov.get("binary_precision"))}</b>',
         f'<b>{pct(ov.get("binary_recall"))}</b>', f'<b>{pct(ov.get("binary_f1"))}</b>', f'<b>{num(ov.get("macro_f1"))}</b>',
         f'<b>{num(ov.get("macro_auc"))}</b>'],
    ]
    comp = table(["Model", "Accuracy", "Precision", "Recall", "F1 (toxic)", "Macro F1 (6)", "Macro AUC (6)"], comp_rows, highlight_last=True)
    chart_groups = ["Accuracy", "Precision", "Recall", "F1", "Macro AUC"]
    chart = grouped_bars(chart_groups, [
        ("TF-IDF + LR", GREY, [bl.get("accuracy"), bl.get("precision"), bl.get("recall"), bl.get("f1_toxic"), bl.get("macro_auc")]),
        ("mBERT (ours)", BLUE, [ov.get("binary_accuracy"), ov.get("binary_precision"), ov.get("binary_recall"), ov.get("binary_f1"), ov.get("macro_auc")]),
    ], y_min=0.5)
    cm = m.get("confusion_matrix")
    mods = []
    if (s := m.get("sentiment_eval")):
        mods.append(["Sentiment (VADER)", esc(s.get("dataset", "")), f'Acc {pct(s["accuracy"])} · F1 {num(s["macro_f1"])}'])
    if (s := m.get("ie_eval")):
        mods.append(["Information extraction", f'{s.get("n", "")} gold sentences', f'F1 {num(s["f1"])} · target acc {pct(s.get("target_accuracy"))}'])
    if (s := m.get("language_id")):
        mods.append(["Language identification", "900 test comments", f'Acc {pct(s["accuracy"])}'])
    if (s := m.get("translation_ensemble")):
        mods.append(["Translation ensemble (max rule)", f'{s["n"]} Hindi/Hinglish', f'F1 {num(s["f1_without"])} → <b>{num(s["f1_with"])}</b>'])
    if (s := m.get("llm_eval")):
        best = max(s, key=lambda k: s[k]["f1"])
        mods.append(["LLM prompting", "zero / few-shot / XAI", f'best: {best.replace("_", "-")} F1 {num(s[best]["f1"])}'])
    modules = table(["Module", "Evaluated on", "Result"], mods, cls="tight") if mods else ""
    results = (comp + '<div class="charts"><div class="c1"><div class="ct">Model comparison (test set, toxic vs non-toxic)</div>' + chart +
               '</div><div class="c2"><div class="ct">Confusion matrix — mBERT</div>' + (f'<div style="width:235px;margin:auto">{confusion_svg(cm)}</div>' if cm else "") + "</div></div>" +
               ('<div class="ct">Other NLP modules — evaluation</div>' + modules if modules else ""))

    col2 = (section("7. Methodology", method, "emph") + section("8. System Architecture", arch) +
            '<div class="pair">' + section("9. Implementation", impl) + section("10. Experimental Setup", setup) + "</div>" +
            section("11. Results & Analysis", results, "emph grow"))

    # ---------------- column 3
    lab_rows = [[LABEL_NAME[k], pct(v.get("precision")), pct(v.get("recall")), pct(v.get("f1")), num(v.get("roc_auc"))]
                for k, v in pl.items()]
    per_label = table(["Category", "Precision", "Recall", "F1", "AUC"], lab_rows, cls="tight")
    lang_rows = [[LANG_NAME.get(k, k), f'{int(v.get("n", 0)):,}', pct(v.get("accuracy")), pct(v.get("f1")), num(v.get("roc_auc"))]
                 for k, v in plang.items()]
    per_lang = table(["Language", "Test n", "Accuracy", "F1", "AUC"], lang_rows, cls="tight")
    res2 = ('<div class="ct">Per-category performance (mBERT, tuned thresholds)</div>' + per_label +
            '<div class="ct">Per-language performance (binary toxic)</div>' + per_lang)
    if xai_html:
        res2 += ('<div class="ct">Explainability example (consensus attribution: '
                 '<span style="background:rgba(220,38,38,.6);padding:0 6px;border-radius:3px">toxic</span> '
                 '<span style="background:rgba(22,163,74,.6);padding:0 6px;border-radius:3px">non-toxic</span>)</div>' + xai_html)

    # highlights from real numbers
    hl = []
    if ov.get("macro_auc") is not None:
        hl.append(f'Macro ROC-AUC <b>{num(ov["macro_auc"])}</b> across 6 toxicity categories')
    if ov.get("macro_f1") is not None and bl.get("macro_f1") is not None:
        hl.append(f'6-category macro F1 <b>{num(ov["macro_f1"])}</b> vs {num(bl["macro_f1"])} for TF-IDF '
                  f'(<b>{(ov["macro_f1"] - bl["macro_f1"]) * 100:+.1f} pts</b>)')
    if ov.get("binary_f1") is not None and bl.get("f1_toxic") is not None:
        hl.append(f'Toxic F1 <b>{pct(ov["binary_f1"])}</b> · recall <b>{pct(ov.get("binary_recall"))}</b> '
                  f'(TF-IDF: {pct(bl["f1_toxic"])} · {pct(bl.get("recall"))})')
    if plang:
        hl.append("One model for 3 languages: F1 " + " · ".join(f'{LANG_NAME.get(k, k)} <b>{pct(v["f1"], 0)}</b>' for k, v in plang.items()))
    hl += ["Faithful word-level explanations from 3 XAI methods",
           "Deployed web app: batch upload, LLM explanations, polite rewrites"]
    highlights = "<ul class='chk'>" + "".join(f"<li>{h}</li>" for h in hl) + "</ul>"
    apps = ("<div class='two'><ul><li>Social-media &amp; comment moderation</li><li>Ed-tech &amp; gaming chat safety</li></ul>"
            "<ul><li>Cyber-bullying monitoring</li><li>“Rewrite before posting” assistant</li></ul></div>")
    stats = ", ".join(x for x in [
        f"macro ROC-AUC {num(ov['macro_auc'])}" if ov.get("macro_auc") is not None else "",
        f"toxic F1 {pct(ov['binary_f1'])}" if ov.get("binary_f1") is not None else ""] if x)
    gain = (f", with the largest gain on fine-grained categories (+{(ov['macro_f1'] - bl['macro_f1']) * 100:.1f} macro-F1 pts)"
            if ov.get("macro_f1") is not None and bl.get("macro_f1") is not None else "")
    weakest = min(plang.items(), key=lambda kv: kv[1].get("f1", 1)) if plang else None
    hard = (f" {LANG_NAME.get(weakest[0], weakest[0])} is the hardest case (F1 {pct(weakest[1]['f1'], 0)}) because of "
            "romanised spelling variation and limited training data." if weakest else "")
    concl = ("<p>A single fine-tuned multilingual BERT detects and categorises toxicity in English, Hindi and Hinglish"
             + (f" ({stats})" if stats else "") + f" and outperforms the TF-IDF baseline{gain}.{hard} "
             "XAI and an evidence-grounded LLM make every decision transparent.</p>"
             "<p class='small'><b>Future scope:</b> more Hinglish and Indic-language data, sarcasm detection, real-time moderation.</p>")
    refs = ("<div class='refs'>[1] Devlin et al., BERT, 2019 · [2] Sundararajan et al., Integrated Gradients, 2017 · "
            "[3] Ribeiro et al., LIME, 2016 · [4] DeYoung et al., ERASER, 2020 · [5] Jigsaw Toxic Comments; TextDetox datasets · "
            "[6] Hutto &amp; Gilbert, VADER, 2014</div>")

    col3 = (section("11. Results & Analysis (contd.)", res2, "emph") + section("12. Key Highlights", highlights) +
            section("13. Applications", apps) + section("14. Conclusion", concl) + section("References", refs, "refbox"))

    team = " &nbsp;|&nbsp; ".join(f"{esc(n)} ({esc(p)})" for n, p in TEAM)
    links = " &nbsp;·&nbsp; ".join(x for x in [APP_URL, GITHUB_URL] if x)
    footer = "ToxicGuardAI — explainable, multilingual toxic comment detection" + (f" &nbsp;·&nbsp; {esc(links)}" if links else "")

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ToxicGuardAI Poster</title>
<style>{CSS}</style></head><body><div class="poster">
<header><div class="inst"><div class="i1">{esc(INSTITUTE)}</div><div class="i2">{esc(INSTITUTE_SUB)}</div></div>
<div class="title"><h1>ToxicGuardAI: Explainable Multilingual Toxic Comment Detection</h1>
<div class="sub">using Multilingual BERT, Explainable AI and Generative AI · English · हिन्दी · Hinglish</div>
<div class="team">Team: {team}</div><div class="team">Faculty Mentor: {esc(MENTOR)} &nbsp;|&nbsp; Course: {esc(COURSE)}</div></div>
<div class="dept">{esc(DEPARTMENT)}</div></header>
<main><div class="col">{col1}</div><div class="col">{col2}</div><div class="col">{col3}</div></main>
<footer>{footer}</footer></div></body></html>"""


CSS = f"""
@page {{ size: 1920px 1080px; margin: 0; }}
* {{ box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
html, body {{ margin: 0; padding: 0; background: #fff; }}
body {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #0F172A; }}
.poster {{ width: 1920px; height: 1080px; display: flex; flex-direction: column; background: #F4F7FC; overflow: hidden; }}
header {{ height: 118px; background: {NAVY}; color: #fff; display: flex; align-items: center; padding: 0 26px; gap: 20px;
         border-bottom: 5px solid {ORANGE}; }}
.inst {{ width: 290px; }} .inst .i1 {{ font-size: 30px; font-weight: 800; letter-spacing: .5px; }}
.inst .i2 {{ font-size: 13px; opacity: .85; }}
.title {{ flex: 1; text-align: center; }}
.title h1 {{ margin: 0; font-size: 31px; font-weight: 800; letter-spacing: -.2px; }}
.title .sub {{ font-size: 17px; font-style: italic; opacity: .92; margin-top: 2px; }}
.title .team {{ font-size: 13.5px; opacity: .9; margin-top: 3px; }}
.dept {{ width: 290px; text-align: right; font-size: 24px; font-weight: 800; }}
main {{ flex: 1; display: grid; grid-template-columns: 26% 45% 29%; gap: 10px; padding: 10px 12px 8px; min-height: 0; }}
.col {{ display: flex; flex-direction: column; gap: 8px; min-height: 0; }}
.box {{ background: #fff; border: 1px solid #C9D6EE; border-radius: 8px; overflow: hidden; flex-shrink: 0; }}
.box h2 {{ margin: 0; background: {NAVY}; color: #fff; font-size: 15px; font-weight: 700; padding: 4px 10px; letter-spacing: .2px; }}
.box.emph h2 {{ background: linear-gradient(90deg, {NAVY}, {BLUE}); }}
.box.grow, .col > .box:last-child {{ flex: 1; }}
.bd {{ padding: 6px 10px; font-size: 13px; line-height: 1.38; }}
.bd p {{ margin: 0 0 4px; }} .bd ul, .bd ol {{ margin: 0; padding-left: 17px; }} .bd li {{ margin-bottom: 1px; }}
.small {{ font-size: 12px; color: #334155; margin-top: 4px !important; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.3px; margin: 2px 0 4px; }}
th {{ background: #E8EEFA; color: {NAVY}; font-weight: 700; text-align: left; padding: 3px 6px; border-bottom: 1px solid #C9D6EE; }}
td {{ padding: 2.5px 6px; border-bottom: 1px solid #EEF2F7; }}
tr.hl td {{ background: #E8F5EC; }}
table.tight td, table.tight th {{ padding: 2px 6px; }}
table.lit td:first-child {{ font-weight: 600; white-space: nowrap; }}
.flow {{ display: flex; align-items: center; gap: 4px; margin: 3px 0 6px; flex-wrap: nowrap; }}
.flow.wrap {{ flex-wrap: wrap; row-gap: 5px; }}
.step {{ background: #EAF1FF; border: 1px solid #9DB8F2; color: {NAVY}; border-radius: 6px; padding: 4px 7px; font-size: 12px;
         font-weight: 600; text-align: center; line-height: 1.2; }}
.arr {{ color: {BLUE}; font-weight: 800; font-size: 15px; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.arch {{ display: flex; align-items: stretch; gap: 5px; }}
.arch .arr {{ align-self: center; }}
.lay {{ flex: 1; border: 1.5px solid #9DB8F2; border-radius: 8px; background: #F7FAFF; padding: 5px 7px; }}
.lay.main {{ flex: 1.35; background: #EAF1FF; border-color: {BLUE}; }}
.lt {{ font-weight: 800; color: {NAVY}; font-size: 13px; }} .li {{ font-size: 12px; color: #334155; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.charts {{ display: grid; grid-template-columns: 1.75fr 1fr; gap: 10px; margin-top: 4px; }}
.ct {{ font-weight: 700; color: {NAVY}; font-size: 12.5px; margin: 4px 0 1px; }}
.chart text.ax {{ font-size: 10.5px; fill: #475569; }} .chart text.val {{ font-size: 10px; fill: #0F172A; font-weight: 600; }}
.chart text.lab {{ font-size: 11.5px; fill: #0F172A; }}
.xai {{ border: 1px dashed #9DB8F2; border-radius: 6px; padding: 4px 7px; margin-bottom: 4px; }}
.xh {{ font-size: 11.5px; color: #334155; }} .xt {{ font-size: 13.5px; line-height: 1.7; }} .tr {{ font-size: 11.5px; color: #475569; font-style: italic; }}
ul.chk {{ list-style: none; padding-left: 0; }} ul.chk li::before {{ content: "✔ "; color: {GREEN}; font-weight: 800; }}
.refs {{ font-size: 11px; line-height: 1.35; }} .refbox .bd {{ padding: 4px 10px; }}
footer {{ height: 30px; background: {NAVY}; color: #fff; font-size: 13.5px; font-style: italic; display: flex; align-items: center;
         justify-content: center; letter-spacing: .3px; }}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(ROOT / "artifacts" / "metrics.json"))
    ap.add_argument("--out", default=str(ROOT / "poster" / "ToxicGuardAI_Poster.html"))
    ap.add_argument("--no-xai", action="store_true", help="skip the live explainability example")
    a = ap.parse_args()
    mp = Path(a.metrics)
    if not mp.exists():
        sys.exit(f"{mp} not found - run the Colab notebook and unzip artifacts.zip into artifacts/ first.")
    metrics = json.loads(mp.read_text(encoding="utf-8"))
    out = Path(a.out)
    out.write_text(build(metrics, xai_examples(not a.no_xai)), encoding="utf-8")
    print(f"Poster written to {out}\nOpen it in Chrome/Edge -> Ctrl+P -> Destination: Save as PDF, Margins: None, "
          "Background graphics: ON.")


if __name__ == "__main__":
    main()
