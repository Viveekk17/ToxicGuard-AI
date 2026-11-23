"""ToxicGuardAI - Explainable multilingual toxic comment detection (Streamlit app).

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import ui
from src.analysis import fit_batch_topics
from src.classifier import ToxicityClassifier
from src.config import APP_NAME, ARTIFACTS_DIR, FALLBACK_MODEL_ID, LABEL_DISPLAY, LABELS, hf_token, model_id
from src.examples import BENCHMARK, SAMPLE_COMMENTS
from src.llm import analyze_with_llm, build_prompt, llm_available, llm_label, offline_analysis
from src.multilingual import LANG_NAMES
from src.pipeline import Settings, analyze, quick_classify
from src.preprocessing import extract_text_from_upload, preprocess

st.set_page_config(page_title=f"{APP_NAME} · Explainable toxicity detection", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)

STRATEGY_NAMES = {"xai_grounded": "XAI-grounded", "few_shot": "Few-shot", "zero_shot": "Zero-shot"}
STRATEGY_HELP = {"xai_grounded": "Gives the LLM the BERT scores and influential words as evidence (recommended).",
                 "few_shot": "Adds worked examples in English, Hindi and Hinglish.",
                 "zero_shot": "Instructions only, no examples or evidence."}


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def short_lang(name: str) -> str:
    return name.replace(" (code-mixed)", "")


# ====================================================================== resources
@st.cache_resource(show_spinner="Loading the BERT model — first start downloads the weights…")
def get_classifier(mid: str) -> ToxicityClassifier:
    return ToxicityClassifier(mid, token=hf_token())


@st.cache_data
def load_metrics() -> dict | None:
    p = ARTIFACTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


MODEL_ID = model_id()
try:
    clf = get_classifier(MODEL_ID)
except Exception as e:  # pragma: no cover - surfaced in UI
    st.error(f"Could not load model `{MODEL_ID}`: {e}")
    if MODEL_ID == FALLBACK_MODEL_ID:
        st.stop()
    MODEL_ID = FALLBACK_MODEL_ID
    clf = get_classifier(MODEL_ID)
IS_FINETUNED = MODEL_ID != FALLBACK_MODEL_ID

st.session_state.setdefault("comment", "")
st.session_state.setdefault("opt_translate", True)
st.session_state.setdefault("opt_lime", False)

# ====================================================================== brand bar + navigation
html(ui.brand_bar(APP_NAME, "Explainable multilingual toxic-comment detection",
                  IS_FINETUNED, "Fine-tuned mBERT" if IS_FINETUNED else "Fallback BERT",
                  llm_available(), llm_label().split(" (")[0] if llm_available() else "LLM offline"))

tab_analyze, tab_batch, tab_perf, tab_about = st.tabs(
    [":material/search: Analyze", ":material/upload_file: Batch analysis", ":material/insights: Model performance",
     ":material/school: Project & rubric"])


# ====================================================================== helpers
def current_settings() -> Settings:
    return Settings(translation_ensemble=st.session_state["opt_translate"], use_ig=True, use_occlusion=True,
                    use_lime=st.session_state["opt_lime"])


def run_llm(res, strategy):
    top = [(w.word, w.consensus) for w in res.explanation.top_words(8, positive_only=False)]
    if llm_available():
        out = analyze_with_llm(strategy, res.text, res.language.name, res.translation, res.probs, top)
        if "error" not in out:
            if out.get("detoxified_rewrite"):  # counterfactual check: re-score the rewrite with BERT
                out["_rewrite_probs"] = dict(zip(LABELS, map(float, clf.predict_proba(
                    [preprocess(out["detoxified_rewrite"]).model_text])[0])))
            return out
        st.toast(out["error"], icon="⚠️")
    out = offline_analysis(res.text, res.positives, res.probs, top, res.info["target"], res.language.code)
    if out.get("detoxified_rewrite"):
        out["_rewrite_probs"] = dict(zip(LABELS, map(float, clf.predict_proba([out["detoxified_rewrite"]])[0])))
    return out


def _fill_sample(lang: str):
    import random

    pool = [c for c in SAMPLE_COMMENTS[lang] if c != st.session_state.get("comment")]
    st.session_state["comment"] = random.choice(pool)


def lead_sentence(res) -> str:
    top = res.explanation.top_words(1)
    if res.is_toxic:
        word = f" The strongest signal is **“{top[0].word.strip('.,!?')}”**." if top else ""
        return f"The model detected **{len(res.positives)} of 6** toxicity categories.{word}"
    top_lab = max(res.probs, key=res.probs.get)
    return (f"No category crossed its decision threshold. The highest score is **{LABEL_DISPLAY[top_lab]}** "
            f"at **{res.probs[top_lab]:.0%}**.")


def trust_text(exp) -> tuple[str, str]:
    if exp.comprehensiveness is None:
        return "–", ""
    drop = exp.comprehensiveness * 100
    verdict = ("These words clearly drive the decision." if drop >= 30 else
               "These words matter, together with the wider context." if drop >= 10 else
               "The decision depends on the whole sentence, not a few words.")
    return f"{drop:+.0f} pts", (f"Removing the top {exp.top_k} highlighted word(s) changes the score by "
                                f"<b>{drop:+.0f} points</b>. {verdict}")


# ====================================================================== ANALYZE
with tab_analyze:
    html(ui.heading("Analyze a comment", "The model detects toxicity and its categories automatically"))
    text = st.text_area("Comment", key="comment", height=118, label_visibility="collapsed",
                        placeholder="Type or paste a comment in English, हिन्दी or Hinglish…")
    b1, b2, b3, b4, b5, sp, b6 = st.columns([1.5, 1, 0.9, 1.05, 0.85, 1.6, 0.95])
    go_btn = b1.button("Analyze", type="primary", width="stretch", icon=":material/shield:")
    for col, lang in zip((b2, b3, b4), ("English", "Hindi", "Hinglish")):
        col.button(lang, key=f"sample_{lang}", on_click=_fill_sample, args=(lang,), width="stretch", icon=":material/casino:",
                   help=f"Insert a random {lang} sample — clean or toxic, the model has to find out")
    b5.button("Clear", on_click=lambda: st.session_state.update(comment=""), width="stretch", icon=":material/close:")
    with b6.popover("Settings", icon=":material/tune:", width="stretch"):
        st.toggle("Use translation for Hindi / Hinglish", key="opt_translate",
                  help="Also scores the Google-translated English text and combines both scores.")
        st.toggle("Add LIME to explanations (slower)", key="opt_lime")

    if go_btn:
        if not text.strip():
            st.toast("Please enter a comment first.", icon="✍️")
        else:
            with st.spinner("Analysing — language detection, BERT scoring and explanation…"):
                st.session_state["res"] = analyze(clf, text, current_settings())
            st.session_state.pop("llm", None)

    res = st.session_state.get("res")
    if res is None:
        html(ui.empty_state())
        html('<div class="tg-muted" style="margin-top:14px;text-align:center">Tip: press a 🎲 language button to try a sample '
             "comment, then <b>Analyze</b>.</div>")
    else:
        exp = res.explanation
        st.write("")
        html(ui.result_hero(res.is_toxic, res.severity, res.positives, res.probs["toxic"], lead_sentence(res)))
        specific = [p for p in res.positives if p != "toxic"]
        main = max(specific, key=res.probs.get) if specific else None
        tgt = res.info["target"]
        html(ui.tiles([
            ("Language", short_lang(res.language.name), f"{res.language.script} script"),
            ("Main category", LABEL_DISPLAY[main] if main else ("Toxic" if res.is_toxic else "None"),
             f"{res.probs[main or 'toxic']:.0%} confidence"),
            ("Sentiment", res.sentiment["label"], f"{res.sentiment['intensity']} tone"),
            ("Aimed at", tgt.split(" (")[0], "directly addressed" if tgt.startswith("Individual")
             else (tgt[len(tgt.split(' (')[0]):].strip(" ()") or "no person or group named")),
        ]))

        st.write("")
        t_sum, t_why, t_lang, t_ai = st.tabs([":material/summarize: Summary", ":material/psychology: Why this result",
                                              ":material/translate: Language", ":material/auto_awesome: AI assistant"])

        # ---------------------------------------------------------- summary
        with t_sum:
            a, b = st.columns([1.05, 1], gap="large")
            with a:
                html(ui.heading("Why this decision?"))
                html(ui.rationale_list(res.rationale))
            with b:
                html(ui.heading("Category scores", "detected when a bar crosses its line"))
                html(ui.category_bars(res.probs, clf.thresholds))
            html(ui.heading("Words that influenced the model",
                            "shown on the English translation" if res.explained_text != res.prep.model_text else ""))
            html(ui.heatmap(res.explained_text, exp.words))

        # ---------------------------------------------------------- explainability
        with t_why:
            reason = "the strongest detected category" if res.positives else "the highest score (nothing was detected)"
            html(ui.note(f"Explaining <b>{LABEL_DISPLAY[exp.target_label]}</b> — chosen automatically as {reason}. "
                         "Several explainability methods are averaged so the result does not depend on any single one.", "🎯"))
            html(ui.heatmap(res.explained_text, exp.words))
            st.write("")
            c1, c2 = st.columns([1.3, 1], gap="large")
            with c1:
                html(ui.heading("Most influential words", "average of Integrated Gradients & Occlusion"
                                + (" & LIME" if "LIME" in exp.methods else "")))
                html(ui.influence_bars(exp.words))
            with c2:
                big, sentence = trust_text(exp)
                html(ui.heading("Can we trust this explanation?"))
                html(f'<div class="tg-card tg-anim d3"><div class="tg-eyebrow">Score change without the top words</div>'
                     f'<div class="tg-big">{big}</div><div style="margin-top:8px;color:var(--ink-2);font-size:.92rem;line-height:1.55">'
                     f'{sentence}</div><div style="margin-top:12px">{ui.chips(exp.methods, "pri")}</div></div>')
            st.write("")
            with st.expander("Technical details — raw scores, method agreement and other categories", icon=":material/science:"):
                cat = st.selectbox("Explain another category", LABELS, index=LABELS.index(exp.target_label),
                                   format_func=lambda x: f"{LABEL_DISPLAY[x]} ({res.probs[x]:.0%})", key="xai_label")
                if cat != exp.target_label:
                    from src.explainer import explain
                    with st.spinner("Computing explanation…"):
                        exp = explain(clf, res.explained_text, cat, True, True, st.session_state["opt_lime"])
                    html(ui.heatmap(res.explained_text, exp.words))
                d1, d2, d3 = st.columns(3)
                d1.metric("Comprehensiveness", f"{exp.comprehensiveness:+.1%}" if exp.comprehensiveness is not None else "–",
                          help="Drop in probability when the top words are removed (ERASER). Higher = more faithful.")
                d2.metric("Sufficiency", f"{exp.sufficiency:+.1%}" if exp.sufficiency is not None else "–",
                          help="Change when only the top words are kept. Near 0 = the top words alone explain the decision.")
                d3.metric("IG completeness residual", f"{exp.ig_delta:+.3f}" if exp.ig_delta is not None else "–",
                          help="Sum of Integrated-Gradients attributions minus the logit difference. Near 0 = complete.")
                st.plotly_chart(ui.method_comparison(exp.words, exp.methods), width="stretch", config={"displayModeBar": False})
                df = pd.DataFrame([{"Word": w.word, "Consensus": round(w.consensus, 3), "Integrated Gradients": round(w.ig, 4),
                                    "Occlusion Δp": round(w.occlusion, 4), "LIME": None if w.lime is None else round(w.lime, 4)}
                                   for w in sorted(exp.words, key=lambda w: -w.consensus)])
                st.dataframe(df if "LIME" in exp.methods else df.drop(columns=["LIME"]), width="stretch", hide_index=True)

        # ---------------------------------------------------------- language (CO1-CO3)
        with t_lang:
            lg, s, info = res.language, res.sentiment, res.info
            how = ("Devanagari script" if lg.code == "hi" else
                   "Romanised Hindi words: " + ", ".join(sorted(set(lg.hinglish_markers))[:4]) if lg.code == "hinglish" else
                   "Latin script, English vocabulary")
            idn = ", ".join(g.replace("_", " ") for g in info["identity_mentions"]) or "none mentioned"
            html(ui.tiles([("Detected language", short_lang(lg.name), how),
                           ("Sentiment", f"{s['label']} · {s['intensity']}", f"{s['compound']:+.2f} on a −1 to +1 scale (VADER)"),
                           ("Aimed at", info["target"].split(" (")[0], f"identity groups: {idn}")], cols=3))
            if lg.code != "en":
                html(ui.heading("Translation", "Google Translate · cloud AI service"))
                html(f'<div class="tg-quote tg-anim">{ui.html_escape(res.translation)}</div>')
                if res.probs_translated:
                    rule = ("An abusive word was found, so the higher score is kept." if res.ensemble_rule == "max" else
                            "No abusive word was found, so both scores are averaged to avoid false alarms.")
                    html(ui.score_flow(res.probs_original["toxic"], res.probs_translated["toxic"], res.probs["toxic"], rule))
                elif not res.translated_ok:
                    html('<div class="tg-muted">Translation service unavailable — the original text was scored on its own.</div>')
            html(ui.heading("Abusive words recognised", "for context only — BERT makes the decision"))
            html(ui.chips([f"{a['term']} · {LABEL_DISPLAY[a['category']]}" for a in info["abusive_terms"]], "hot")
                 if info["abusive_terms"] else '<span class="tg-muted">None from the multilingual lexicon.</span>')
            st.write("")
            with st.expander("NLP pipeline — cleaning, vectorisation and entity extraction", icon=":material/account_tree:"):
                p1, p2 = st.columns([1.15, 1], gap="large")
                with p1:
                    html("<b>Cleaning steps</b>")
                    for step in res.prep.steps:
                        html(f'<div class="tg-step"><div class="t">{ui.html_escape(step["step"])}</div>'
                             f'<code>{ui.html_escape((step["output"] or "∅")[:400])}</code></div>')
                with p2:
                    html("<b>TF-IDF terms</b> <span class='tg-muted'>(sparse vector)</span>")
                    if res.tfidf:
                        st.dataframe(pd.DataFrame(res.tfidf), hide_index=True, width="stretch", height=190)
                    html(f"<b>BERT tokens</b> <span class='tg-muted'>({len(res.subwords)} sub-words, dense input)</span>")
                    html(ui.chips([t["token"] for t in res.subwords[:36]]))
                    html("<br><b>Named entities</b> <span class='tg-muted'>(spaCy)</span>")
                    ents = [f"{e['text']} · {e['label']}" for e in info["named_entities"]] + info["mentions"] + info["hashtags"]
                    html(ui.chips(ents) if ents else '<span class="tg-muted">None found.</span>')

        # ---------------------------------------------------------- AI assistant (CO4)
        with t_ai:
            g1, g2 = st.columns([2.6, 1], vertical_alignment="bottom")
            strat = g1.segmented_control("Prompt strategy", list(STRATEGY_NAMES), format_func=STRATEGY_NAMES.get,
                                         default="xai_grounded", key="strategy") or "xai_grounded"
            g1.caption(STRATEGY_HELP[strat])
            if g2.button("Generate explanation", type="primary", width="stretch", icon=":material/auto_awesome:"):
                with st.spinner(f"Asking {llm_label()}…" if llm_available() else "Preparing explanation…"):
                    st.session_state["llm"] = run_llm(res, strat)

            out = st.session_state.get("llm")
            if out is None:
                html(ui.note("Get a plain-language explanation, the meaning of any slang, and a polite rewrite of the comment — "
                             "which our BERT model then re-checks." + ("" if llm_available() else
                             " <b>No LLM key configured</b>: offline templates will be used."), "✨"))
            else:
                meta = out.get("_meta", {})
                agree = out["agrees_with_classifier"]
                html('<div class="tg-anim">' + ui.chips([f"Verdict · {out['verdict'].replace('_', '-')}", f"Severity · {out['severity']}",
                                                          f"Action · {out['moderation_action']}"], "pri")
                     + ui.chips(["Agrees with BERT ✓" if agree else "Disagrees with BERT"], "ok" if agree else "warn") + "</div>")
                e1, e2 = st.columns(2, gap="large")
                with e1:
                    html(ui.heading("Explanation"))
                    html(f'<div class="tg-card tg-anim d1" style="line-height:1.6">{ui.html_escape(out["explanation"])}</div>')
                with e2:
                    html(ui.heading("Slang & cultural context"))
                    ctx = out["cultural_context"] or ""
                    body = (ui.html_escape(ctx) if ctx and not ctx.lower().startswith(("not applicable", "not available"))
                            else '<span class="tg-muted">No slang to explain.</span>')
                    html(f'<div class="tg-card tg-anim d2" style="line-height:1.6">{body}</div>')
                html(ui.heading("Polite rewrite"))
                html(f'<div class="tg-rewrite tg-anim d3">{ui.html_escape(out["detoxified_rewrite"])}</div>')
                if out.get("_rewrite_probs"):
                    before, after = res.probs["toxic"], out["_rewrite_probs"]["toxic"]
                    html(f'<div class="tg-delta">Re-checked by our BERT model: toxicity <span class="from">{before:.0%}</span> → '
                         f'<span class="to">{after:.0%}</span></div>')
                src = (f"{meta.get('provider', 'LLM')} · {meta.get('model', '')} · {STRATEGY_NAMES.get(meta.get('strategy'), '')} prompt"
                       if meta.get("source") == "llm" else "Offline template (no LLM response)")
                html(f'<div class="tg-muted" style="margin-top:10px">Source: {ui.html_escape(src)}</div>')
                st.write("")
                with st.expander("Prompt engineering — the exact prompts sent", icon=":material/code:"):
                    if meta.get("source") == "llm":
                        sysp, userp = meta["system_prompt"], meta["user_prompt"]
                    else:
                        sysp, userp = build_prompt(strat, res.text, res.language.name, res.translation, res.probs,
                                                   [(w.word, w.consensus) for w in res.explanation.top_words(8, False)])
                    st.markdown("**System prompt**")
                    st.code(sysp, language=None)
                    st.markdown("**User prompt**")
                    st.code(userp, language=None)

# ====================================================================== BATCH
with tab_batch:
    html(ui.heading("Batch analysis", "CSV · Excel · TXT · PDF · JSON — one comment per line or a comment/text column"))
    up = st.file_uploader("Upload comments", type=["csv", "xlsx", "txt", "pdf", "json"], label_visibility="collapsed")
    o1, o2, o3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    max_rows = o1.number_input("Max comments", 5, 1000, 200, step=25)
    n_topics = o2.number_input("Topics to discover", 2, 8, 4)
    run_batch = o3.button("Analyze file", type="primary", width="stretch", icon=":material/play_arrow:", disabled=up is None)
    if run_batch and up is not None:
        comments = extract_text_from_upload(up.name, up.getvalue())[: int(max_rows)]
        if not comments:
            st.error("No text could be extracted from this file.")
        else:
            with st.spinner(f"Analysing {len(comments)} comments…"):
                rows = quick_classify(clf, comments, st.session_state["opt_translate"])
            df = pd.DataFrame(rows)
            topics = fit_batch_topics([preprocess(t, "en").normalized_text for t in df["translation"]], int(n_topics))
            if topics:
                df["topic"] = [topics["topics"][i]["topic"] for i in topics["doc_topic"]]
            st.session_state["batch"] = (df, topics)

    if "batch" not in st.session_state:
        html(ui.note("Upload a file to see how many comments are toxic, which categories dominate, how each language "
                     "compares and which topics appear — then download the full results.", "📂"))
    else:
        df, topics = st.session_state["batch"]
        detected = {lab: int((df[lab] >= clf.thresholds.get(lab, .5)).sum()) for lab in LABELS}
        top_cat = max(LABELS[1:], key=detected.get)
        html(ui.tiles([("Comments analysed", f"{len(df):,}", " · ".join(map(short_lang, df["language"].unique()[:3]))),
                       ("Toxic comments", f"{int(df['is_toxic'].sum()):,}", f"{df['is_toxic'].mean():.0%} of the file"),
                       ("Most common category", LABEL_DISPLAY[top_cat] if detected[top_cat] else "—",
                        f"{detected[top_cat]} comments" if detected[top_cat] else "no sub-category detected")], cols=3))
        st.write("")
        p1, p2 = st.columns(2, gap="large")
        with p1:
            html(ui.heading("Comments per category"))
            mx = max(max(detected.values()), 1)
            html('<div class="tg-card">' + ui.value_bars([(LABEL_DISPLAY[l], detected[l] / mx, str(detected[l])) for l in LABELS],
                                                        "on") + "</div>")
        with p2:
            html(ui.heading("Toxicity rate by language"))
            rates = df.groupby("language")["is_toxic"].agg(["mean", "size"])
            html('<div class="tg-card">' + ui.value_bars([(short_lang(l), r["mean"], f"{r['mean']:.0%} of {int(r['size'])}")
                                                         for l, r in rates.iterrows()]) + "</div>")
        if topics:
            html(ui.heading("Topics found in this file", "LDA topic modelling"))
            html('<div class="tg-empty" style="grid-template-columns:repeat(%d,1fr);margin-top:0">' % min(4, len(topics["topics"]))
                 + "".join(f'<div class="tg-feature tg-anim d{i % 4 + 1}"><div class="t">{t["topic"]} · '
                           f'{df[df["topic"] == t["topic"]]["is_toxic"].mean():.0%} toxic</div>'
                           f'<div class="d">{ui.html_escape(ui.mask_offensive(", ".join(t["keywords"][:6])))}</div></div>'
                           for i, t in enumerate(topics["topics"])) + "</div>")
        html(ui.heading("Results"))
        show = df[["comment", "language", "categories", "toxic"]].rename(
            columns={"comment": "Comment", "language": "Language", "categories": "Detected categories", "toxic": "Toxicity"})
        st.dataframe(show, width="stretch", hide_index=True,
                     column_config={"Toxicity": st.column_config.ProgressColumn("Toxicity", min_value=0, max_value=1, format="%.2f")})
        st.download_button("Download full results (CSV)", df.to_csv(index=False).encode(), "toxicguard_results.csv", "text/csv",
                           icon=":material/download:")

# ====================================================================== PERFORMANCE
with tab_perf:
    metrics = load_metrics()
    if not metrics:
        html(ui.note("Test-set metrics appear here after you run the Colab notebook and copy <code>metrics.json</code> "
                     "into <code>artifacts/</code>.", "📓"))
    else:
        ov, bl = metrics.get("overall", {}), metrics.get("baseline_tfidf", {})
        n_test = metrics.get("training", {}).get("test_rows") or metrics.get("dataset", {}).get("test")
        html(ui.heading("Model performance", f"fine-tuned multilingual BERT · {n_test:,} unseen test comments" if n_test else ""))
        html(ui.kpi_rings([
            ("Accuracy", ov.get("binary_accuracy", 0), f"{ov.get('binary_accuracy', 0):.1%}", "test comments classified correctly"),
            ("Recall", ov.get("binary_recall", 0), f"{ov.get('binary_recall', 0):.1%}", "of toxic comments are caught"),
            ("Precision", ov.get("binary_precision", 0), f"{ov.get('binary_precision', 0):.1%}", "of flagged comments are truly toxic"),
            ("ROC-AUC", ov.get("macro_auc", 0), f"{ov.get('macro_auc', 0):.3f}", "ranking quality across 6 categories (1 = perfect)"),
        ]))
        st.write("")
        c1, c2 = st.columns([1.35, 1], gap="large")
        with c1:
            html(ui.heading("Fine-tuned BERT vs a classical baseline"))
            st.plotly_chart(ui.comparison_chart(
                ["Accuracy", "Precision", "Recall", "F1 (toxic)", "F1 (6 categories)"],
                [bl.get("accuracy"), bl.get("precision"), bl.get("recall"), bl.get("f1_toxic"), bl.get("macro_f1")],
                [ov.get("binary_accuracy"), ov.get("binary_precision"), ov.get("binary_recall"), ov.get("binary_f1"), ov.get("macro_f1")]),
                width="stretch", config={"displayModeBar": False})
            if ov.get("macro_f1") is not None and bl.get("macro_f1") is not None:
                html(f'<div class="tg-muted">Biggest gain: telling the <b>6 categories</b> apart — F1 {bl["macro_f1"]:.2f} → '
                     f'<b>{ov["macro_f1"]:.2f}</b> (+{(ov["macro_f1"] - bl["macro_f1"]) * 100:.1f} points).</div>')
        with c2:
            html(ui.heading("F1 by language"))
            if "per_language" in metrics:
                plang = metrics["per_language"]
                html('<div class="tg-card tg-anim d2">' + ui.value_bars(
                    [(short_lang(LANG_NAMES.get(k, k)), v["f1"], f"{v['f1']:.0%}") for k, v in plang.items()])
                     + '<div class="tg-muted" style="margin-top:12px">'
                     + " · ".join(f"{short_lang(LANG_NAMES.get(k, k))}: {v['n']:,} test comments" for k, v in plang.items())
                     + "</div></div>")
                worst = min(plang.items(), key=lambda kv: kv[1]["f1"])
                html(f'<div class="tg-muted" style="margin-top:10px">{short_lang(LANG_NAMES.get(worst[0], worst[0]))} is hardest: '
                     "romanised spelling varies and there is less training data — the translation step helps here.</div>")
        st.write("")
        with st.expander("Detailed evaluation — per category, confusion matrix, other NLP modules, training", icon=":material/table_chart:"):
            if "per_label" in metrics:
                pc = pd.DataFrame([{"Category": LABEL_DISPLAY[k], "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"],
                                    "ROC-AUC": v["roc_auc"], "Threshold": v.get("threshold")} for k, v in metrics["per_label"].items()])
                st.dataframe(pc.round(3), hide_index=True, width="stretch")
                st.caption("Rare categories (threat, severe toxic, identity hate) have few training examples, so their F1 is lower.")
            e1, e2 = st.columns(2, gap="large")
            if "confusion_matrix" in metrics:
                (tn, fp), (fn, tp) = metrics["confusion_matrix"]
                e1.dataframe(pd.DataFrame({"Predicted clean": [tn, fn], "Predicted toxic": [fp, tp]},
                                          index=["Actually clean", "Actually toxic"]), width="stretch")
                e1.caption("Confusion matrix on the test set (toxic vs clean).")
            mods = []
            if "sentiment_eval" in metrics:
                se = metrics["sentiment_eval"]
                mods.append(("Sentiment (VADER)", f"accuracy {se['accuracy']:.1%}", se.get("dataset", "")))
            if "ie_eval" in metrics:
                mods.append(("Information extraction", f"F1 {metrics['ie_eval']['f1']:.2f}", f"{metrics['ie_eval'].get('n', '')} labelled sentences"))
            if "language_id" in metrics:
                mods.append(("Language identification", f"accuracy {metrics['language_id']['accuracy']:.1%}", "900 test comments"))
            if "translation_ensemble" in metrics:
                te = metrics["translation_ensemble"]
                mods.append(("Translation ensemble", f"F1 {te['f1_without']:.3f} → {te['f1_with']:.3f}", f"{te['n']} Hindi/Hinglish"))
            if mods:
                e2.dataframe(pd.DataFrame(mods, columns=["Module", "Result", "Tested on"]), hide_index=True, width="stretch")
            if "history" in metrics:
                h = pd.DataFrame(metrics["history"])
                st.dataframe(h[[c for c in ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_macro_auc"] if c in h]].round(3),
                             hide_index=True, width="stretch")
            d = clf.describe()
            st.caption(f"Model: {d['model_type']} · {d['parameters']} parameters · {d['layers']} layers · vocabulary {d['vocab_size']:,}")

    st.write("")
    lc1, lc2 = st.columns([3, 1], vertical_alignment="bottom")
    lc1.markdown(ui.heading("Live check", f"{len(BENCHMARK)} hand-labelled comments scored by the deployed model right now"),
                 unsafe_allow_html=True)
    if lc2.button("Run live check", width="stretch", icon=":material/play_arrow:"):
        with st.spinner("Scoring…"):
            rows = quick_classify(clf, [b[0] for b in BENCHMARK], st.session_state["opt_translate"])
        bdf = pd.DataFrame({"Comment": [b[0] for b in BENCHMARK], "Language": [b[2] for b in BENCHMARK],
                            "Expected": ["toxic" if b[1] else "clean" for b in BENCHMARK],
                            "Predicted": ["toxic" if r["is_toxic"] else "clean" for r in rows]})
        ok = bdf["Expected"] == bdf["Predicted"]
        bdf["Correct"] = ok.map({True: "✅", False: "❌"})
        acc = ok.groupby(bdf["Language"]).mean()
        html(ui.tiles([("Overall", f"{ok.mean():.0%}", f"{int(ok.sum())} of {len(ok)} correct")]
                      + [(short_lang(LANG_NAMES.get(k, k)), f"{v:.0%}", "accuracy") for k, v in acc.items()], cols=len(acc) + 1))
        st.write("")
        st.dataframe(bdf, hide_index=True, width="stretch")

# ====================================================================== ABOUT
with tab_about:
    html(ui.heading("Project demonstration rubric", "30 marks · where each criterion is shown in the app"))
    html(ui.rubric_cards([
        ("CO1", 7, "Text extraction, preprocessing, cleaning & vectorisation of unstructured data",
         "Analyze › Language › NLP pipeline · Batch analysis",
         ["File upload (CSV/PDF/…)", "Cleaning steps", "Hinglish normalisation", "TF-IDF", "BERT tokens"]),
        ("CO2", 8, "Implementation & evaluation of sentiment analysis and information extraction",
         "Analyze › Summary & Language · Model performance",
         ["Fine-tuned mBERT", "VADER sentiment", "NER & targets", "Baseline comparison"]),
        ("CO3", 7, "Topic modelling or multilingual NLP using NLP techniques & cloud AI services",
         "Analyze › Language · Batch analysis",
         ["EN · HI · Hinglish detection", "Google Translate", "Translation ensemble", "LDA topics"]),
        ("CO4", 8, "Generative AI / LLM application, prompt engineering, integration & working demo",
         "Analyze › AI assistant",
         ["Gemini (JSON output)", "Zero / few-shot / XAI-grounded", "Polite rewrite", "BERT re-check"]),
    ]))
    st.write("")
    html(ui.heading("How it works"))
    html(ui.architecture([
        ("1 · Input", "Comment or file", "Text box, or CSV / Excel / TXT / PDF upload"),
        ("2 · Language", "Detect & translate", "English / Hindi / Hinglish, Google Translate"),
        ("3 · Preprocess", "Clean & tokenise", "Normalisation, stop-words, TF-IDF, WordPiece"),
        ("4 · Detect", "Fine-tuned mBERT", "6 category scores with tuned thresholds"),
        ("5 · Explain", "XAI", "Integrated Gradients + Occlusion (+ LIME)"),
        ("6 · Analyse", "Sentiment & entities", "VADER, spaCy NER, targets, LDA topics"),
        ("7 · Generate", "LLM assistant", "Explanation, slang meaning, polite rewrite"),
    ]))
    st.write("")
    html(ui.heading("Data & models"))
    html('<div class="tg-card"><div style="line-height:1.9;color:var(--ink-2)">'
         "<b>Data:</b> Jigsaw Toxic Comment Classification (English, 6 labels) · TextDetox multilingual toxicity (Hindi, Hinglish, English)<br>"
         "<b>Detector:</b> bert-base-multilingual-cased, fine-tuned with a masked multi-label loss and per-category thresholds<br>"
         "<b>Other components:</b> VADER sentiment · spaCy NER · scikit-learn LDA &amp; TF-IDF · Google Gemini</div></div>")
