import html
import streamlit as st

from utils.prediction import predict_comment, explain_comment

st.set_page_config(
    page_title="ToxicGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LABEL_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severe toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
}
MAX_SHAP_EVALS = 120

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');
:root{--page:#f2f4f0;--card:#fff;--ink:#1b2a29;--muted:#667570;--line:#dfe5e0;--teal:#2f5d5e;--deep:#1f3d3e;
--red:#bf4f47;--sred:#fbecea;--amber:#b27a2c;--samber:#fbf3e3;--green:#3f7a5c;--sgreen:#ecf4ee;--track:#e8ede9}
html,body,.stApp{font-family:'DM Sans',sans-serif;color:var(--ink);color-scheme:light}
.stApp{background:var(--page)}
.stApp [data-testid="stMarkdownContainer"],.stApp label p{color:var(--ink)}
header[data-testid="stHeader"],#MainMenu,footer{display:none}
.block-container{max-width:1600px;padding:1.6rem 3rem 2rem}
div[data-testid="stVerticalBlock"]{gap:1rem}
div[data-testid="stHorizontalBlock"]{gap:1.5rem}

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border:1px solid var(--line)!important;
border-radius:16px!important;padding:1.6rem 1.7rem;box-shadow:0 1px 2px rgba(27,42,41,.04)}
div[data-testid="stForm"]{border:0!important;padding:0!important;background:transparent!important}

/* top bar + hero */
.topbar{display:flex;justify-content:space-between;align-items:center;padding-bottom:1.1rem;border-bottom:1px solid var(--line)}
.logo{display:flex;align-items:center;gap:14px}
.mark{width:44px;height:44px;border-radius:12px;background:var(--teal);color:#fff;display:grid;place-items:center;
font:800 17px 'Manrope',sans-serif;letter-spacing:-.5px}
.brand{font:800 25px 'Manrope',sans-serif;letter-spacing:-.6px;line-height:1.1}
.brand-sub{font-size:14px;color:var(--muted);margin-top:2px}
.pill{display:inline-flex;align-items:center;gap:9px;font-size:14px;font-weight:600;color:var(--teal);
background:var(--sgreen);border:1px solid #d5e6da;border-radius:999px;padding:8px 16px}
.dot{width:9px;height:9px;border-radius:50%;background:#4f9a72;box-shadow:0 0 0 4px #dcece2}
.hero{display:flex;justify-content:flex-start;align-items:flex-end;padding:1.5rem 0 .5rem}
.headline{font:400 clamp(44px,4.8vw,78px)/1.02 'Newsreader',serif;letter-spacing:-2px;color:var(--deep)}
.hero-copy{max-width:520px;font-size:18px;line-height:1.6;color:var(--muted);padding-bottom:8px}

/* section headers */
.sec-title{font:700 21px 'Manrope',sans-serif;letter-spacing:-.3px}
.sec-sub{font-size:15px;color:var(--muted);margin:3px 0 4px}

/* input */
div[data-baseweb="textarea"]{background:#fff!important;border:1.5px solid #cfd9d2!important;border-radius:12px!important}
div[data-baseweb="textarea"]:focus-within{border-color:var(--teal)!important;box-shadow:0 0 0 3px rgba(47,93,94,.14)!important}
.stTextArea textarea{background:#fff!important;color:#000!important;-webkit-text-fill-color:#000!important;
opacity:1!important;font:400 18px/1.6 'DM Sans',sans-serif!important;padding:16px 18px!important;min-height:210px!important}
.stTextArea textarea::placeholder{color:#93a09a!important}
div[data-testid="stWidgetLabel"] p{font-size:15px!important}
div[data-testid="stFormSubmitButton"] button,button[kind="secondaryFormSubmit"]{background:var(--deep)!important;color:#fff!important;
border:0!important;border-radius:12px!important;min-height:56px;font-weight:700;transition:background .15s}
div[data-testid="stFormSubmitButton"] button p{color:#fff!important;font:700 17px 'DM Sans',sans-serif!important}
div[data-testid="stFormSubmitButton"] button:hover{background:var(--teal)!important}
div[data-testid="stButton"] button{background:#f3f6f3;border:1px solid var(--line);border-radius:999px;min-height:40px;color:var(--ink)}
div[data-testid="stButton"] button p{font:600 14.5px 'DM Sans',sans-serif;color:var(--ink)}
div[data-testid="stButton"] button:hover{border-color:var(--teal);background:var(--sgreen)}
div[data-testid="stAlert"] p{font-size:15px}
.hint{font-size:14px;color:var(--muted)}

/* pipeline */
.pipe{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:10px}
.step{background:#f6f8f5;border:1px solid #e6ece7;border-radius:10px;padding:14px}
.step b{display:block;font:800 13px 'Manrope',sans-serif;color:var(--teal);margin-bottom:6px}
.step span{display:block;font-weight:600;font-size:15px}
.step em{display:block;font-style:normal;font-size:13px;color:var(--muted);margin-top:3px;line-height:1.4}

/* verdict */
.verdict{display:flex;align-items:center;gap:26px;border-radius:14px;padding:22px 24px;margin:6px 0 4px}
.verdict.bad{background:var(--sred)}.verdict.warn{background:var(--samber)}.verdict.good{background:var(--sgreen)}.verdict.idle{background:#f3f6f3}
.ring{width:132px;height:132px;flex:none;border-radius:50%;display:grid;place-items:center;
background:conic-gradient(var(--c) calc(var(--p)*1%),var(--track) 0)}
.ring-in{width:102px;height:102px;border-radius:50%;background:#fff;display:grid;place-items:center;text-align:center}
.ring-v{font:800 28px 'Manrope',sans-serif;line-height:1}
.ring-l{font-size:12px;color:var(--muted);margin-top:3px}
.v-level{font:700 15px 'DM Sans',sans-serif;margin-bottom:4px}
.bad .v-level{color:var(--red)}.warn .v-level{color:var(--amber)}.good .v-level{color:var(--green)}.idle .v-level{color:var(--muted)}
.v-title{font:800 26px/1.2 'Manrope',sans-serif;letter-spacing:-.5px}
.v-sub{font-size:16px;color:var(--muted);margin-top:6px;line-height:1.5}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0 8px}
.stat{border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.stat-l{font-size:14px;color:var(--muted)}
.stat-v{font:800 26px 'Manrope',sans-serif;letter-spacing:-.5px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* score rows */
.row{display:grid;grid-template-columns:150px 1fr 78px 108px 100px;align-items:center;gap:14px;padding:13px 0;border-top:1px solid #edf1ed}
.row:first-child{border-top:0}
.r-name{font-size:17px;font-weight:600}
.track{height:12px;background:var(--track);border-radius:99px;overflow:hidden}
.fill{height:100%;border-radius:99px;background:#6f9a96}.fill.hit{background:var(--red)}
.r-val{text-align:right;font:700 17px 'Manrope',sans-serif}
.r-threshold{text-align:right;font-size:13px;color:var(--muted)}
.r-tag{justify-self:end;font-size:13px;font-weight:700;border-radius:99px;padding:5px 12px;color:var(--muted);background:#f0f3f0}
.r-tag.hit{color:var(--red);background:var(--sred)}

/* explanation */
.xgrid{display:grid;grid-template-columns:1.7fr 1fr;gap:36px;margin-top:8px}
.explain-select{max-width:320px;margin:12px 0 16px}
.explain-summary{margin-top:16px;padding:13px 15px;border-left:3px solid var(--teal);background:#f3f7f4;
font-size:14px;line-height:1.55;color:var(--muted)}
.annot{font:400 25px/1.9 'Newsreader',serif;color:var(--deep);white-space:pre-wrap;overflow-wrap:anywhere}
.tok{padding:2px 5px;border-radius:5px;-webkit-box-decoration-break:clone;box-decoration-break:clone}
.legend{display:flex;gap:22px;font-size:14px;color:var(--muted);margin-top:16px}
.sw{display:inline-block;width:14px;height:14px;border-radius:4px;vertical-align:-2px;margin-right:7px}
.drv-h{font:700 16px 'Manrope',sans-serif;margin-bottom:6px}
.drv{display:grid;grid-template-columns:110px 1fr 74px;align-items:center;gap:12px;padding:8px 0}
.d-w{font:500 19px 'Newsreader',serif;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.d-v{text-align:right;font:700 14px 'Manrope',sans-serif}
.empty{border:1.5px dashed #cbd8d1;border-radius:12px;padding:26px;color:var(--muted);font-size:16px;line-height:1.6;background:#f8faf8}
.foot{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:13.5px;border-top:1px solid var(--line);padding-top:14px;margin-top:6px}

@media(max-width:1100px){.block-container{padding:1rem 1.2rem}.hero{display:block}.hero-copy{margin-top:14px}
.xgrid{grid-template-columns:1fr}.pipe{grid-template-columns:1fr 1fr}.verdict{flex-direction:column;align-items:flex-start}
.row{grid-template-columns:110px 1fr 64px}.r-tag,.r-threshold{display:none}.stats{grid-template-columns:1fr}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def h(markup: str) -> str:
    """Flatten HTML to one line so Markdown never treats it as a code block."""
    return "".join(line.strip() for line in markup.splitlines())


def label_of(key: str) -> str:
    return LABEL_NAMES.get(key, key.replace("_", " ").title())


@st.cache_data(show_spinner=False, max_entries=512)
def _cached_explanation(text: str, key: str):
    # SHAP is intentionally computed only after the user submits an analysis.
    return explain_comment(text, key, max_evals=MAX_SHAP_EVALS)


def verdict_state(results):
    top = max(float(r["probability"]) for r in results)
    flagged = [r for r in results if r["detected"]]
    if flagged and top >= 0.8:
        return "bad", "High risk", "Likely to need moderation", f"{len(flagged)} of 6 categories crossed their threshold."
    if flagged:
        return "warn", "Elevated", "Worth a human review", f"{len(flagged)} of 6 categories crossed their threshold."
    return "good", "Clean", "No category crossed its threshold", "The model did not flag this comment."


# ------------------------------------------------------------
# Header + hero
# ------------------------------------------------------------
st.markdown(h("""
<div class="topbar">
  <div class="logo"><div class="mark">TG</div>
    <div><div class="brand">ToxicGuard AI</div><div class="brand-sub">Language intelligence system</div></div></div>
  <div class="pill"><span class="dot"></span>Local model: BERT base uncased</div>
</div>
<div class="hero">
  <div class="headline">Understand harmful language.</div>
</div>
"""), unsafe_allow_html=True)

left, right = st.columns([0.9, 1.25], gap="large")

# ------------------------------------------------------------
# Left: input + pipeline
# ------------------------------------------------------------
with left:
    with st.container(border=True):
        st.markdown(h('<div class="sec-title">Your comment</div>'
                      '<div class="sec-sub">Type or paste a comment to begin.</div>'),
                    unsafe_allow_html=True)

        with st.form("comment_analysis_form", clear_on_submit=False):
            comment = st.text_area(
                "Comment",
                placeholder="Paste a comment to inspect its language patterns...",
                height=230,
                label_visibility="collapsed",
                key="comment_input",
            )
            explain_on = st.toggle("Highlight the words behind the score", value=True, key="explain_toggle")
            submitted = st.form_submit_button("Analyse comment", use_container_width=True)
        st.markdown('<div class="hint">Original wording is kept, so the model reads the comment in context.</div>',
                    unsafe_allow_html=True)

        if submitted:
            clean = comment.strip()
            if not clean:
                for k in ("tg_results", "tg_error", "tg_expl"):
                    st.session_state.pop(k, None)
                st.warning("Enter a comment before running the analysis.")
            else:
                try:
                    with st.spinner("Evaluating comment..."):
                        results = predict_comment(clean)
                        st.session_state.update(tg_results=results, tg_comment=clean, tg_expl=None, tg_note="")
                        st.session_state.pop("tg_error", None)
                        if explain_on:
                            # Explain the strongest flagged category; if none is flagged,
                            # explain the category with the highest probability.
                            flagged = [item for item in results if item["detected"]]
                            selected = max(
                                flagged or results,
                                key=lambda item: float(item["probability"])
                            )
                            st.session_state["tg_explain_category"] = selected["category"]
                            try:
                                with st.spinner("Generating SHAP word attributions..."):
                                    expl = _cached_explanation(clean, selected["category"])
                                st.session_state["tg_expl"] = expl
                            except Exception as exc:
                                st.session_state["tg_note"] = f"SHAP explanation could not be generated: {exc}"
                        else:
                            st.session_state["tg_note"] = "SHAP highlighting is switched off."
                except Exception as exc:
                    st.session_state.pop("tg_results", None)
                    st.session_state["tg_error"] = str(exc)

        if st.session_state.get("tg_error"):
            st.error(f"Analysis failed: {st.session_state['tg_error']}")

    with st.container(border=True):
        st.markdown(h("""
        <div class="sec-title">How it works</div>
        <div class="sec-sub">What happens between the text box and the scores.</div>
        <div class="pipe">
          <div class="step"><b>1</b><span>Tokenise</span><em>Split into WordPiece tokens</em></div>
          <div class="step"><b>2</b><span>Encode</span><em>BERT reads each word in context</em></div>
          <div class="step"><b>3</b><span>Score</span><em>Six independent sigmoid outputs</em></div>
          <div class="step"><b>4</b><span>Decide</span><em>Each label has its own threshold</em></div>
        </div>
        """), unsafe_allow_html=True)

# ------------------------------------------------------------
# Right: classification report
# ------------------------------------------------------------
results = st.session_state.get("tg_results")

with right:
    with st.container(border=True):
        st.markdown(h('<div class="sec-title">Classification report</div>'
                      '<div class="sec-sub">One score for each of the six toxicity categories.</div>'),
                    unsafe_allow_html=True)

        if results:
            state, level, title, sub = verdict_state(results)
            highest = max(results, key=lambda x: float(x["probability"]))
            peak = float(highest["probability"])
            color = {"bad": "var(--red)", "warn": "var(--amber)", "good": "var(--green)"}[state]
            n_hit = sum(1 for r in results if r["detected"])
            st.markdown(h(f"""
            <div class="verdict {state}">
              <div class="ring" style="--p:{peak * 100:.1f};--c:{color}">
                <div class="ring-in"><div><div class="ring-v">{peak:.0%}</div><div class="ring-l">peak score</div></div></div>
              </div>
              <div><div class="v-level">{level}</div><div class="v-title">{title}</div><div class="v-sub">{sub}</div></div>
            </div>
            <div class="stats">
              <div class="stat"><div class="stat-l">Flagged labels</div><div class="stat-v">{n_hit} / 6</div></div>
              <div class="stat"><div class="stat-l">Highest score</div><div class="stat-v">{peak:.1%}</div></div>
              <div class="stat"><div class="stat-l">Top category</div><div class="stat-v">{html.escape(label_of(highest['category']))}</div></div>
            </div>
            """), unsafe_allow_html=True)
            rows = []
            for item in results:
                p = min(max(float(item["probability"]), 0.0), 1.0)
                hit = "hit" if item["detected"] else ""
                tag = "Detected" if item["detected"] else "Below limit"
                rows.append(f"""<div class="row"><div class="r-name">{html.escape(label_of(item['category']))}</div>
                <div class="track"><div class="fill {hit}" style="width:{p * 100:.2f}%"></div></div>
                <div class="r-val">{p:.1%}</div><div class="r-threshold">Limit {float(item['threshold']):.1%}</div>
                <div class="r-tag {hit}">{tag}</div></div>""")
            st.markdown(
                h('<div class="hint" style="display:grid;grid-template-columns:150px 1fr 78px 108px 100px;'
                  'gap:14px;padding:8px 0 2px;font-size:12px"><span>Category</span><span>Probability</span>'
                  '<span style="text-align:right">Score</span><span style="text-align:right">Threshold</span>'
                  '<span style="text-align:right">Status</span></div>') + h("".join(rows)),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(h("""
            <div class="verdict idle">
              <div class="ring" style="--p:0;--c:var(--teal)"><div class="ring-in"><div><div class="ring-v">&ndash;</div><div class="ring-l">peak score</div></div></div></div>
              <div><div class="v-level">Ready</div><div class="v-title">Waiting for a comment</div>
              <div class="v-sub">Add a comment on the left and choose Analyse comment. All six scores appear here.</div></div>
            </div>
            """), unsafe_allow_html=True)
            rows = "".join(
                f"""<div class="row"><div class="r-name">{html.escape(n)}</div><div class="track"><div class="fill" style="width:0%"></div></div>
                <div class="r-val">&ndash;</div><div class="r-threshold">Configured limit</div><div class="r-tag">Pending</div></div>"""
                for n in LABEL_NAMES.values())
            st.markdown(h(rows), unsafe_allow_html=True)

# ------------------------------------------------------------
# Full width: explanation
# ------------------------------------------------------------
with st.container(border=True):
    st.markdown(h('<div class="sec-title">Explainable AI</div>'
                  '<div class="sec-sub">Understand which parts of the comment influenced the model prediction.</div>'),
                unsafe_allow_html=True)
    if results:
        category_keys = [item["category"] for item in results]
        current_category = st.session_state.get("tg_explain_category", category_keys[0])
        selected_category = st.selectbox(
            "Category to explain",
            category_keys,
            index=category_keys.index(current_category) if current_category in category_keys else 0,
            format_func=label_of,
            key="explain_category_selector",
        )
        if selected_category != st.session_state.get("tg_explain_category"):
            st.session_state["tg_explain_category"] = selected_category
            try:
                with st.spinner("Generating SHAP word attributions..."):
                    st.session_state["tg_expl"] = _cached_explanation(
                        st.session_state["tg_comment"],
                        selected_category,
                    )
                st.session_state.pop("tg_note", None)
            except Exception as exc:
                st.session_state["tg_expl"] = None
                st.session_state["tg_note"] = f"SHAP explanation could not be generated: {exc}"
    expl = st.session_state.get("tg_expl")

    if results and expl:
        tokens = expl["tokens"]
        values = expl["values"]
        peak_i = max((abs(v) for v in values), default=0.0) or 1.0

        spans = []
        for token, value in zip(tokens, values):
            if abs(value) < 1e-6:
                spans.append(f'<span class="tok">{html.escape(token)}</span>')
                continue
            alpha = 0.14 + 0.48 * abs(value) / peak_i
            rgb = "191,79,71" if value > 0 else "63,122,92"
            direction = "raises" if value > 0 else "lowers"
            spans.append(
                f'<span class="tok" style="background:rgba({rgb},{alpha:.2f});'
                f'box-shadow:inset 0 -2px 0 rgba({rgb},.85)" '
                f'title="{direction} selected score; SHAP value {value:+.5f}">'
                f'{html.escape(token)}</span>'
            )

        order = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)[:6]
        drivers = []
        for i in order:
            value = values[i]
            if abs(value) < 1e-6:
                continue
            color = "var(--red)" if value > 0 else "var(--green)"
            direction = "↑" if value > 0 else "↓"
            drivers.append(
                f'<div class="drv"><div class="d-w">{html.escape(tokens[i])}</div>'
                f'<div class="track"><div class="fill" style="width:{abs(value) / peak_i * 100:.0f}%;background:{color}"></div></div>'
                f'<div class="d-v" style="color:{color}">{direction} {value:+.4f}</div></div>'
            )

        category = expl["category"]
        body = "".join(drivers) or '<div class="hint">No individual token had a measurable contribution.</div>'
        st.markdown(h(f"""
        <div class="xgrid">
          <div>
            <div class="annot" style="white-space:pre-wrap">{"".join(spans)}</div>
            <div class="legend">
              <span><i class="sw" style="background:rgba(191,79,71,.5)"></i>Pushes score higher</span>
              <span><i class="sw" style="background:rgba(63,122,92,.5)"></i>Pushes score lower</span>
              <span>Explaining: {html.escape(label_of(category))}</span>
            </div>
            <div class="hint" style="margin-top:10px">
              SHAP baseline: {expl["base"]:.4f} · Reconstructed output: {expl["output"]:.4f}.
              Token attributions explain this model output; they do not establish intent or causation.
            </div>
          </div>
          <div><div class="drv-h">Strongest token contributions</div>{body}</div>
        </div>
        """), unsafe_allow_html=True)
    else:
        if results:
            msg = st.session_state.get("tg_note") or "SHAP explanation is not available for this analysis."
        else:
            msg = "Analyse a comment to see SHAP token contributions. Red highlights push the selected category score higher; green highlights push it lower."
        st.markdown(f'<div class="empty">{html.escape(msg)}</div>', unsafe_allow_html=True)


st.markdown(h("""
<div class="foot"><span>ToxicGuard AI: BERT-powered multi-label classification</span>
<span>Scores are indicative, not a final judgement of intent.</span></div>
"""), unsafe_allow_html=True)
