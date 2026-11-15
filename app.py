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
:root{--page:#e8edf0;--card:rgba(255,255,255,.72);--card-solid:#fbfcfc;--ink:#122528;--muted:#687b80;--line:rgba(18,37,40,.12);--teal:#087f86;--deep:#12363a;
--red:#c95751;--sred:#fff0ee;--amber:#b47a2c;--samber:#fff7e7;--green:#33805d;--sgreen:#eaf8f1;--track:#dfe8e9;--shadow:0 24px 70px rgba(18,54,58,.12)}
html,body,.stApp{font-family:'DM Sans',sans-serif;color:var(--ink);color-scheme:light}
.stApp{background:radial-gradient(circle at 8% 2%,rgba(139,215,220,.45),transparent 30%),radial-gradient(circle at 94% 18%,rgba(255,255,255,.96),transparent 28%),linear-gradient(145deg,#e8edf0 0%,#f4f6f5 52%,#e0e9eb 100%);background-attachment:fixed}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;background-image:radial-gradient(rgba(18,54,58,.13) .6px,transparent .6px);background-size:8px 8px;mask-image:linear-gradient(to bottom,black,transparent 65%)}
.stApp [data-testid="stMarkdownContainer"],.stApp label p{color:var(--ink)}
header[data-testid="stHeader"],#MainMenu,footer{display:none}
.block-container{max-width:1560px;padding:2.2rem 3.5rem 2.5rem}
div[data-testid="stVerticalBlock"]{gap:1.15rem}
div[data-testid="stHorizontalBlock"]{gap:1.5rem}

div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border:1px solid rgba(255,255,255,.9)!important;border-radius:26px!important;padding:1.7rem 1.8rem;box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.9);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);animation:rise .55s ease both}
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stHorizontalBlock"]):not(:has(div[data-baseweb="textarea"])){position:relative;overflow:hidden;background:linear-gradient(135deg,rgba(219,249,242,.68),rgba(255,255,255,.52) 48%,rgba(226,244,241,.62))!important;border-color:rgba(255,255,255,.98)!important;box-shadow:0 28px 80px rgba(18,54,58,.16),inset 0 1px 0 rgba(255,255,255,.98),inset 0 -1px 0 rgba(8,127,134,.08)}
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stHorizontalBlock"]):not(:has(div[data-baseweb="textarea"])):before{content:"";position:absolute;inset:-45% 28% 42% -10%;background:linear-gradient(115deg,rgba(255,255,255,.62),rgba(255,255,255,0));transform:rotate(-8deg);pointer-events:none}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.auto-category){position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(232,250,245,.74),rgba(255,255,255,.48) 56%,rgba(221,242,241,.62))!important;border-color:rgba(255,255,255,.98)!important;box-shadow:0 20px 55px rgba(18,54,58,.13),inset 0 1px 0 rgba(255,255,255,.98),inset 0 -1px 0 rgba(8,127,134,.08)}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.auto-category):before{content:"";position:absolute;inset:-35% -12% 58% 24%;background:linear-gradient(115deg,rgba(255,255,255,.68),rgba(255,255,255,0));transform:rotate(8deg);pointer-events:none}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{box-shadow:0 28px 80px rgba(18,54,58,.15),inset 0 1px 0 rgba(255,255,255,.95)}
div[data-testid="stForm"]{border:0!important;padding:0!important;background:transparent!important}

.topbar{display:flex;justify-content:space-between;align-items:center;padding-bottom:1.35rem;border-bottom:1px solid rgba(18,54,58,.14)}
.logo{display:flex;align-items:center;gap:14px}
.mark{width:48px;height:48px;border-radius:17px;background:linear-gradient(145deg,#159ba1,#075f68);color:#fff;display:grid;place-items:center;font:800 17px 'Manrope',sans-serif;letter-spacing:-.5px;box-shadow:0 9px 22px rgba(8,127,134,.3),inset 0 1px 0 rgba(255,255,255,.45)}
.brand{font:800 25px 'Manrope',sans-serif;letter-spacing:-.7px;line-height:1.1}
.brand-sub{font-size:13px;color:var(--muted);margin-top:4px;letter-spacing:.02em}
.pill{display:inline-flex;align-items:center;gap:9px;font-size:13px;font-weight:700;color:var(--teal);background:rgba(234,248,241,.78);border:1px solid rgba(51,128,93,.22);border-radius:999px;padding:9px 15px;box-shadow:inset 0 1px 0 rgba(255,255,255,.9)}
.dot{width:8px;height:8px;border-radius:50%;background:#3fa374;box-shadow:0 0 0 4px rgba(63,163,116,.15)}
.command{font:800 clamp(25px,3vw,38px)/1.1 'Manrope',sans-serif;letter-spacing:-1.3px;color:var(--deep);margin:2.3rem 0 .35rem}
.command-sub{font-size:15px;color:var(--muted);margin-bottom:1rem}

.sec-title{font:800 20px 'Manrope',sans-serif;letter-spacing:-.4px}
.sec-sub{font-size:14px;color:var(--muted);margin:5px 0 7px}
.auto-category{display:inline-flex;align-items:center;gap:8px;margin:12px 0 16px;padding:8px 12px;border:1px solid rgba(8,127,134,.18);border-radius:999px;background:rgba(234,248,241,.72);color:var(--teal);font-size:13px;font-weight:700}
.auto-category strong{color:var(--deep);font-weight:800}

div[data-baseweb="textarea"]{background:linear-gradient(135deg,rgba(210,247,239,.86),rgba(255,255,255,.82) 52%,rgba(228,248,244,.82))!important;border:1px solid rgba(18,54,58,.16)!important;border-radius:18px!important;overflow:hidden;box-shadow:0 8px 24px rgba(18,54,58,.06),inset 0 1px 0 rgba(255,255,255,.95)}
div[data-baseweb="textarea"]:focus-within{border-color:var(--teal)!important;box-shadow:0 0 0 4px rgba(8,127,134,.12),0 8px 24px rgba(18,54,58,.08),inset 0 1px 0 #fff!important}
.stTextArea textarea{background:linear-gradient(135deg,rgba(210,247,239,.86),rgba(255,255,255,.82) 52%,rgba(228,248,244,.82))!important;border:0!important;border-radius:17px!important;outline:0!important;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;opacity:1!important;font:400 18px/1.65 'DM Sans',sans-serif!important;padding:18px 19px!important;min-height:210px!important;resize:none!important}
.stTextArea textarea::placeholder{color:#93a4a8!important}
div[data-testid="stWidgetLabel"] p{font-size:14px!important}
div[data-testid="stFormSubmitButton"] button,button[kind="secondaryFormSubmit"]{background:linear-gradient(135deg,#16494d,#092f35)!important;color:#fff!important;border:0!important;border-radius:16px!important;min-height:56px;font-weight:700;box-shadow:0 10px 24px rgba(9,47,53,.2);transition:transform .2s,box-shadow .2s,filter .2s}
div[data-testid="stFormSubmitButton"] button p{color:#fff!important;font:700 16px 'DM Sans',sans-serif!important}
div[data-testid="stFormSubmitButton"] button:hover{filter:brightness(1.12);transform:translateY(-2px);box-shadow:0 14px 28px rgba(9,47,53,.27)}
div[data-testid="stButton"] button{background:rgba(255,255,255,.55);border:1px solid var(--line);border-radius:999px;min-height:40px;color:var(--ink)}
div[data-testid="stButton"] button p{font:600 14.5px 'DM Sans',sans-serif;color:var(--ink)}
div[data-testid="stButton"] button:hover{border-color:var(--teal);background:var(--sgreen)}
div[data-testid="stAlert"]{border-radius:15px}
div[data-testid="stAlert"] p{font-size:15px}
.hint{font-size:13px;color:var(--muted)}

.pipe{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}
.step{background:rgba(246,250,250,.62);border:1px solid rgba(18,54,58,.1);border-radius:17px;padding:15px;transition:transform .2s,background .2s}
.step:hover{transform:translateY(-3px);background:rgba(255,255,255,.88)}
.step b{display:block;font:800 12px 'Manrope',sans-serif;color:var(--teal);margin-bottom:7px}
.step span{display:block;font-weight:700;font-size:14px}
.step em{display:block;font-style:normal;font-size:12px;color:var(--muted);margin-top:4px;line-height:1.45}

.verdict{display:flex;align-items:center;gap:26px;border-radius:21px;padding:23px 24px;margin:8px 0 7px;border:1px solid rgba(255,255,255,.82);box-shadow:inset 0 1px 0 rgba(255,255,255,.9)}
.verdict.bad{background:linear-gradient(120deg,rgba(255,240,238,.92),rgba(255,249,247,.58))}.verdict.warn{background:linear-gradient(120deg,rgba(255,247,231,.92),rgba(255,252,244,.58))}.verdict.good{background:linear-gradient(120deg,rgba(234,248,241,.92),rgba(246,255,251,.58))}.verdict.idle{background:rgba(243,248,248,.72)}
.ring{width:132px;height:132px;flex:none;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--c) calc(var(--p)*1%),var(--track) 0);box-shadow:0 12px 22px rgba(18,54,58,.1)}
.ring-in{width:102px;height:102px;border-radius:50%;background:rgba(255,255,255,.9);display:grid;place-items:center;text-align:center;box-shadow:inset 0 1px 4px rgba(18,54,58,.08)}
.ring-v{font:800 28px 'Manrope',sans-serif;line-height:1}
.ring-l{font-size:12px;color:var(--muted);margin-top:3px}
.v-level{font:700 14px 'DM Sans',sans-serif;margin-bottom:5px}
.bad .v-level{color:var(--red)}.warn .v-level{color:var(--amber)}.good .v-level{color:var(--green)}.idle .v-level{color:var(--muted)}
.v-title{font:800 25px/1.2 'Manrope',sans-serif;letter-spacing:-.6px}
.v-sub{font-size:15px;color:var(--muted);margin-top:7px;line-height:1.5}
.stats{display:grid;grid-template-columns:1fr;gap:10px;margin:12px 0 10px}
.stat{border:1px solid rgba(18,54,58,.11);border-radius:17px;padding:14px 16px;background:rgba(255,255,255,.43)}
.stat-l{font-size:13px;color:var(--muted)}
.stat-v{font:800 25px 'Manrope',sans-serif;letter-spacing:-.6px;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

.row{display:grid;grid-template-columns:150px 1fr 78px 108px 100px;align-items:center;gap:14px;padding:13px 0;border-top:1px solid rgba(18,54,58,.08)}
.row:first-child{border-top:0}
.r-name{font-size:16px;font-weight:700}
.track{height:10px;background:var(--track);border-radius:99px;overflow:hidden}
.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#7bb2b0,#3d8888);transition:width .6s ease}.fill.hit{background:linear-gradient(90deg,#e18378,var(--red))}
.r-val{text-align:right;font:700 16px 'Manrope',sans-serif}
.r-threshold{text-align:right;font-size:12px;color:var(--muted)}
.r-tag{justify-self:end;font-size:12px;font-weight:700;border-radius:99px;padding:5px 11px;color:var(--muted);background:rgba(18,54,58,.06)}
.r-tag.hit{color:var(--red);background:var(--sred)}
.category-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:4px 0 20px}
.category-card{min-width:0;padding:13px 14px;border:1px solid rgba(18,54,58,.1);border-radius:17px;background:rgba(255,255,255,.42);box-shadow:inset 0 1px 0 rgba(255,255,255,.85)}
.category-card.detected{border-color:rgba(201,87,81,.22);background:rgba(255,240,238,.5)}
.category-card-name{font-size:13px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.category-card-score{font:800 23px 'Manrope',sans-serif;letter-spacing:-.5px;margin-top:6px;color:var(--deep)}
.category-card-meta{font-size:11px;color:var(--muted);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.category-card-track{height:5px;margin-top:10px;border-radius:99px;background:var(--track);overflow:hidden}
.category-card-track span{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#7bb2b0,#3d8888)}
.category-card.detected .category-card-track span{background:linear-gradient(90deg,#e18378,var(--red))}

.xgrid{display:grid;grid-template-columns:1.7fr 1fr;gap:36px;margin-top:8px}
.explain-select{max-width:320px;margin:12px 0 16px}
.annot{font:400 25px/1.9 'Newsreader',serif;color:var(--deep);white-space:pre-wrap;overflow-wrap:anywhere}
.tok{padding:2px 5px;border-radius:7px;-webkit-box-decoration-break:clone;box-decoration-break:clone}
.legend{display:flex;gap:22px;font-size:13px;color:var(--muted);margin-top:16px}
.sw{display:inline-block;width:13px;height:13px;border-radius:4px;vertical-align:-2px;margin-right:7px}
.drv-h{font:800 15px 'Manrope',sans-serif;margin-bottom:6px}
.drv{display:grid;grid-template-columns:110px 1fr 74px;align-items:center;gap:12px;padding:8px 0}
.d-w{font:500 19px 'Newsreader',serif;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.d-v{text-align:right;font:700 13px 'Manrope',sans-serif}
.empty{border:1px dashed rgba(18,54,58,.22);border-radius:17px;padding:26px;color:var(--muted);font-size:15px;line-height:1.6;background:rgba(248,252,252,.55)}
.foot{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:13px;border-top:1px solid rgba(18,54,58,.14);padding-top:16px;margin-top:8px}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
@media(max-width:1100px){.block-container{padding:1rem 1.2rem}.pipe{grid-template-columns:1fr 1fr}.verdict{flex-direction:column;align-items:flex-start}.row{grid-template-columns:110px 1fr 64px}.r-tag,.r-threshold{display:none}.category-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:650px){.topbar{align-items:flex-start;gap:14px;flex-direction:column}.pill{font-size:12px}.pipe{grid-template-columns:1fr}.category-grid{grid-template-columns:1fr}.legend{flex-wrap:wrap;gap:10px}.foot{flex-direction:column;gap:5px}}
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
# Header
# ------------------------------------------------------------
st.markdown(h("""
<div class="topbar">
  <div class="logo"><div class="mark">TG</div>
    <div><div class="brand">ToxicGuard AI</div><div class="brand-sub">Language intelligence system</div></div></div>
  <div class="pill"><span class="dot"></span>Local model: BERT base uncased</div>
</div>
"""), unsafe_allow_html=True)

st.markdown('<div class="command">Enter the text to analyse</div>'
            '<div class="command-sub">Paste a comment below to inspect its language patterns and risk signals.</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    with st.form("comment_analysis_form", clear_on_submit=False):
        comment = st.text_area(
            "Comment",
            placeholder="Paste a comment to inspect its language patterns...",
            height=150,
            label_visibility="collapsed",
            key="comment_input",
        )
        input_controls = st.columns([1, 0.25])
        with input_controls[0]:
            explain_on = st.toggle("Highlight the words behind the score", value=True, key="explain_toggle")
        with input_controls[1]:
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
                        selected = max(results, key=lambda item: float(item["probability"]))
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

# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------
results = st.session_state.get("tg_results")

report_card = st.container(border=True)
explain_card = st.container(border=True)

with report_card:
    st.markdown(h('<div class="sec-title">Classification report</div>'
                  '<div class="sec-sub">One score for each of the six toxicity categories.</div>'),
                unsafe_allow_html=True)

    report_summary, report_scores = st.columns([0.82, 1.18], gap="large")
    with report_summary:
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
        else:
            st.markdown(h("""
            <div class="verdict idle">
              <div class="ring" style="--p:0;--c:var(--teal)"><div class="ring-in"><div><div class="ring-v">&ndash;</div><div class="ring-l">peak score</div></div></div></div>
              <div><div class="v-level">Ready</div><div class="v-title">Waiting for a comment</div>
              <div class="v-sub">Add a comment above to see the six category scores.</div></div>
            </div>
            """), unsafe_allow_html=True)

    with report_scores:
        if results:
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
            rows = "".join(
                f"""<div class="row"><div class="r-name">{html.escape(n)}</div><div class="track"><div class="fill" style="width:0%"></div></div>
                <div class="r-val">&ndash;</div><div class="r-threshold">Configured limit</div><div class="r-tag">Pending</div></div>"""
                for n in LABEL_NAMES.values())
            st.markdown(h(rows), unsafe_allow_html=True)

# ------------------------------------------------------------
# Full width: explanation
# ------------------------------------------------------------
with explain_card:
    st.markdown(h('<div class="sec-title">Explainable AI</div>'
                  '<div class="sec-sub">Understand which parts of the comment influenced the model prediction.</div>'),
                unsafe_allow_html=True)
    if results:
        highest_category = max(results, key=lambda item: float(item["probability"]))
        st.session_state["tg_explain_category"] = highest_category["category"]
        st.markdown(
            f'<div class="auto-category"><span>Auto-explaining highest score</span>'
            f'<strong>{html.escape(label_of(highest_category["category"]))}</strong>'
            f'<span>{float(highest_category["probability"]):.1%}</span></div>',
            unsafe_allow_html=True,
        )
        category_cards = []
        for item in results:
            probability = min(max(float(item["probability"]), 0.0), 1.0)
            hit = item["detected"]
            status = "Detected" if hit else "Below limit"
            category_cards.append(
                f'<div class="category-card {"detected" if hit else ""}">'
                f'<div class="category-card-name">{html.escape(label_of(item["category"]))}</div>'
                f'<div class="category-card-score">{probability:.1%}</div>'
                f'<div class="category-card-meta">{status} · limit {float(item["threshold"]):.1%}</div>'
                f'<div class="category-card-track"><span style="width:{probability * 100:.2f}%"></span></div>'
                f'</div>'
            )
        st.markdown(f'<div class="category-grid">{"".join(category_cards)}</div>', unsafe_allow_html=True)
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


st.markdown(h("""
<div class="foot"><span>ToxicGuard AI: BERT-powered multi-label classification</span>
<span>Scores are indicative, not a final judgement of intent.</span></div>
"""), unsafe_allow_html=True)
