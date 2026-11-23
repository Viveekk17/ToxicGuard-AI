"""ToxicGuardAI design system: tokens, CSS (with motion) and HTML components.

Every component returns an HTML string rendered with ``st.markdown(..., unsafe_allow_html=True)``.
Animations are pure CSS (no JavaScript) so they are fast, reliable and work on Streamlit Cloud.
"""
from __future__ import annotations

import html
import math
import re

import plotly.graph_objects as go

from .config import LABEL_DISPLAY, LABELS

# ------------------------------------------------------------------ tokens (also used by plotly)
PRIMARY = "#4F46E5"
VIOLET = "#7C3AED"
DANGER = "#E11D48"
WARN = "#F59E0B"
SAFE = "#059669"
MUTED = "#64748B"
INK = "#0B1020"

SEVERITY = {  # severity -> (gradient start, gradient end, icon)
    "Safe": ("#059669", "#10B981", "✓"),
    "Borderline": ("#D97706", "#F59E0B", "!"),
    "Moderate": ("#EA580C", "#F97316", "!"),
    "High": ("#E11D48", "#F43F5E", "!"),
    "Severe": ("#9F1239", "#E11D48", "!"),
}

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#F5F6FB; --ink:#0B1020; --ink-2:#334155; --muted:#64748B; --line:#E6E8F0; --line-2:#EEF0F6;
  --surface:rgba(255,255,255,.8); --glass-b:rgba(255,255,255,.9);
  --primary:#4F46E5; --violet:#7C3AED; --primary-50:#EEF2FF; --primary-100:#E0E7FF;
  --danger:#E11D48; --safe:#059669; --warn:#F59E0B;
  --r-md:14px; --r-lg:20px; --r-xl:24px;
  --sh-1:0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06);
  --sh-2:0 4px 6px -2px rgba(16,24,40,.03), 0 12px 24px -4px rgba(16,24,40,.08);
  --ease:cubic-bezier(.2,.8,.2,1);
}
html, body, [class*="css"], .stMarkdown, button, input, textarea, select, label{font-family:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif !important;}
code, pre{font-family:"JetBrains Mono",ui-monospace,monospace !important;}

/* ---------- chrome ---------- */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"], [data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"]{display:none !important;}
header[data-testid="stHeader"]{background:transparent; height:0;}
.stApp{background:
   radial-gradient(1100px 520px at 0% -10%, rgba(79,70,229,.10), transparent 60%),
   radial-gradient(900px 460px at 100% -5%, rgba(124,58,237,.08), transparent 55%), var(--bg);}
.block-container{max-width:1240px; padding:1.25rem 2rem 4rem;}
h1,h2,h3,h4,h5{letter-spacing:-.02em; color:var(--ink);}

/* ---------- motion ---------- */
@keyframes tgFadeUp{from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:none}}
@keyframes tgFade{from{opacity:0} to{opacity:1}}
@keyframes tgGrow{from{width:0}}
@keyframes tgRing{from{stroke-dashoffset:var(--c)}}
@keyframes tgShift{0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
@keyframes tgPulse{0%{box-shadow:0 0 0 0 rgba(5,150,105,.45)} 70%{box-shadow:0 0 0 7px rgba(5,150,105,0)} 100%{box-shadow:0 0 0 0 rgba(5,150,105,0)}}
@keyframes tgPop{0%{transform:scale(.6); opacity:0} 70%{transform:scale(1.08)} 100%{transform:scale(1); opacity:1}}
.tg-anim{animation:tgFadeUp .6s var(--ease) both;}
.d1{animation-delay:.06s}.d2{animation-delay:.12s}.d3{animation-delay:.18s}.d4{animation-delay:.24s}.d5{animation-delay:.3s}
@media (prefers-reduced-motion: reduce){*{animation:none !important; transition:none !important;}}

/* ---------- brand bar ---------- */
.tg-brand{display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; margin-bottom:16px;
  background:var(--surface); backdrop-filter:blur(16px) saturate(160%); -webkit-backdrop-filter:blur(16px) saturate(160%);
  border:1px solid var(--glass-b); border-radius:var(--r-lg); box-shadow:var(--sh-2); animation:tgFade .5s var(--ease) both;}
.tg-brand .left{display:flex; align-items:center; gap:14px;}
.tg-logo{width:44px; height:44px; border-radius:13px; display:grid; place-items:center; color:#fff; font-size:22px;
  background:linear-gradient(135deg,#4F46E5,#7C3AED,#06B6D4,#4F46E5); background-size:300% 300%; animation:tgShift 8s ease infinite;
  box-shadow:0 8px 20px rgba(79,70,229,.35);}
.tg-brand .name{font-size:1.25rem; font-weight:800; letter-spacing:-.03em; color:var(--ink); line-height:1.1;}
.tg-brand .tag{font-size:.82rem; color:var(--muted); margin-top:2px;}
.tg-status{display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end;}
.tg-status .s{display:inline-flex; align-items:center; gap:7px; padding:6px 11px; border-radius:999px; font-size:.76rem; font-weight:600;
  color:var(--ink-2); background:#fff; border:1px solid var(--line);}
.tg-dot{width:8px; height:8px; border-radius:50%; background:var(--safe); animation:tgPulse 2.2s infinite;}
.tg-dot.off{background:#F59E0B; animation:none;}

/* ---------- navigation: segmented control ---------- */
.stTabs [role="tablist"]{gap:4px; padding:5px; background:rgba(15,23,42,.045); border-radius:14px; width:fit-content; max-width:100%;
  flex-wrap:wrap; border:1px solid rgba(15,23,42,.04);}
.stTabs [role="tab"]{padding:8px 16px; border-radius:10px; transition:all .2s var(--ease);}
.stTabs [role="tab"] p{font-weight:600; font-size:.9rem; color:var(--muted);}
.stTabs [role="tab"]:hover p{color:var(--ink);}
.stTabs [role="tab"][aria-selected="true"]{background:#fff; box-shadow:var(--sh-1), 0 0 0 1px rgba(15,23,42,.04);}
.stTabs [role="tab"][aria-selected="true"] p{color:var(--ink);}
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none !important;}
.stTabs [role="tabpanel"]{padding-top:18px; animation:tgFade .35s var(--ease) both;}

/* ---------- inputs & buttons ---------- */
.stTextArea textarea{background:#fff !important; border:1px solid var(--line) !important; border-radius:var(--r-md) !important;
  font-size:1.05rem !important; line-height:1.55 !important; padding:16px 18px !important; box-shadow:var(--sh-1) !important;
  transition:border-color .2s, box-shadow .2s;}
.stTextArea textarea:focus{border-color:var(--primary) !important; box-shadow:0 0 0 4px rgba(79,70,229,.14), var(--sh-1) !important;}
.stButton>button, .stDownloadButton>button, [data-testid="stPopover"] button{border-radius:12px; font-weight:600; min-height:42px;
  border:1px solid var(--line); background:#fff; color:var(--ink); box-shadow:var(--sh-1); transition:all .18s var(--ease);}
.stButton>button:hover, .stDownloadButton>button:hover, [data-testid="stPopover"] button:hover{border-color:#C7CCF8; color:var(--primary);
  transform:translateY(-1px); box-shadow:var(--sh-2);}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--primary),var(--violet)); color:#fff; border:0;
  box-shadow:0 10px 22px -6px rgba(79,70,229,.55);}
.stButton>button[kind="primary"]:hover{color:#fff; filter:brightness(1.06); box-shadow:0 14px 28px -6px rgba(79,70,229,.6);}
.stButton>button:active{transform:translateY(0) scale(.98);}
div[data-baseweb="select"]>div{border-radius:12px !important;}
div[data-testid="stExpander"] details{background:var(--surface); border:1px solid var(--line) !important; border-radius:var(--r-md) !important; box-shadow:var(--sh-1);}
div[data-testid="stExpander"] summary p{font-weight:600; color:var(--ink-2);}
div[data-testid="stDataFrame"]{border-radius:var(--r-md); overflow:hidden; border:1px solid var(--line); box-shadow:var(--sh-1);}
div[data-testid="stFileUploaderDropzone"]{background:#fff; border:1.5px dashed #C7CCF8; border-radius:var(--r-lg); padding:26px;}
div[data-testid="stPlotlyChart"]{background:#fff; border:1px solid var(--line); border-radius:var(--r-lg); padding:8px; box-shadow:var(--sh-1);}
div[data-testid="stAlert"]{border-radius:var(--r-md);}

/* ---------- typography helpers ---------- */
.tg-h{display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:6px 0 10px;}
.tg-h .t{font-size:1.02rem; font-weight:700; color:var(--ink); letter-spacing:-.01em;}
.tg-h .sub{font-size:.82rem; color:var(--muted); text-align:right;}
.tg-eyebrow{font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; font-weight:700; color:var(--muted);}
.tg-muted{color:var(--muted); font-size:.86rem;}
.tg-label{font-weight:700; font-size:1rem; color:var(--ink); margin:4px 0 8px;}

/* ---------- surfaces ---------- */
.tg-card{background:#fff; border:1px solid var(--line); border-radius:var(--r-lg); padding:18px 20px; box-shadow:var(--sh-1);
  transition:transform .2s var(--ease), box-shadow .2s var(--ease);}
.tg-card.hover:hover{transform:translateY(-3px); box-shadow:var(--sh-2);}
.tg-tile{background:#fff; border:1px solid var(--line); border-radius:var(--r-md); padding:14px 16px; box-shadow:var(--sh-1); height:100%;
  transition:transform .2s var(--ease), box-shadow .2s var(--ease);}
.tg-tile:hover{transform:translateY(-2px); box-shadow:var(--sh-2);}
.tg-tile .v{font-size:1.12rem; font-weight:750; color:var(--ink); margin-top:4px; letter-spacing:-.02em;}
.tg-tile .s{font-size:.8rem; color:var(--muted); margin-top:2px; line-height:1.35;}

/* ---------- result hero ---------- */
.tg-result{display:grid; grid-template-columns:auto 1fr; gap:26px; align-items:center; padding:22px 26px; border-radius:var(--r-xl);
  background:#fff; border:1px solid var(--line); box-shadow:var(--sh-2); position:relative; overflow:hidden;}
.tg-result::before{content:""; position:absolute; inset:0 auto 0 0; width:5px; background:var(--g2);}
.tg-result::after{content:""; position:absolute; right:-90px; top:-90px; width:280px; height:280px; border-radius:50%;
  background:radial-gradient(circle, var(--glow), transparent 70%); pointer-events:none;}
.tg-ringwrap{position:relative; width:136px; height:136px;}
.tg-ring{width:136px; height:136px;}
.tg-ring .trk{fill:none; stroke:var(--line-2); stroke-width:11;}
.tg-ring .val{fill:none; stroke-width:11; stroke-linecap:round; transform:rotate(-90deg); transform-origin:60px 60px;
  animation:tgRing 1.3s var(--ease) both .1s;}
.tg-ringwrap .num{position:absolute; inset:0; display:grid; place-items:center; text-align:center;}
.tg-ringwrap .num b{display:block; font-size:1.8rem; font-weight:800; letter-spacing:-.04em; color:var(--ink); line-height:1;}
.tg-ringwrap .num span{display:block; font-size:.66rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-top:5px; font-weight:700;}
.tg-verdict{display:flex; align-items:center; gap:12px; flex-wrap:wrap;}
.tg-verdict .badge{display:inline-flex; align-items:center; gap:8px; padding:6px 14px 6px 7px; border-radius:999px; color:#fff; font-weight:800;
  font-size:.78rem; letter-spacing:.08em; background:linear-gradient(135deg,var(--g1),var(--g2)); box-shadow:0 8px 18px -6px var(--glow2);}
.tg-verdict .badge i{width:22px; height:22px; border-radius:50%; display:grid; place-items:center; background:rgba(255,255,255,.25);
  font-style:normal; font-size:.8rem; animation:tgPop .6s var(--ease) both .4s;}
.tg-verdict .title{font-size:1.6rem; font-weight:800; letter-spacing:-.03em; color:var(--ink);}
.tg-result .lead{color:var(--ink-2); font-size:.95rem; margin:8px 0 12px; line-height:1.5;}
.tg-tiles{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0 6px;}
@media (max-width:900px){.tg-result{grid-template-columns:1fr} .tg-tiles{grid-template-columns:repeat(2,1fr)}}

/* ---------- chips ---------- */
.tg-chip{display:inline-flex; align-items:center; gap:6px; padding:4px 11px; margin:3px 6px 3px 0; border-radius:999px; font-size:.8rem;
  font-weight:600; color:var(--ink-2); background:#F1F3F9; border:1px solid var(--line);}
.tg-chip.hot{color:#9F1239; background:#FFF1F3; border-color:#FECDD6;}
.tg-chip.ok{color:#065F46; background:#ECFDF5; border-color:#A7F3D0;}
.tg-chip.pri{color:#3730A3; background:var(--primary-50); border-color:var(--primary-100);}
.tg-chip.warn{color:#92400E; background:#FFFBEB; border-color:#FDE68A;}

/* ---------- animated bars ---------- */
.tg-bars .row{display:grid; grid-template-columns:118px 1fr 56px; gap:12px; align-items:center; padding:7px 0;}
.tg-bars .lbl{font-size:.86rem; font-weight:600; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.tg-bars .lbl.on{color:var(--ink);}
.tg-bars .track{position:relative; height:10px; background:#EEF0F6; border-radius:999px;}
.tg-bars .fill{position:absolute; left:0; top:0; bottom:0; width:var(--w); border-radius:999px; background:#CBD2E1; animation:tgGrow 1s var(--ease) both;}
.tg-bars .fill.on{background:linear-gradient(90deg,#FB7185,#E11D48); box-shadow:0 4px 10px -2px rgba(225,29,72,.4);}
.tg-bars .fill.pri{background:linear-gradient(90deg,#A5B4FC,#4F46E5);}
.tg-bars .fill.neg{background:linear-gradient(90deg,#6EE7B7,#059669);}
.tg-bars .thr{position:absolute; top:-5px; bottom:-5px; width:2px; background:var(--ink); border-radius:2px; opacity:.5;}
.tg-bars .val{font-size:.84rem; font-weight:700; color:var(--ink); text-align:right; font-variant-numeric:tabular-nums;}
.tg-legend{display:flex; gap:16px; flex-wrap:wrap; font-size:.76rem; color:var(--muted); margin-top:10px;}
.tg-legend i{display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px;}

/* ---------- heat-map ---------- */
.tg-heat{font-size:1.12rem; line-height:2.4; padding:16px 18px; background:#fff; border:1px solid var(--line); border-radius:var(--r-md); box-shadow:var(--sh-1);}
.tg-heat .w{padding:4px 7px; border-radius:8px; margin:0 1px; animation:tgFadeUp .5s var(--ease) both; display:inline-block; line-height:1.4;
  transition:transform .15s var(--ease);}
.tg-heat .w:hover{transform:translateY(-2px);}

/* ---------- misc ---------- */
.tg-note{display:flex; gap:10px; align-items:flex-start; background:var(--primary-50); border:1px solid var(--primary-100); color:#312E81;
  border-radius:var(--r-md); padding:12px 14px; font-size:.9rem; margin:0 0 14px;}
.tg-quote{background:#fff; border:1px solid var(--line); border-left:4px solid var(--primary); border-radius:var(--r-md);
  padding:14px 16px; font-size:1.02rem; color:var(--ink); box-shadow:var(--sh-1);}
.tg-list{list-style:none; padding:0; margin:0;}
.tg-list li{position:relative; padding:10px 0 10px 26px; border-bottom:1px dashed var(--line); font-size:.93rem; color:var(--ink-2);
  animation:tgFadeUp .5s var(--ease) both; line-height:1.5;}
.tg-list li:last-child{border-bottom:0;}
.tg-list li::before{content:""; position:absolute; left:2px; top:17px; width:10px; height:10px; border-radius:3px;
  background:linear-gradient(135deg,var(--primary),var(--violet));}
.tg-flow{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:10px;}
.tg-flow .n{background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px 14px; box-shadow:var(--sh-1); min-width:118px;}
.tg-flow .n b{display:block; font-size:1.2rem; color:var(--ink); font-weight:800;}
.tg-flow .n span{font-size:.7rem; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.06em;}
.tg-flow .n.final{border-color:var(--primary-100); background:var(--primary-50);}
.tg-flow .arrow{color:#A5B4FC; font-size:1.2rem; font-weight:800;}
.tg-big{font-size:2.4rem; font-weight:800; letter-spacing:-.04em; color:var(--ink); line-height:1.1;}
.tg-step{border-left:3px solid var(--primary); padding:6px 0 6px 12px; margin-bottom:8px;}
.tg-step .t{font-weight:700; font-size:.8rem; color:#4338CA;}
.tg-step code{white-space:pre-wrap; word-break:break-word; background:transparent; font-size:.82rem;}
.tg-rewrite{background:linear-gradient(180deg,#F0FDF9,#ECFDF5); border:1px solid #A7F3D0; border-radius:var(--r-md); padding:16px 18px;
  font-size:1.06rem; color:#064E3B; box-shadow:var(--sh-1);}
.tg-delta{display:flex; align-items:center; gap:10px; margin-top:10px; font-size:.86rem; color:var(--muted); flex-wrap:wrap;}
.tg-delta .from{color:#9F1239; font-weight:800;} .tg-delta .to{color:#065F46; font-weight:800;}

/* ---------- empty state ---------- */
.tg-empty{display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:18px;}
.tg-feature{background:#fff; border:1px solid var(--line); border-radius:var(--r-lg); padding:20px; box-shadow:var(--sh-1);
  transition:transform .25s var(--ease), box-shadow .25s var(--ease);}
.tg-feature:hover{transform:translateY(-4px); box-shadow:var(--sh-2);}
.tg-feature .ic{width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:20px; margin-bottom:12px;
  background:var(--primary-50); border:1px solid var(--primary-100);}
.tg-feature .t{font-weight:700; color:var(--ink); font-size:1rem;}
.tg-feature .d{color:var(--muted); font-size:.86rem; margin-top:4px; line-height:1.5;}
@media (max-width:900px){.tg-empty{grid-template-columns:1fr}}

/* ---------- KPI rings ---------- */
.tg-kpis{display:grid; grid-template-columns:repeat(4,1fr); gap:14px;}
.tg-kpi{background:#fff; border:1px solid var(--line); border-radius:var(--r-lg); padding:18px; box-shadow:var(--sh-1); display:flex; gap:14px;
  align-items:center; transition:transform .2s var(--ease), box-shadow .2s var(--ease);}
.tg-kpi:hover{transform:translateY(-3px); box-shadow:var(--sh-2);}
.tg-kpi svg{width:64px; height:64px; flex:none;}
.tg-kpi .trk{fill:none; stroke:var(--line-2); stroke-width:10;}
.tg-kpi .val{fill:none; stroke:var(--primary); stroke-width:10; stroke-linecap:round; transform:rotate(-90deg); transform-origin:60px 60px;
  animation:tgRing 1.3s var(--ease) both .15s;}
.tg-kpi .v{font-size:1.45rem; font-weight:800; letter-spacing:-.03em; color:var(--ink);}
.tg-kpi .k{font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:700;}
.tg-kpi .s{font-size:.77rem; color:var(--muted); margin-top:2px; line-height:1.35;}
@media (max-width:900px){.tg-kpis{grid-template-columns:repeat(2,1fr)}}

/* ---------- about ---------- */
.tg-rubric{display:grid; grid-template-columns:repeat(2,1fr); gap:14px;}
.tg-co{display:inline-block; font-size:.72rem; font-weight:800; letter-spacing:.06em; color:#fff; padding:3px 9px; border-radius:7px;
  background:linear-gradient(135deg,var(--primary),var(--violet));}
.tg-marks{float:right; font-size:.8rem; font-weight:700; color:var(--primary); background:var(--primary-50); padding:3px 9px; border-radius:7px;}
.tg-arch{display:flex; flex-wrap:wrap; gap:10px; align-items:stretch;}
.tg-arch .n{flex:1 1 150px; background:#fff; border:1px solid var(--line); border-radius:var(--r-md); padding:12px 14px; box-shadow:var(--sh-1);
  animation:tgFadeUp .5s var(--ease) both;}
.tg-arch .n .k{font-size:.66rem; font-weight:800; color:var(--primary); letter-spacing:.08em; text-transform:uppercase;}
.tg-arch .n .t{font-weight:700; color:var(--ink); font-size:.92rem; margin-top:2px;}
.tg-arch .n .d{font-size:.78rem; color:var(--muted); margin-top:3px; line-height:1.4;}
@media (max-width:900px){.tg-rubric{grid-template-columns:1fr}}
</style>
"""


# ------------------------------------------------------------------ helpers
def html_escape(text) -> str:
    return html.escape(str(text))


def md_inline(text: str) -> str:
    """Tiny markdown -> HTML for **bold** and *italic* inside generated sentences."""
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)


def chips(items: list[str], tone: str = "") -> str:
    if not items:
        return '<span class="tg-muted">—</span>'
    return "".join(f'<span class="tg-chip {tone}">{html.escape(str(i))}</span>' for i in items)


def heading(title: str, sub: str = "") -> str:
    return f'<div class="tg-h"><div class="t">{html.escape(title)}</div><div class="sub">{html.escape(sub)}</div></div>'


def note(text_html: str, icon: str = "💡") -> str:
    return f'<div class="tg-note tg-anim"><div>{icon}</div><div>{text_html}</div></div>'


def tile(label: str, value: str, sub: str = "", delay: int = 1) -> str:
    return (f'<div class="tg-tile tg-anim d{delay}"><div class="tg-eyebrow">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(value)}</div><div class="s">{html.escape(sub)}</div></div>')


def tiles(items: list[tuple[str, str, str]], cols: int = 4) -> str:
    return (f'<div class="tg-tiles" style="grid-template-columns:repeat({cols},1fr)">'
            + "".join(tile(l, v, s, i + 1) for i, (l, v, s) in enumerate(items)) + "</div>")


# ------------------------------------------------------------------ brand bar & empty state
def brand_bar(name: str, tagline: str, model_ok: bool, model_text: str, llm_ok: bool, llm_text: str) -> str:
    def s(ok, txt):
        return f'<span class="s"><span class="tg-dot {"" if ok else "off"}"></span>{html.escape(txt)}</span>'
    return (f'<div class="tg-brand"><div class="left"><div class="tg-logo">🛡️</div><div><div class="name">{html.escape(name)}</div>'
            f'<div class="tag">{html.escape(tagline)}</div></div></div>'
            f'<div class="tg-status">{s(model_ok, model_text)}{s(llm_ok, llm_text)}'
            f'<span class="s">🌐 English · हिन्दी · Hinglish</span></div></div>')


def empty_state() -> str:
    feats = [("🧠", "Detects automatically", "A fine-tuned multilingual BERT scores every comment on six categories — "
              "toxic, severe toxic, obscene, threat, insult and identity hate."),
             ("🔬", "Explains every decision", "Highlights the exact words that drove the result and checks that the "
              "explanation is faithful to the model."),
             ("✨", "Suggests a better version", "An LLM explains slang in plain language and rewrites the comment politely, "
              "then BERT re-checks the rewrite.")]
    return '<div class="tg-empty">' + "".join(
        f'<div class="tg-feature tg-anim d{i + 1}"><div class="ic">{ic}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, (ic, t, d) in enumerate(feats)) + "</div>"


# ------------------------------------------------------------------ result hero
def _ring(p: float, color: str, r: int = 52) -> str:
    c = 2 * math.pi * r
    off = c * (1 - max(0.0, min(1.0, p)))
    return (f'<svg class="tg-ring" viewBox="0 0 120 120"><circle class="trk" cx="60" cy="60" r="{r}"/>'
            f'<circle class="val" cx="60" cy="60" r="{r}" style="stroke:{color};stroke-dasharray:{c:.1f};'
            f'stroke-dashoffset:{off:.1f};--c:{c:.1f}"/></svg>')


def result_hero(is_toxic: bool, severity: str, positives: list[str], prob: float, lead: str) -> str:
    g1, g2, icon = SEVERITY.get(severity, SEVERITY["High"] if is_toxic else SEVERITY["Safe"])
    title = "Toxic comment" if is_toxic else ("Borderline comment" if severity == "Borderline" else "Clean comment")
    badge = "TOXIC" if is_toxic else ("BORDERLINE" if severity == "Borderline" else "NON-TOXIC")
    cats = chips([LABEL_DISPLAY[p] for p in positives], "hot") if positives else chips(["No category detected"], "ok")
    style = f"--g1:{g1};--g2:{g2};--glow:{g2}26;--glow2:{g2}99"
    return (f'<div class="tg-result tg-anim" style="{style}">'
            f'<div class="tg-ringwrap">{_ring(prob, g2)}<div class="num"><div><b>{prob * 100:.0f}%</b><span>toxicity</span></div></div></div>'
            f'<div><div class="tg-verdict"><span class="badge"><i>{icon}</i>{badge}</span><span class="title">{title}</span>'
            f'<span class="tg-chip">Severity · {html.escape(severity)}</span></div>'
            f'<div class="lead">{md_inline(lead)}</div><div>{cats}</div></div></div>')


# ------------------------------------------------------------------ animated bars
def category_bars(probs: dict, thresholds: dict) -> str:
    rows = []
    for i, lab in enumerate(LABELS):
        p, t = probs[lab], thresholds.get(lab, 0.5)
        on = p >= t
        rows.append(f'<div class="row"><div class="lbl {"on" if on else ""}">{LABEL_DISPLAY[lab]}</div>'
                    f'<div class="track"><div class="fill {"on" if on else ""}" style="--w:{max(p, .006) * 100:.1f}%;'
                    f'animation-delay:{.06 * i:.2f}s"></div><div class="thr" style="left:{t * 100:.1f}%" '
                    f'title="decision threshold {t:.0%}"></div></div><div class="val">{p * 100:.0f}%</div></div>')
    legend = ('<div class="tg-legend"><span><i style="background:#E11D48"></i>detected</span>'
              '<span><i style="background:#CBD2E1"></i>not detected</span>'
              '<span><i style="background:#0B1020;width:3px"></i>decision threshold</span></div>')
    return f'<div class="tg-card tg-anim d2"><div class="tg-bars">{"".join(rows)}</div>{legend}</div>'


def influence_bars(words, n: int = 7) -> str:
    ws = sorted(words, key=lambda w: -abs(w.consensus))[:n]
    if not ws:
        return '<div class="tg-card tg-muted">No words to rank.</div>'
    m = max(abs(w.consensus) for w in ws) or 1
    rows = []
    for i, w in enumerate(ws):
        pos = w.consensus > 0
        rows.append(f'<div class="row"><div class="lbl on" title="{html.escape(w.word)}">{html.escape(w.word)}</div>'
                    f'<div class="track"><div class="fill {"on" if pos else "neg"}" style="--w:{abs(w.consensus) / m * 100:.1f}%;'
                    f'animation-delay:{.06 * i:.2f}s"></div></div>'
                    f'<div class="val" style="color:{DANGER if pos else SAFE}">{"↑" if pos else "↓"} {abs(w.consensus):.2f}</div></div>')
    legend = ('<div class="tg-legend"><span><i style="background:#E11D48"></i>raises the score</span>'
              '<span><i style="background:#059669"></i>lowers the score</span></div>')
    return f'<div class="tg-card tg-anim d2"><div class="tg-bars">{"".join(rows)}</div>{legend}</div>'


def heatmap(text: str, words, attr: str = "consensus") -> str:
    vals = [getattr(w, attr) or 0.0 for w in words]
    m = max((abs(v) for v in vals), default=0) or 1.0
    out, cursor = [], 0
    for i, (w, v) in enumerate(zip(words, vals)):
        out.append(html.escape(text[cursor:w.start]))
        a = min(abs(v) / m, 1.0) * 0.85
        rgb = "225,29,72" if v > 0 else "5,150,105"
        strong = a > 0.45
        tip = f"influence {w.consensus:+.2f} · IG {w.ig:+.3f} · occlusion {w.occlusion:+.3f}"
        out.append(f'<span class="w" title="{html.escape(tip)}" style="background:rgba({rgb},{a:.2f});'
                   f'{"color:#fff;font-weight:700;" if strong else ""}animation-delay:{min(i * .035, 1.2):.2f}s">'
                   f'{html.escape(text[w.start:w.end])}</span>')
        cursor = w.end
    out.append(html.escape(text[cursor:]))
    legend = ('<div class="tg-legend"><span><i style="background:#E11D48"></i>pushes toward toxic</span>'
              '<span><i style="background:#059669"></i>pushes toward clean</span><span>darker = stronger · hover a word for scores</span></div>')
    return f'<div class="tg-heat tg-anim">{"".join(out)}</div>{legend}'


def rationale_list(lines: list[str]) -> str:
    return '<ul class="tg-list">' + "".join(
        f'<li style="animation-delay:{.08 * i:.2f}s">{md_inline(l)}</li>' for i, l in enumerate(lines)) + "</ul>"


def score_flow(orig: float, tr: float, final: float, rule_text: str) -> str:
    return (f'<div class="tg-flow tg-anim d2"><div class="n"><span>Original text</span><b>{orig:.0%}</b></div>'
            f'<div class="arrow">+</div><div class="n"><span>Translation</span><b>{tr:.0%}</b></div>'
            f'<div class="arrow">→</div><div class="n final"><span>Final score</span><b>{final:.0%}</b></div>'
            f'<div class="tg-muted" style="flex:1 1 220px">{html.escape(rule_text)}</div></div>')


# ------------------------------------------------------------------ performance
def kpi_rings(items: list[tuple[str, float, str, str]]) -> str:
    """items: (label, value 0..1, display text, explanation)."""
    c = 2 * math.pi * 48
    cards = []
    for i, (lab, v, disp, expl) in enumerate(items):
        off = c * (1 - max(0.0, min(1.0, v)))
        cards.append(f'<div class="tg-kpi tg-anim d{i + 1}"><svg viewBox="0 0 120 120"><circle class="trk" cx="60" cy="60" r="48"/>'
                     f'<circle class="val" cx="60" cy="60" r="48" style="stroke-dasharray:{c:.1f};stroke-dashoffset:{off:.1f};--c:{c:.1f}"/></svg>'
                     f'<div><div class="k">{html.escape(lab)}</div><div class="v">{html.escape(disp)}</div>'
                     f'<div class="s">{html.escape(expl)}</div></div></div>')
    return f'<div class="tg-kpis">{"".join(cards)}</div>'


def value_bars(rows: list[tuple[str, float, str]], cls: str = "pri") -> str:
    """Generic animated horizontal bars: (label, value 0..1, display)."""
    out = "".join(f'<div class="row"><div class="lbl on">{html.escape(l)}</div><div class="track"><div class="fill {cls}" '
                  f'style="--w:{v * 100:.1f}%;animation-delay:{.08 * i:.2f}s"></div></div><div class="val">{html.escape(d)}</div></div>'
                  for i, (l, v, d) in enumerate(rows))
    return f'<div class="tg-bars">{out}</div>'


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif", size=12.5, color=INK),
                      yaxis=dict(gridcolor="#EEF0F6", zeroline=False), xaxis=dict(showgrid=False))
    return fig


def comparison_chart(names: list[str], base: list, ours: list) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=names, y=base, name="TF-IDF + Logistic Regression", marker_color="#CBD2E1",
                text=[f"{v:.0%}" if v is not None else "" for v in base], textposition="outside")
    fig.add_bar(x=names, y=ours, name="Fine-tuned mBERT (ours)", marker_color=PRIMARY,
                text=[f"{v:.0%}" if v is not None else "" for v in ours], textposition="outside")
    _base_layout(fig, 330)
    fig.update_layout(barmode="group", bargap=.35, bargroupgap=.08, legend=dict(orientation="h", y=1.14, x=0))
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
    return fig


def simple_bar(x, y, colors=None, height: int = 300, title: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors or PRIMARY, text=y, textposition="outside"))
    _base_layout(fig, height)
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return fig


def method_comparison(words, methods: list[str], n: int = 10) -> go.Figure:
    ws = sorted(words, key=lambda w: -abs(w.consensus))[:n]

    def norm(vals):
        m = max((abs(v) for v in vals), default=0) or 1
        return [v / m for v in vals]
    fig = go.Figure()
    if "Integrated Gradients" in methods:
        fig.add_bar(x=[w.word for w in ws], y=norm([w.ig for w in ws]), name="Integrated Gradients", marker_color=PRIMARY)
    if "Occlusion" in methods:
        fig.add_bar(x=[w.word for w in ws], y=norm([w.occlusion for w in ws]), name="Occlusion", marker_color=WARN)
    if "LIME" in methods:
        fig.add_bar(x=[w.word for w in ws], y=norm([w.lime or 0 for w in ws]), name="LIME", marker_color=SAFE)
    _base_layout(fig, 300)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15))
    fig.update_yaxes(title="normalised score")
    return fig


# ------------------------------------------------------------------ about page
def rubric_cards(items: list[tuple[str, int, str, str, list[str]]]) -> str:
    """items: (CO, marks, criterion, where to find it, feature tags)."""
    out = []
    for i, (co, marks, crit, where, tags) in enumerate(items):
        out.append(f'<div class="tg-card hover tg-anim d{i + 1}"><span class="tg-co">{co}</span><span class="tg-marks">{marks} marks</span>'
                   f'<div style="font-weight:700;color:var(--ink);margin:12px 0 6px;font-size:.98rem">{html.escape(crit)}</div>'
                   f'<div class="tg-muted" style="margin-bottom:10px">{html.escape(where)}</div>{chips(tags, "pri")}</div>')
    return f'<div class="tg-rubric">{"".join(out)}</div>'


def architecture(steps: list[tuple[str, str, str]]) -> str:
    return '<div class="tg-arch">' + "".join(
        f'<div class="n" style="animation-delay:{.06 * i:.2f}s"><div class="k">{html.escape(k)}</div>'
        f'<div class="t">{html.escape(t)}</div><div class="d">{html.escape(d)}</div></div>'
        for i, (k, t, d) in enumerate(steps)) + "</div>"


# ------------------------------------------------------------------ display masking
_SLURS = {"nigger", "nigga", "niggers", "faggot", "faggots", "fag", "fags", "retard", "retards", "chink", "spic",
          "kike", "tranny", "cunt", "cunts", "whore", "slut"}


def _offensive_words() -> set[str]:
    from .lexicon import ABUSE_LEXICON

    return _SLURS | {w for w, c in ABUSE_LEXICON.items() if c in ("obscene", "severe_toxic")} | {
        "fuck", "fucking", "fucker", "fucked", "shit", "bitch", "bitches", "dick", "cock", "pussy", "ass", "asshole"}


def mask_offensive(text: str) -> str:
    """Mask slurs / profanity in *display-only* strings such as LDA topic labels and keywords."""
    bad = _offensive_words()
    return re.sub(r"[A-Za-zऀ-ॿ]+",
                  lambda m: (m.group()[0] + "*" * (len(m.group()) - 1)) if m.group().lower() in bad else m.group(), text)
