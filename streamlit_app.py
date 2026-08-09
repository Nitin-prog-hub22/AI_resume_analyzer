"""
app.py
------
Module 7: Streamlit Dashboard.

Main entry point for the AI Resume Analyzer and Job Recommendation
System. Ties together all the other modules:

    resume_parser -> text_cleaner -> skill_extractor
        -> job_matcher -> roadmap_generator

Run with:
    streamlit run app.py
"""

from datetime import datetime
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from resume_parser import extract_resume_text, validate_file, ResumeParseError
from text_cleaner import clean_resume_text
from skill_extractor import load_skill_dictionary, extract_skills, group_by_category
from job_matcher import load_job_roles, match_resume_to_roles, top_n_roles
from roadmap_generator import analyze_skill_gap, generate_roadmap
from section_extractor import extract_sections
from pdf_report_generator import generate_pdf_report
from ai_feedback import get_ai_feedback, get_configured_provider, AIFeedbackError
from database import save_analysis, get_history, clear_history

# Advanced Approach modules (Section 9 of the guide) — optional. The app
# runs fully on the Beginner approach without these; they're only offered
# as a sidebar toggle when actually installed and ready to use.
try:
    from advanced_skill_extractor import extract_skills_spacy, SpacyModelNotFound, SPACY_INSTALLED
except ImportError:
    SPACY_INSTALLED = False

try:
    from semantic_matcher import match_resume_to_roles_semantic, SENTENCE_TRANSFORMERS_INSTALLED
except ImportError:
    SENTENCE_TRANSFORMERS_INSTALLED = False

st.set_page_config(
    page_title="AI Resume Analyzer & Job Recommender",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
# Theme: "route map" — a resume is read as a journey from where a
# candidate stands today to the role they're aiming for. Ink-navy control
# panel (sidebar) + parchment map (canvas) + a brass/gold way-marker
# accent used for scores, chips, and the roadmap timeline.

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --ink: #16233F;
    --ink-soft: #223257;
    --parchment: #F6F3EC;
    --surface: #FFFFFF;
    --gold: #C89B3C;
    --gold-soft: #EFE1BE;
    --slate: #5B6B82;
    --success: #3F7D58;
    --success-soft: #DCEBE1;
    --danger: #B5555C;
    --danger-soft: #F3DEDF;
    --line: #E4DECD;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- App canvas ---------- */
[data-testid="stAppViewContainer"] {
    background: var(--parchment);
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2rem;
    max-width: 1180px;
}

/* ---------- Sidebar / control panel ---------- */
[data-testid="stSidebar"] {
    background: #193B3D;
}
[data-testid="stSidebar"] * {
    color: #000000 !important;
}
/* Make normal text slightly stronger */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #000000 !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-size: 0.85rem !important;
    color: var(--gold) !important;
    border-bottom: 1px solid var(--ink-soft);
    padding-bottom: 0.4rem;
    margin-top: 1.2rem;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: var(--ink-soft);
    border: 1px dashed #48588A;
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--ink-soft);
    border-color: #48588A;
    border-radius: 8px;
}
[data-testid="stSidebar"] hr {
    border-color: var(--ink-soft);
}
/* ---------- Privacy text ---------- */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #AEB8CC !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #AEB8CC !important;
}
/* ---------- Headings on the main canvas ---------- */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
}
h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
}
[data-testid="stCaptionContainer"] {
    color: var(--slate) !important;
}

/* ---------- Card containers (targeted via st.container(key=...)) ---------- */
div[class*="st-key-"] {
    background: var(--surface) !important;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
div[class*="st-key-hero"] {
    background: linear-gradient(135deg, var(--ink) 0%, #1E2E52 100%) !important;
    border: none !important;
    padding: 1.8rem 2rem;
}
.st-key-hero h1 { color: #FFFFFF !important; }
.st-key-hero p, .st-key-hero [data-testid="stCaptionContainer"] { color: #C9D0E0 !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    background: var(--gold) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #B5872F !important;
}

/* ---------- Skill pill chips ---------- */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0 1rem 0; }
.chip {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}
.chip-found { background: var(--success-soft); color: var(--success); border: 1px solid #BFE0CC; }
.chip-missing { background: var(--danger-soft); color: var(--danger); border: 1px solid #E6C2C4; }
.category-label {
    font-family: 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
    color: var(--slate);
    border-bottom: 2px solid var(--gold-soft);
    display: inline-block;
    padding-bottom: 2px;
    margin-top: 0.6rem;
}

/* ---------- Score ring (signature element) ---------- */
.score-wrap { display: flex; align-items: center; gap: 1.6rem; }
.score-ring {
    width: 132px; height: 132px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.score-ring-inner {
    width: 104px; height: 104px; border-radius: 50%;
    background: var(--surface);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.score-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem; font-weight: 700; color: var(--ink);
}
.score-label {
    font-size: 0.65rem; color: var(--slate); text-transform: uppercase; letter-spacing: 0.04em;
}

/* ---------- Roadmap timeline ---------- */
.timeline { border-left: 2px dashed var(--gold); margin-left: 14px; padding-left: 24px; }
.timeline-item { position: relative; padding-bottom: 1.3rem; }
.timeline-item:last-child { padding-bottom: 0; }
.timeline-dot {
    position: absolute; left: -31px; top: 2px;
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--gold); border: 3px solid var(--parchment);
}
.timeline-week {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: var(--gold); text-transform: uppercase; letter-spacing: 0.05em;
}
.timeline-skill {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; color: var(--ink); font-size: 1rem;
}
.timeline-focus { color: var(--slate); font-size: 0.88rem; margin-top: 0.1rem; }

/* ---------- Recommended role rows ---------- */
.role-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.55rem 0; border-bottom: 1px solid var(--line);
}
.role-row:last-child { border-bottom: none; }
.role-rank {
    font-family: 'JetBrains Mono', monospace; font-weight: 700;
    color: var(--gold); width: 22px;
}
.role-name { flex: 1; font-weight: 500; color: var(--ink); }
.role-score { font-family: 'JetBrains Mono', monospace; color: var(--slate); }

/* ---------- Misc ---------- */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    color: var(--ink);
}
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def score_ring_html(score: float) -> str:
    """Build a CSS conic-gradient ring showing the match score visually."""
    degrees = max(0, min(100, score)) * 3.6
    return (
        f'<div class="score-ring" style="background: conic-gradient(var(--gold) {degrees}deg, var(--gold-soft) {degrees}deg);">'
        f'<div class="score-ring-inner">'
        f'<span class="score-value">{score:.0f}%</span>'
        f'<span class="score-label">Match</span>'
        f'</div></div>'
    )


def chip_row_html(items: list[str], kind: str) -> str:
    """Render a list of skills as pill-shaped badges. kind: 'found' | 'missing'."""
    if not items:
        return '<p style="color:var(--slate); font-size:0.9rem;">None</p>'
    css_class = "chip-found" if kind == "found" else "chip-missing"
    chips = "".join(f'<span class="chip {css_class}">{s}</span>' for s in items)
    return f'<div class="chip-row">{chips}</div>'


def generate_html_report(
    resume_filename: str,
    target_role: str,
    target_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
    top_matches_df: pd.DataFrame,
    roadmap: list[dict],
) -> str:
    """Build a standalone, self-styled HTML report (opens in any browser, prints cleanly to PDF)."""
    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    degrees = max(0, min(100, target_score)) * 3.6

    matched_chips = "".join(f'<span class="chip found">{s}</span>' for s in matched_skills) or '<span class="muted">None</span>'
    missing_chips = "".join(f'<span class="chip missing">{s}</span>' for s in missing_skills) or '<span class="muted">None &#127881;</span>'

    top_rows = "".join(
        f'<div class="role-row"><span class="rank">{i+1:02d}</span>'
        f'<span class="rname">{row.job_role}</span>'
        f'<span class="rscore">{row.match_score:.1f}%</span></div>'
        for i, row in enumerate(top_matches_df.itertuples())
    )

    if roadmap:
        timeline_items = "".join(
            f'<div class="t-item"><div class="t-dot"></div>'
            f'<div class="t-week">Week {r["week"]}</div>'
            f'<div class="t-skill">{r["skill"]}</div>'
            f'<div class="t-focus">{r["focus"]}</div></div>'
            for r in roadmap
        )
        roadmap_html = f'<div class="timeline">{timeline_items}</div>'
    else:
        roadmap_html = '<p class="muted">You already match all required skills for this role.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Resume Analysis Report — {resume_filename}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');
:root {{
    --ink: #16233F; --gold: #C89B3C; --gold-soft: #EFE1BE; --parchment: #F6F3EC;
    --surface: #FFFFFF; --slate: #5B6B82; --success: #3F7D58; --success-soft: #DCEBE1;
    --danger: #B5555C; --danger-soft: #F3DEDF; --line: #E4DECD;
}}
* {{ box-sizing: border-box; }}
body {{ font-family: 'Inter', sans-serif; background: var(--parchment); color: var(--ink); margin: 0; padding: 2.5rem 1.5rem; }}
.page {{ max-width: 760px; margin: 0 auto; }}
.hero {{ background: linear-gradient(135deg, var(--ink) 0%, #1E2E52 100%); border-radius: 16px; padding: 2rem 2.2rem; color: #fff; }}
.hero h1 {{ font-family: 'Space Grotesk', sans-serif; margin: 0.2rem 0 0.3rem 0; font-size: 1.6rem; }}
.hero .meta {{ color: #C9D0E0; font-size: 0.88rem; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1.5rem 1.8rem; margin-top: 1.1rem; }}
h2 {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; margin-top: 0; color: var(--ink); }}
.muted {{ color: var(--slate); font-size: 0.9rem; }}
.score-wrap {{ display: flex; align-items: center; gap: 1.6rem; flex-wrap: wrap; }}
.score-ring {{ width: 118px; height: 118px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    background: conic-gradient(var(--gold) {degrees}deg, var(--gold-soft) {degrees}deg); flex-shrink: 0; }}
.score-ring-inner {{ width: 92px; height: 92px; border-radius: 50%; background: var(--surface); display: flex; flex-direction: column; align-items: center; justify-content: center; }}
.score-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; }}
.score-label {{ font-size: 0.62rem; color: var(--slate); text-transform: uppercase; letter-spacing: 0.05em; }}
.label {{ font-family: 'Space Grotesk', sans-serif; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.72rem;
    color: var(--slate); border-bottom: 2px solid var(--gold-soft); display: inline-block; padding-bottom: 2px; margin: 0.9rem 0 0.5rem 0; }}
.chip-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
.chip {{ display: inline-block; padding: 0.28rem 0.75rem; border-radius: 999px; font-size: 0.82rem; font-weight: 500; }}
.chip.found {{ background: var(--success-soft); color: var(--success); border: 1px solid #BFE0CC; }}
.chip.missing {{ background: var(--danger-soft); color: var(--danger); border: 1px solid #E6C2C4; }}
.role-row {{ display: flex; align-items: center; gap: 1rem; padding: 0.55rem 0; border-bottom: 1px solid var(--line); }}
.role-row:last-child {{ border-bottom: none; }}
.rank {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--gold); width: 22px; }}
.rname {{ flex: 1; font-weight: 500; }}
.rscore {{ font-family: 'JetBrains Mono', monospace; color: var(--slate); }}
.timeline {{ border-left: 2px dashed var(--gold); margin-left: 12px; padding-left: 22px; }}
.t-item {{ position: relative; padding-bottom: 1.2rem; }}
.t-item:last-child {{ padding-bottom: 0; }}
.t-dot {{ position: absolute; left: -29px; top: 2px; width: 14px; height: 14px; border-radius: 50%; background: var(--gold); border: 3px solid var(--parchment); }}
.t-week {{ font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--gold); text-transform: uppercase; letter-spacing: 0.05em; }}
.t-skill {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; }}
.t-focus {{ color: var(--slate); font-size: 0.86rem; }}
.footer-note {{ text-align: center; color: var(--slate); font-size: 0.78rem; margin-top: 1.5rem; }}
@media print {{ body {{ padding: 0; background: #fff; }} .card, .hero {{ box-shadow: none; }} }}
</style>
</head>
<body>
<div class="page">
    <div class="hero">
        <div style="font-size:1.8rem;">&#129517;</div>
        <h1>AI Resume Analyzer Report</h1>
        <div class="meta">Generated {generated_at} &nbsp;&bull;&nbsp; Resume: {resume_filename}</div>
    </div>

    <div class="card">
        <h2>&#127919; Match Score &mdash; {target_role}</h2>
        <div class="score-wrap">
            <div class="score-ring"><div class="score-ring-inner">
                <span class="score-value">{target_score:.0f}%</span>
                <span class="score-label">Match</span>
            </div></div>
            <div>
                <div class="label">Skills Found</div>
                <div class="chip-row">{matched_chips}</div>
                <div class="label">Missing Skills</div>
                <div class="chip-row">{missing_chips}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>&#127942; Top Recommended Roles</h2>
        {top_rows}
    </div>

    <div class="card">
        <h2>&#128506;&#65039; Suggested Learning Roadmap</h2>
        {roadmap_html}
    </div>

    <p class="footer-note">Match scores are estimates based on keyword and TF-IDF similarity.
    This report is for guidance only and does not represent a hiring or rejection decision.</p>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Cached data loaders (skill dictionary & job roles rarely change per session)
# ---------------------------------------------------------------------------


@st.cache_data
def get_skill_dictionary():
    return load_skill_dictionary()


@st.cache_data
def get_job_roles():
    return load_job_roles()


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

with st.container(key="hero"):
    st.title("🧭 AI Resume Analyzer and Job Recommendation System")
    st.caption(
        "Upload your resume to see a match score, missing skills, and a simple "
        "learning roadmap for your target role. This tool is for guidance only "
        "— it does not make hiring or rejection decisions."
    )

# ---------------------------------------------------------------------------
# Sidebar: upload + target role selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Control Panel")
    st.markdown("### 1. Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX resume", type=["pdf", "docx"], label_visibility="collapsed"
    )

    job_roles_df = get_job_roles()
    st.markdown("### 2. Select Target Role")
    target_role = st.selectbox(
        "Which role are you aiming for?",
        job_roles_df["job_role"].tolist(),
        label_visibility="collapsed",
    )
    st.markdown("### 3. Advanced Approach (Optional)")
    skill_mode = st.radio(
        "Skill extraction",
        ["Keyword (Beginner)", "spaCy (Advanced)"],
        disabled=not SPACY_INSTALLED,
        help=None if SPACY_INSTALLED else "Install spaCy + a language model to enable this.",
    )
    match_mode = st.radio(
        "Matching method",
        ["TF-IDF (Beginner)", "Sentence Transformers (Advanced)"],
        disabled=not SENTENCE_TRANSFORMERS_INSTALLED,
        help=None if SENTENCE_TRANSFORMERS_INSTALLED else "Install sentence-transformers to enable this.",
    )
    if not SPACY_INSTALLED or not SENTENCE_TRANSFORMERS_INSTALLED:
        st.caption("Advanced options need extra packages — see requirements.txt.")

    st.divider()
    st.caption(
        "🔒 Your resume is processed only in this session and is not "
        "stored permanently."
    )

# ---------------------------------------------------------------------------
# Main analysis flow
# ---------------------------------------------------------------------------

if uploaded_file is None:
    st.info("👈 Upload a resume from the sidebar to get started.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    validate_file(uploaded_file.name, len(file_bytes))
    raw_text = extract_resume_text(uploaded_file.name, file_bytes)
except ResumeParseError as e:
    st.error(f"⚠️ {e}")
    st.stop()

cleaned_text = clean_resume_text(raw_text)
resume_sections = extract_sections(raw_text)

skill_dict = get_skill_dictionary()
active_skill_mode = "Keyword (Beginner)"
if skill_mode == "spaCy (Advanced)":
    try:
        found_skills = extract_skills_spacy(cleaned_text, skill_dict)
        active_skill_mode = "spaCy (Advanced)"
    except (RuntimeError, SpacyModelNotFound) as e:
        st.warning(f"⚠️ spaCy extraction unavailable ({e}) — using keyword matching instead.")
        found_skills = extract_skills(cleaned_text, skill_dict)
else:
    found_skills = extract_skills(cleaned_text, skill_dict)

found_skill_names = [s["skill"] for s in found_skills]
grouped_skills = group_by_category(found_skills)

active_match_mode = "TF-IDF (Beginner)"
if match_mode == "Sentence Transformers (Advanced)":
    try:
        match_df = match_resume_to_roles_semantic(cleaned_text, job_roles_df)
        active_match_mode = "Sentence Transformers (Advanced)"
    except RuntimeError as e:
        st.warning(f"⚠️ Sentence Transformers unavailable ({e}) — using TF-IDF instead.")
        match_df = match_resume_to_roles(cleaned_text, job_roles_df)
else:
    match_df = match_resume_to_roles(cleaned_text, job_roles_df)

top_matches = top_n_roles(match_df, 3)

target_row = job_roles_df[job_roles_df["job_role"] == target_role].iloc[0]
target_score = match_df.loc[match_df["job_role"] == target_role, "match_score"].values[0]
gap = analyze_skill_gap(found_skill_names, target_row["skills_list"])
roadmap = generate_roadmap(gap["missing"])

st.success(f"✅ Resume processed: **{uploaded_file.name}**")
st.caption(f"Skill extraction: **{active_skill_mode}** · Matching: **{active_match_mode}**")

# --- Section: resume sections (education / experience / projects) ----------
with st.container(key="sections-card"):
    st.markdown("### 🧾 Resume Sections")
    sec_col1, sec_col2, sec_col3 = st.columns(3)
    section_meta = [
        (sec_col1, "education", "🎓 Education"),
        (sec_col2, "experience", "💼 Experience"),
        (sec_col3, "projects", "🛠️ Projects"),
    ]
    for col, key, label in section_meta:
        with col:
            st.markdown(f'<div class="category-label">{label}</div>', unsafe_allow_html=True)
            content = resume_sections.get(key, "")
            if content:
                st.markdown(
                    f'<div style="font-size:0.85rem; color:var(--ink); white-space:pre-wrap;">{html.escape(content)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="color:var(--slate); font-size:0.85rem;">Not detected in this resume.</p>',
                    unsafe_allow_html=True,
                )
    st.caption(
        "Detected using resume heading patterns (e.g. \"Education\", \"Experience\", "
        "\"Projects\"). Unusual layouts or missing headings may not be picked up."
    )

# --- Section: extracted skills -------------------------------------------
with st.container(key="skills-card"):
    st.markdown("### 📋 Extracted Skills")
    if grouped_skills:
        for category, skills in grouped_skills.items():
            st.markdown(
                f'<div class="category-label">{category.replace("_", " ").title()}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(chip_row_html(skills, "found"), unsafe_allow_html=True)
    else:
        st.warning("No known skills were detected. Try a different resume or update the skill dictionary.")

# --- Section: match score for target role ---------------------------------
with st.container(key="score-card"):
    st.markdown(f"### 🎯 Match Score — {target_role}")
    ring_col, detail_col = st.columns([1, 2.2])
    with ring_col:
        st.markdown(score_ring_html(target_score), unsafe_allow_html=True)
    with detail_col:
        st.markdown('<div class="category-label">Skills Found</div>', unsafe_allow_html=True)
        st.markdown(chip_row_html(gap["matched"], "found"), unsafe_allow_html=True)
        st.markdown('<div class="category-label">Missing Skills</div>', unsafe_allow_html=True)
        st.markdown(chip_row_html(gap["missing"], "missing"), unsafe_allow_html=True)

# --- Section: recommended roles chart --------------------------------------
with st.container(key="roles-card"):
    st.markdown("### 🏆 Recommended Roles")

    chart_df = match_df.head(5).sort_values("match_score")
    bar_colors = ["#C89B3C" if s == chart_df["match_score"].max() else "#D8D0B8" for s in chart_df["match_score"]]

    fig = go.Figure(
        go.Bar(
            x=chart_df["match_score"],
            y=chart_df["job_role"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=chart_df["match_score"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(family="JetBrains Mono", color="#16233F"),
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#5B6B82"),
        xaxis=dict(range=[0, 105], showgrid=False, title="Match Score (%)"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="category-label">Top 3</div>', unsafe_allow_html=True)
    rows_html = "".join(
        f'<div class="role-row"><span class="role-rank">{i+1:02d}</span>'
        f'<span class="role-name">{row.job_role}</span>'
        f'<span class="role-score">{row.match_score:.1f}%</span></div>'
        for i, row in enumerate(top_matches.itertuples())
    )
    st.markdown(rows_html, unsafe_allow_html=True)

# --- Section: learning roadmap ----------------------------------------------
with st.container(key="roadmap-card"):
    st.markdown("### 🗺️ Suggested Learning Roadmap")
    if roadmap:
        items_html = "".join(
            f'<div class="timeline-item">'
            f'<div class="timeline-dot"></div>'
            f'<div class="timeline-week">Week {r["week"]}</div>'
            f'<div class="timeline-skill">{r["skill"]}</div>'
            f'<div class="timeline-focus">{r["focus"]}</div>'
            f'</div>'
            for r in roadmap
        )
        st.markdown(f'<div class="timeline">{items_html}</div>', unsafe_allow_html=True)
    else:
        st.success("You already match all required skills for this role! 🎉")

# --- Section: optional AI-generated feedback --------------------------------
with st.container(key="ai-card"):
    st.markdown("### 🤖 AI-Generated Feedback (Optional)")
    configured_provider = get_configured_provider()

    if "ai_feedback_text" not in st.session_state:
        st.session_state.ai_feedback_text = ""

    if configured_provider is None:
        st.info(
            "This feature is optional and off by default. To enable it, add "
            "`GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY` to your `.env` "
            "file, then restart the app."
        )
    else:
        st.caption(f"Provider configured: **{configured_provider}**")
        if st.button("✨ Generate personalized feedback", use_container_width=False):
            with st.spinner("Asking the AI for feedback..."):
                try:
                    st.session_state.ai_feedback_text = get_ai_feedback(
                        resume_skills=found_skill_names,
                        missing_skills=gap["missing"],
                        target_role=target_role,
                        match_score=target_score,
                        provider=configured_provider,
                    )
                except AIFeedbackError as e:
                    st.error(f"⚠️ {e}")

        if st.session_state.ai_feedback_text:
            st.markdown(st.session_state.ai_feedback_text)

# --- Section: downloadable report ------------------------------------------
with st.container(key="report-card"):
    st.markdown("### 📥 Download Report")

    ai_feedback_text = st.session_state.get("ai_feedback_text", "")

    pdf_report = generate_pdf_report(
        resume_filename=uploaded_file.name,
        target_role=target_role,
        target_score=target_score,
        matched_skills=gap["matched"],
        missing_skills=gap["missing"],
        top_matches_df=top_matches,
        roadmap=roadmap,
        ai_feedback=ai_feedback_text,
    )

    html_report = generate_html_report(
        resume_filename=uploaded_file.name,
        target_role=target_role,
        target_score=target_score,
        matched_skills=gap["matched"],
        missing_skills=gap["missing"],
        top_matches_df=top_matches,
        roadmap=roadmap,
    )

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            label="⬇ Download PDF (.pdf)",
            data=pdf_report,
            file_name=f"resume_analysis_{uploaded_file.name.split('.')[0]}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
 
    with dl_col3:
        report_lines = [
            "AI Resume Analyzer Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Resume file: {uploaded_file.name}",
            "",
            f"Target Role: {target_role}",
            f"Match Score: {target_score:.1f}%",
            "",
            "Skills Found:",
        ] + [f"  - {s}" for s in found_skill_names] + [
            "",
            "Missing Skills:",
        ] + [f"  - {s}" for s in gap["missing"]] + [
            "",
            "Top Recommended Roles:",
        ] + [
            f"  {i+1}. {row.job_role} - {row.match_score:.1f}%"
            for i, row in enumerate(top_matches.itertuples())
        ] + [
            "",
            "Suggested Roadmap:",
        ] + [f"  Week {r['week']}: {r['skill']} - {r['focus']}" for r in roadmap] + (
            ["", "AI-Generated Feedback:", ai_feedback_text] if ai_feedback_text else []
        )

       

    st.caption(
        "The PDF is ready to submit as-is. The HTML report opens in any "
        "browser and can also be converted to PDF via Print → Save as PDF."
    )
    st.caption(
        "Note: match scores are estimates based on keyword and TF-IDF "
        "similarity. They do not evaluate soft skills, seniority, or "
        "context, and should be used as guidance only."
    )
# --- Section: analysis history (SQLite database) ----------------------------
with st.container(key="history-card"):
    st.markdown("### 💾 Analysis History (Optional)")
    st.caption(
        "Saving is opt-in — nothing is stored automatically. Only the "
        "score and skill lists are saved, never the resume file itself."
    )

    hist_col1, hist_col2 = st.columns([1, 1])
    with hist_col1:
        if st.button("💾 Save this analysis to history", use_container_width=True):
            save_analysis(
                resume_filename=uploaded_file.name,
                target_role=target_role,
                match_score=float(target_score),
                matched_skills=gap["matched"],
                missing_skills=gap["missing"],
            )
            st.success("Saved to local history database.")
    with hist_col2:
        if st.button("🗑️ Clear history", use_container_width=True):
            clear_history()
            st.info("History cleared.")

    history_records = get_history(limit=10)
    if history_records:
        history_df = pd.DataFrame(history_records)[
            ["created_at", "resume_filename", "target_role", "match_score"]
        ].rename(columns={
            "created_at": "When", "resume_filename": "Resume",
            "target_role": "Target Role", "match_score": "Score (%)",
        })
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No saved analyses yet.")
