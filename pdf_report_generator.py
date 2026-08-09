"""
pdf_report_generator.py
------------------------
Optional Advanced Feature: Downloadable PDF analysis report.

Builds a single polished PDF that mirrors the dashboard's visual theme
(navy hero band, gold accents, skill chips, a match-score donut, and a
roadmap timeline) using reportlab directly on a canvas. so it installs
cleanly with `pip install -r requirements.txt` on any machine.
"""

from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Design tokens (mirrors the Streamlit dashboard's CSS variables)
# ---------------------------------------------------------------------------
INK = colors.HexColor("#16233F")
GOLD = colors.HexColor("#C89B3C")
GOLD_SOFT = colors.HexColor("#EFE1BE")
PARCHMENT = colors.HexColor("#F6F3EC")
SURFACE = colors.white
SLATE = colors.HexColor("#5B6B82")
SUCCESS = colors.HexColor("#3F7D58")
SUCCESS_SOFT = colors.HexColor("#DCEBE1")
SUCCESS_BORDER = colors.HexColor("#BFE0CC")
DANGER = colors.HexColor("#B5555C")
DANGER_SOFT = colors.HexColor("#F3DEDF")
DANGER_BORDER = colors.HexColor("#E6C2C4")
LINE = colors.HexColor("#E4DECD")

MARGIN = 42
PAGE_W, PAGE_H = A4


def _new_page(c: canvas.Canvas) -> float:
    """Start a fresh page with a parchment background; return the top y."""
    c.showPage()
    c.setFillColor(PARCHMENT)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    return PAGE_H - MARGIN


def _ensure_space(c: canvas.Canvas, y: float, needed: float) -> float:
    """Start a new page if the remaining space is too small."""
    if y - needed < MARGIN:
        return _new_page(c)
    return y


def _wrap_text(c: canvas.Canvas, text: str, font: str, size: float, max_width: float) -> list:
    """Simple greedy word-wrap based on measured string width."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if c.stringWidth(trial, font, size) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _draw_chips(c: canvas.Canvas, x: float, y: float, max_x: float, items: list, fill, border, text_color) -> float:
    """Draw wrapping pill-shaped chips; returns the y below the last row."""
    if not items:
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(SLATE)
        c.drawString(x, y - 12, "None")
        return y - 20

    font, size = "Helvetica", 9
    pad_x, chip_h, gap_x, gap_y = 9, 18, 6, 8
    cur_x, cur_y = x, y - chip_h

    for item in items:
        chip_w = c.stringWidth(item, font, size) + pad_x * 2
        if cur_x + chip_w > max_x:
            cur_x = x
            cur_y -= chip_h + gap_y
        c.setFillColor(fill)
        c.setStrokeColor(border)
        c.roundRect(cur_x, cur_y, chip_w, chip_h, chip_h / 2, stroke=1, fill=1)
        c.setFillColor(text_color)
        c.setFont(font, size)
        c.drawString(cur_x + pad_x, cur_y + 5.5, item)
        cur_x += chip_w + gap_x

    return cur_y - gap_y


def _draw_score_ring(c: canvas.Canvas, cx: float, cy: float, r: float, pct: float) -> None:
    """Draw a donut-style progress ring showing the match score, gold on soft-gold track."""
    pct = max(0, min(100, pct))
    c.setFillColor(GOLD_SOFT)
    c.circle(cx, cy, r, stroke=0, fill=1)
    if pct > 0:
        c.setFillColor(GOLD)
        c.wedge(cx - r, cy - r, cx + r, cy + r, 90, -pct * 3.6, stroke=0, fill=1)
    c.setFillColor(SURFACE)
    c.circle(cx, cy, r * 0.7, stroke=0, fill=1)

    label = f"{pct:.0f}%"
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 17)
    c.drawCentredString(cx, cy - 5, label)
    c.setFillColor(SLATE)
    c.setFont("Helvetica", 6.5)
    c.drawCentredString(cx, cy - 16, "MATCH")


def _card_top(c: canvas.Canvas, x: float, y: float, w: float, title: str) -> float:
    """Draw a card's title; returns the y just below the title."""
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 12.5)
    c.drawString(x + 18, y - 24, title)
    return y - 40


def generate_pdf_report(
    resume_filename: str,
    target_role: str,
    target_score: float,
    matched_skills: list,
    missing_skills: list,
    top_matches_df,
    roadmap: list,
    ai_feedback: str = "",
) -> bytes:
    """
    Build a complete, styled PDF analysis report.

    Args mirror the dashboard's session data: matched/missing skills for
    the chosen target role, the ranked top-N job matches DataFrame, the
    week-by-week roadmap list from roadmap_generator.generate_roadmap,
    and optionally a block of AI-generated feedback text.

    Returns:
        Raw PDF bytes, ready to hand to st.download_button.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    content_w = PAGE_W - 2 * MARGIN
    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    c.setFillColor(PARCHMENT)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    y = PAGE_H - MARGIN

    # ---- Hero band --------------------------------------------------
    hero_h = 78
    c.setFillColor(INK)
    c.roundRect(MARGIN, y - hero_h, content_w, hero_h, 12, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN + 20, y - 32, "AI Resume Analyzer Report")
    c.setFont("Helvetica", 9.5)
    c.setFillColor(colors.HexColor("#C9D0E0"))
    c.drawString(MARGIN + 20, y - 50, f"Generated {generated_at}  |  Resume: {resume_filename}")
    y -= hero_h + 18

    # ---- Score card ---------------------------------------------------
    card_h = 150
    y = _ensure_space(c, y, card_h + 16)
    c.setFillColor(SURFACE)
    c.setStrokeColor(LINE)
    c.roundRect(MARGIN, y - card_h, content_w, card_h, 12, stroke=1, fill=1)
    inner_y = _card_top(c, MARGIN, y, content_w, f"Match Score - {target_role}")

    ring_cx, ring_cy, ring_r = MARGIN + 55, inner_y - 45, 42
    _draw_score_ring(c, ring_cx, ring_cy, ring_r, target_score)

    detail_x = MARGIN + 125
    detail_max_x = MARGIN + content_w - 18
    c.setFillColor(SLATE)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(detail_x, inner_y - 4, "SKILLS FOUND")
    chip_y = _draw_chips(c, detail_x, inner_y - 10, detail_max_x, matched_skills, SUCCESS_SOFT, SUCCESS_BORDER, SUCCESS)

    c.setFillColor(SLATE)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(detail_x, chip_y - 8, "MISSING SKILLS")
    _draw_chips(c, detail_x, chip_y - 14, detail_max_x, missing_skills, DANGER_SOFT, DANGER_BORDER, DANGER)

    y -= card_h + 16

    # ---- Top recommended roles card ------------------------------------
    rows = list(top_matches_df.itertuples())
    roles_card_h = 40 + 24 * len(rows) + 10
    y = _ensure_space(c, y, roles_card_h + 16)
    c.setFillColor(SURFACE)
    c.setStrokeColor(LINE)
    c.roundRect(MARGIN, y - roles_card_h, content_w, roles_card_h, 12, stroke=1, fill=1)
    inner_y = _card_top(c, MARGIN, y, content_w, "Top Recommended Roles")

    row_y = inner_y
    for i, row in enumerate(rows):
        c.setFillColor(GOLD)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(MARGIN + 18, row_y - 12, f"{i + 1:02d}")
        c.setFillColor(INK)
        c.setFont("Helvetica", 10.5)
        c.drawString(MARGIN + 45, row_y - 12, str(row.job_role))
        c.setFillColor(SLATE)
        c.setFont("Helvetica", 10)
        c.drawRightString(MARGIN + content_w - 18, row_y - 12, f"{row.match_score:.1f}%")
        if i < len(rows) - 1:
            c.setStrokeColor(LINE)
            c.line(MARGIN + 18, row_y - 20, MARGIN + content_w - 18, row_y - 20)
        row_y -= 24

    y -= roles_card_h + 16

    # ---- Roadmap card ---------------------------------------------------
    focus_font, focus_size = "Helvetica", 9
    focus_max_w = content_w - 70
    item_blocks = []
    for r in roadmap:
        focus_lines = _wrap_text(c, r["focus"], focus_font, focus_size, focus_max_w)
        item_blocks.append((r, focus_lines))

    roadmap_h = 40 + sum(38 + (len(fl) - 1) * 12 for _, fl in item_blocks) + 14 if roadmap else 60

    y = _ensure_space(c, y, roadmap_h + 16)
    c.setFillColor(SURFACE)
    c.setStrokeColor(LINE)
    c.roundRect(MARGIN, y - roadmap_h, content_w, roadmap_h, 12, stroke=1, fill=1)
    inner_y = _card_top(c, MARGIN, y, content_w, "Suggested Learning Roadmap")

    if not roadmap:
        c.setFillColor(SLATE)
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(MARGIN + 18, inner_y - 8, "You already match all required skills for this role.")
    else:
        line_x = MARGIN + 24
        item_y = inner_y
        last_item = item_blocks[-1][0]
        for r, focus_lines in item_blocks:
            c.setFillColor(GOLD)
            c.circle(line_x, item_y - 4, 3.2, stroke=0, fill=1)

            c.setFillColor(GOLD)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(line_x + 14, item_y, f"WEEK {r['week']}")

            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 10.5)
            c.drawString(line_x + 14, item_y - 13, str(r["skill"]))

            c.setFillColor(SLATE)
            c.setFont(focus_font, focus_size)
            fy = item_y - 26
            for line in focus_lines:
                c.drawString(line_x + 14, fy, line)
                fy -= 12

            block_h = 38 + (len(focus_lines) - 1) * 12
            if r is not last_item:
                c.setStrokeColor(GOLD)
                c.setDash(2, 2)
                c.line(line_x, item_y - 6, line_x, item_y - block_h + 2)
                c.setDash()
            item_y -= block_h

    y -= roadmap_h + 16

    # ---- Optional AI feedback card --------------------------------------
    if ai_feedback:
        fb_font, fb_size = "Helvetica", 9.5
        fb_max_w = content_w - 36
        fb_lines = []
        for para in ai_feedback.split("\n"):
            if para.strip():
                fb_lines.extend(_wrap_text(c, para.strip(), fb_font, fb_size, fb_max_w))
                fb_lines.append("")
            else:
                fb_lines.append("")

        fb_card_h = 40 + len(fb_lines) * 13 + 10
        y = _ensure_space(c, y, fb_card_h + 16)
        c.setFillColor(SURFACE)
        c.setStrokeColor(LINE)
        c.roundRect(MARGIN, y - fb_card_h, content_w, fb_card_h, 12, stroke=1, fill=1)
        inner_y = _card_top(c, MARGIN, y, content_w, "AI-Generated Feedback")

        c.setFillColor(INK)
        c.setFont(fb_font, fb_size)
        fy = inner_y - 6
        for line in fb_lines:
            fy = _ensure_space(c, fy, 20)
            c.drawString(MARGIN + 18, fy, line)
            fy -= 13
        y = fy - 10

    # ---- Footer -----------------------------------------------------
    y = _ensure_space(c, y, 30)
    c.setFillColor(SLATE)
    c.setFont("Helvetica-Oblique", 8)
    footer = (
        "Match scores are estimates based on keyword and TF-IDF similarity. "
        "This report is for guidance only and does not represent a hiring or rejection decision."
    )
    c.drawCentredString(PAGE_W / 2, y - 10, footer)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()
