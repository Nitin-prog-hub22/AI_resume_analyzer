"""
section_extractor.py
----------------------
resume section detection for Education, Experience,
and Projects — completing the workflow step "Identify skills, education,
projects, and experience" (skills are handled separately by
skill_extractor.py, which works on the fully cleaned text).

This module works on the RAW extracted text (before text_cleaner
collapses everything to a single lowercase line), because section
detection depends on line breaks and heading-style formatting that
cleaning intentionally destroys.

Approach: scan line by line for short lines that look like section
headings (e.g. "Education", "Work Experience:", "PROJECTS"). Content
between a recognized heading and the next heading is captured as that
section's text. This is a simple heuristic — it won't catch every
resume layout (e.g. heavily columnar PDFs), which is a documented
limitation consistent with the project's "beginner approach".
"""

import re

TARGET_HEADER_SYNONYMS = {
    "education": {
        "education", "academics", "academic background",
        "academic qualifications", "educational qualifications",
    },
    "experience": {
        "experience", "work experience", "professional experience",
        "employment history", "work history", "professional background",
        "internship experience", "internships",
    },
    "projects": {
        "projects", "academic projects", "personal projects",
        "key projects", "project experience", "major projects",
    },
}

# Headings that mark the end of a target section but aren't extracted
# themselves — without these, a "Skills" or "Certifications" section
# right after "Experience" would get swallowed into it.
BOUNDARY_ONLY_HEADERS = {
    "skills", "technical skills", "core skills", "summary", "objective",
    "profile", "about", "about me", "certifications", "certificates",
    "achievements", "awards", "hobbies", "interests", "references",
    "contact", "contact information", "personal details", "declaration",
    "languages", "extracurricular activities",
}


def _normalize_header(line: str) -> str:
    """Strip punctuation/digits and collapse whitespace for header matching."""
    text = line.strip().lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match_header(norm_line: str) -> str | None:
    """Return the matched target section name, 'boundary', or None."""
    if not norm_line or len(norm_line.split()) > 5:
        return None
    for section, synonyms in TARGET_HEADER_SYNONYMS.items():
        if norm_line in synonyms:
            return section
    if norm_line in BOUNDARY_ONLY_HEADERS:
        return "boundary"
    return None


def extract_sections(raw_text: str) -> dict[str, str]:
    """
    Scan raw resume text for Education, Experience, and Projects sections.

    Args:
        raw_text: unprocessed text straight from resume_parser
            (must still have its original line breaks).

    Returns:
        dict with keys "education", "experience", "projects" mapping to
        the extracted section text (empty string if not detected).
    """
    lines = raw_text.splitlines()
    buffers: dict[str, list[str]] = {"education": [], "experience": [], "projects": []}
    current: str | None = None

    for line in lines:
        norm = _normalize_header(line)
        match = _match_header(norm)

        if match == "boundary":
            current = None
            continue
        if match in buffers:
            current = match
            continue
        if current and line.strip():
            buffers[current].append(line.strip())

    return {section: "\n".join(content).strip() for section, content in buffers.items()}


if __name__ == "__main__":
    sample = """
    John Doe
    Software Engineer

    Education
    B.Tech in Computer Science, XYZ University, 2022

    Skills
    Python, SQL, Machine Learning

    Experience
    Software Engineer Intern, ABC Corp, 2021-2022
    Built internal tools using Python and Flask.

    Projects
    Resume Analyzer - built an NLP-based resume matching tool.
    Chat App - real-time chat application using WebSockets.

    Certifications
    AWS Certified Cloud Practitioner
    """
    result = extract_sections(sample)
    for section, text in result.items():
        print(f"--- {section.upper()} ---")
        print(text or "(not detected)")
        print()
