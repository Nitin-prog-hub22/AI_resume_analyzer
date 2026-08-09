# AI Resume Analyzer and Job Recommendation System

An NLP-based Streamlit application that analyzes a resume (PDF or DOCX),
extracts skills, compares it against a set of job roles, and produces:

- A resume-to-role **match score**
- The list of **skills found** and **missing skills** for a chosen target role
- A **ranked list of recommended roles**
- A simple **week-by-week learning roadmap** for closing skill gaps
- A **downloadable text report**

This project follows the "beginner approach" from the project spec:
keyword-based skill extraction + TF-IDF vectorization + cosine similarity.
It is structured so the matching engine can later be swapped for Sentence
Transformers or an LLM without touching the rest of the pipeline.

## Project Workflow

```
Upload PDF or DOCX resume
        |
Extract resume text            (resume_parser.py)
        |
Clean and normalize the text   (text_cleaner.py)
        |
Identify skills                (skill_extractor.py)
        |
Load job-role requirements     (data/job_roles.csv)
        |
Compare resume with each role  (job_matcher.py)
        |
Calculate match scores
        |
Recommend roles
        |
Show missing skills + roadmap  (roadmap_generator.py)
```

## Folder Structure

```
ai_resume_analyzer/
|-- app.py                  # Streamlit dashboard (entry point)
|-- resume_parser.py        # PDF/DOCX text extraction
|-- text_cleaner.py         # Text cleaning/normalization
|-- skill_extractor.py      # Keyword-based skill extraction
|-- advanced_skill_extractor.py  # spaCy-based skill extraction (Advanced)
|-- section_extractor.py    # Education / Experience / Projects detection
|-- job_matcher.py          # TF-IDF + cosine similarity matching
|-- semantic_matcher.py     # Sentence Transformers matching (Advanced)
|-- roadmap_generator.py    # Skill-gap analysis + learning roadmap
|-- pdf_report_generator.py # Optional: styled downloadable PDF report
|-- ai_feedback.py          # Optional: LLM feedback (Groq/Gemini/OpenAI)
|-- requirements.txt
|-- README.md
|-- .env                    # Optional API keys (not committed)
|-- .gitignore
|-- database.py             # SQLite analysis-history storage
|-- data/
|   |-- job_roles.csv          # Job roles and required skills
|   |-- skill_dictionary.csv   # Controlled skill list with categories
|   |-- history.db             # SQLite database (created on first save)
|
|-- sample_resumes/         # Sample DOCX resumes used for testing
|-- reports/                # Generated reports land here (gitignored)
|-- tests/
    |-- test_cases.csv       # Manual test log: expected vs actual role
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Enable AI-generated feedback: open `.env`, uncomment ONE of
   `GROQ_API_KEY`.

4. Run the app:

   ```bash
   streamlit run app.py
   ```

   On Windows, if `streamlit` isn't recognized as a command, use:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## How Matching Works (for your viva)

1. **Text extraction** — `pypdf` reads every PDF page; `python-docx` reads
   every paragraph and table cell in a DOCX.
2. **Cleaning** — text is lowercased and stripped of stray symbols, while
   protecting technical tokens like `C++`, `C#`, and `.NET` so they aren't
   mangled.
3. **Skill extraction** — a controlled dictionary (`data/skill_dictionary.csv`)
   of ~45 skills across 8 categories is searched for in the cleaned text
   using regex word-boundary matching.
4. **Section extraction** — separately, `section_extractor.py` scans the
   *raw* (pre-cleaning) text for heading lines like "Education" or "Work
   Experience" and captures the content under each one, since cleaning
   collapses line breaks that this step depends on.
5. **Matching** — the resume text and each job role's required-skills list
   are converted into **TF-IDF vectors**; **cosine similarity** between the
   resume vector and each role vector produces a 0-100 match score.
6. **Skill-gap analysis** — set difference between the target role's
   required skills and the resume's found skills gives matched/missing
   skills.
7. **Roadmap** — each missing skill gets a one-week study slot with a short
   rule-based learning hint.
8. **Reports** — the same matched/missing skills, ranked roles, and roadmap
   feed three downloadable formats: a styled PDF (`pdf_report_generator.py`,
   built with reportlab), a styled HTML file, and plain text.
9. **Optional AI feedback** — if an API key is configured, `ai_feedback.py`
   sends a short, fixed-structure prompt (role, match score, found/missing
   skills only — never the raw resume text) to Groq, Gemini, or OpenAI and
   shows the response on the dashboard and in the reports.

### spaCy skill extraction
```bash
pip install spacy
python -m spacy download en_core_web_sm
```
Then pick **"spaCy (Advanced)"** in the sidebar's "Skill extraction"
toggle. If the model isn't installed, the app shows a warning and
automatically falls back to keyword matching — it never crashes.

### Sentence Transformers matching
```bash
pip install sentence-transformers
```
Pick **"Sentence Transformers (Advanced)"** under "Matching method" in
the sidebar. The embedding model (`all-MiniLM-L6-v2`) downloads
automatically the first time it's used (needs internet once). Same
automatic fallback to TF-IDF if unavailable.

### Database (SQLite)
`database.py` stores an opt-in **history** of past analyses (score,
target role, matched/missing skills, timestamp) in
`data/history.db` — never the resume file itself, per the guide's
"avoid storing resumes permanently" rule. Use the "💾 Save this
analysis" button on the dashboard, or `save=true` on the `/analyze` API
call. No setup needed — SQLite ships with Python.

## Responsible AI Notes

- This tool is for **guidance only** — it does not make hiring or
  rejection decisions.
- It does **not** score gender, age, religion, nationality, photos, marital
  status, or disability — only job-related skills, education, projects,
  and experience.
- Match scores are estimates based on keyword/TF-IDF similarity, not a
  measure of a candidate's actual ability.
- Uploaded resumes are processed in-memory for the session only; nothing
  is written to disk unless the user explicitly downloads a report.

## Testing

See `tests/test_cases.csv` for the manual test log. Three sample resumes
(`sample_resumes/`) were run through the full pipeline and each correctly
ranked its intended role at or near the top — see that file for exact
scores.


