"""
semantic_matcher.py
---------------------
Advanced Approach: "Sentence Transformers for semantic matching"
(project guide, Section 9). This is a drop-in alternative to
job_matcher.py's TF-IDF + cosine similarity — it returns the same
DataFrame shape (job_role, match_score, required_skills), so app.py
can switch between the two freely.

Like job_matcher.py, this compares the resume's EXTRACTED SKILLS
against each role's required skills — not raw resume text — so
switching between "TF-IDF (Beginner)" and "Sentence Transformers
(Advanced)" in the dashboard gives scores on the same intuitive scale
(full skill coverage -> a score close to 100%), rather than one mode
being diluted by unrelated resume text and the other not.

Why this is still "advanced" despite comparing skill lists rather than
free text: embeddings capture *meaning*, so a resume skill written as
"ML" or "predictive modelling" can still match a role requiring
"machine learning" even without an exact string match — something
TF-IDF's literal term overlap cannot do.

Requires:
    pip install sentence-transformers
(the model itself, e.g. all-MiniLM-L6-v2, downloads automatically on
first use — this needs an internet connection the first time only).
"""

import pandas as pd

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_INSTALLED = True
except ImportError:
    SENTENCE_TRANSFORMERS_INSTALLED = False

_MODEL_CACHE: dict = {}


def _load_model(model_name: str = "all-MiniLM-L6-v2"):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def match_resume_to_roles_semantic(
    resume_skills: list[str],
    job_roles_df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Rank job roles by semantic similarity between the resume's skills
    and each role's required skills, using sentence embeddings instead
    of TF-IDF term overlap.

    Args:
        resume_skills: skill names extracted from the resume (same
            input job_matcher.match_resume_to_roles takes) — NOT raw
            resume text.
        job_roles_df: job roles DataFrame from job_matcher.load_job_roles()
            (must already have the "skills_list" column).
        model_name: any sentence-transformers model name.

    Returns:
        DataFrame with columns [job_role, match_score, required_skills],
        sorted highest to lowest, same shape as job_matcher's output.

    Raises:
        RuntimeError: if sentence-transformers isn't installed.
    """
    if not SENTENCE_TRANSFORMERS_INSTALLED:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )

    model = _load_model(model_name)

    role_documents = job_roles_df["skills_list"].apply(
        lambda skills: ", ".join(str(s) for s in skills)
    ).tolist()
    resume_document = ", ".join(str(s) for s in resume_skills)

    resume_embedding = model.encode(resume_document, convert_to_tensor=True)
    role_embeddings = model.encode(role_documents, convert_to_tensor=True)

    scores = util.cos_sim(resume_embedding, role_embeddings)[0].tolist()

    results = job_roles_df.copy()
    results["match_score"] = [round(max(0.0, s) * 100, 1) for s in scores]
    results = results.sort_values("match_score", ascending=False).reset_index(drop=True)

    return results[["job_role", "match_score", "required_skills"]]


if __name__ == "__main__":
    from job_matcher import load_job_roles

    sample_skills = ["python", "machine learning", "sql"]
    try:
        result = match_resume_to_roles_semantic(sample_skills, load_job_roles())
        print(result)
    except RuntimeError as e:
        print(f"Skipped: {e}")
