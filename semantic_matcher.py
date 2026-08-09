"""
semantic_matcher.py
---------------------
Advanced Approach: "Sentence Transformers for semantic matching". This is a drop-in alternative to
job_matcher.py's TF-IDF + cosine similarity — it returns the same
DataFrame shape (job_role, match_score, required_skills), so app.py
can switch between the two freely.

Why this is "advanced": TF-IDF only matches exact/overlapping words.
Sentence Transformers embed meaning, so a resume that says "built
predictive models" can match a role requiring "machine learning" even
without that literal phrase. The trade-off is a larger dependency and
a model download on first use.

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
    cleaned_resume_text: str,
    job_roles_df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Rank job roles by semantic similarity to the resume, using sentence
    embeddings instead of TF-IDF term overlap.

    Args:
        cleaned_resume_text: cleaned resume text (from text_cleaner).
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

    role_documents = job_roles_df["skills_list"].apply(lambda skills: ", ".join(skills)).tolist()

    resume_embedding = model.encode(cleaned_resume_text, convert_to_tensor=True)
    role_embeddings = model.encode(role_documents, convert_to_tensor=True)

    scores = util.cos_sim(resume_embedding, role_embeddings)[0].tolist()

    results = job_roles_df.copy()
    results["match_score"] = [round(max(0.0, s) * 100, 1) for s in scores]
    results = results.sort_values("match_score", ascending=False).reset_index(drop=True)

    return results[["job_role", "match_score", "required_skills"]]


if __name__ == "__main__":
    from job_matcher import load_job_roles

    sample = "built predictive models and analyzed data using python"
    try:
        result = match_resume_to_roles_semantic(sample, load_job_roles())
        print(result)
    except RuntimeError as e:
        print(f"Skipped: {e}")
