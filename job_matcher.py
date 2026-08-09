"""
job_matcher.py
--------------
Module 4 & 5: Job Role Dataset + Matching and Recommendation.

Beginner approach (implemented here): convert the resume and each job
role's requirements into TF-IDF vectors, then rank roles by cosine
similarity. This is deliberately simple and explainable for a viva.

An "advanced" version could swap TfidfVectorizer for a Sentence
Transformer embedding model and still return the same shape of
results, since only `_vectorize` would change.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_JOB_ROLES_PATH = Path(__file__).parent / "data" / "job_roles.csv"


def load_job_roles(path: Path = DEFAULT_JOB_ROLES_PATH) -> pd.DataFrame:
    """Load job roles and their required skills (semicolon-separated)."""
    df = pd.read_csv(path)
    df["required_skills"] = df["required_skills"].str.lower()
    df["skills_list"] = df["required_skills"].apply(
        lambda s: [skill.strip() for skill in s.split(";") if skill.strip()]
    )
    return df


def match_resume_to_roles(
    cleaned_resume_text: str,
    job_roles_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Rank every job role by how well the resume matches it.

    Args:
        cleaned_resume_text: cleaned resume text (from text_cleaner).
        job_roles_df: optional pre-loaded job roles DataFrame.

    Returns:
        DataFrame with columns [job_role, match_score, required_skills]
        sorted from highest to lowest match_score (0-100 scale).
    """
    if job_roles_df is None:
        job_roles_df = load_job_roles()

    # Build one "document" per job role out of its required skills,
    # so TF-IDF can compare it against the resume text on the same footing.
    role_documents = job_roles_df["skills_list"].apply(lambda skills: " ".join(skills))

    corpus = [cleaned_resume_text] + role_documents.tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    resume_vector = tfidf_matrix[0:1]
    role_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(resume_vector, role_vectors)[0]

    results = job_roles_df.copy()
    results["match_score"] = (similarities * 100).round(1)
    results = results.sort_values("match_score", ascending=False).reset_index(drop=True)

    return results[["job_role", "match_score", "required_skills"]]


def top_n_roles(match_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return the top-N recommended roles."""
    return match_df.head(n)


if __name__ == "__main__":
    sample_cleaned = (
        "python pandas machine learning scikit-learn sql experience "
        "building models and analyzing data"
    )
    matches = match_resume_to_roles(sample_cleaned)
    print(matches)
    print("\nTop 3:")
    print(top_n_roles(matches, 3))
