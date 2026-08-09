"""
job_matcher.py
--------------
Module 4 & 5: Job Role Dataset + Matching and Recommendation.

Beginner approach (implemented here): convert the resume and each job
role's requirements into TF-IDF vectors, then rank roles by cosine
similarity.

Both sides of the comparison are built from SKILL LISTS, not raw text —
the resume side uses the skills skill_extractor.py already found, and
the role side uses its required-skills list from job_roles.csv. This
keeps the two vectors "comparable" (same kind of short, skill-only
text) as Module 5 asks, so a resume that has every required skill
scores close to 100%, not diluted by unrelated words elsewhere in the
resume (job titles, university names, narrative sentences, etc.).

An "advanced" version could swap TfidfVectorizer for a Sentence
Transformer embedding model and still return the same shape of
results — see semantic_matcher.py, which follows the same skill-list
convention.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_JOB_ROLES_PATH = Path(__file__).parent / "data" / "job_roles.csv"


def load_job_roles(path: Path = DEFAULT_JOB_ROLES_PATH) -> pd.DataFrame:
    """Load job roles and their required skills (semicolon-separated)."""
    df = pd.read_csv(path)
    # Defensive: drop any blank/malformed rows (e.g. a stray trailing blank
    # line in the CSV) before they can turn into NaN and break downstream
    # string operations.
    df = df.dropna(subset=["job_role", "required_skills"]).reset_index(drop=True)
    df["required_skills"] = df["required_skills"].astype(str).str.lower()
    df["skills_list"] = df["required_skills"].apply(
        lambda s: [skill.strip() for skill in s.split(";") if skill.strip()]
    )
    return df


def match_resume_to_roles(
    resume_skills: list[str],
    job_roles_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Rank every job role by how well the resume's skills match it.

    Args:
        resume_skills: skill names extracted from the resume (from
            skill_extractor.extract_skills or the spaCy equivalent) —
            NOT raw resume text. Passing the full resume text here
            would dilute the score with unrelated words.
        job_roles_df: optional pre-loaded job roles DataFrame.

    Returns:
        DataFrame with columns [job_role, match_score, required_skills]
        sorted from highest to lowest match_score (0-100 scale).
    """
    if job_roles_df is None:
        job_roles_df = load_job_roles()

    # Build one "document" per job role out of its required skills, and
    # one for the resume out of its found skills — same representation
    # on both sides, so TF-IDF compares like with like. Every element is
    # explicitly cast to str: TfidfVectorizer requires plain strings, and
    # this guards against any stray non-string value (e.g. NaN) reaching
    # it and causing an opaque AttributeError deep inside scikit-learn.
    role_documents = job_roles_df["skills_list"].apply(
        lambda skills: " ".join(str(s) for s in skills)
    )
    resume_document = " ".join(str(s) for s in resume_skills)

    corpus = [str(resume_document)] + [str(d) for d in role_documents.tolist()]

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
    sample_skills = ["python", "pandas", "machine learning", "scikit-learn", "sql"]
    matches = match_resume_to_roles(sample_skills)
    print(matches)
    print("\nTop 3:")
    print(top_n_roles(matches, 3))
