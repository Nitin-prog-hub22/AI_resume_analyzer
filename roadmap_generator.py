"""
roadmap_generator.py
---------------------
Skill-Gap Analysis + rule-based learning roadmap.

Compares the skills found in a resume against a chosen target role's
required skills, reports what's present/missing, and produces a
simple week-by-week roadmap for the missing skills.
"""

# Very small, rule-based topic hints so the roadmap gives at least a
# one-line pointer for common missing skills instead of just a bare name.
LEARNING_HINTS = {
    "fastapi": "Learn FastAPI basics and build a small REST API.",
    "docker": "Learn Docker fundamentals and containerize a sample app.",
    "mlflow": "Learn MLflow for experiment tracking and model versioning.",
    "kubernetes": "Learn Kubernetes basics for container orchestration.",
    "aws": "Get familiar with core AWS services (S3, EC2, Lambda).",
    "azure": "Get familiar with core Azure services.",
    "gcp": "Get familiar with core GCP services.",
    "sql": "Practice SQL queries: joins, aggregations, and subqueries.",
    "power bi": "Learn Power BI for building interactive dashboards.",
    "tableau": "Learn Tableau for data visualization.",
    "deep learning": "Study neural network fundamentals (forward/backprop).",
    "transformers": "Study the Transformer architecture and attention.",
    "hugging face": "Explore the Hugging Face model hub and pipelines.",
    "opencv": "Practice image processing basics with OpenCV.",
    "cnn": "Study Convolutional Neural Networks for image tasks.",
    "yolo": "Learn the YOLO object-detection framework.",
    "llm": "Study how large language models are trained and used.",
    "rag": "Learn Retrieval-Augmented Generation (RAG) pipelines.",
}

DEFAULT_HINT = "Study the fundamentals and build one small hands-on project."


def analyze_skill_gap(found_skills: list[str], required_skills: list[str]) -> dict:
    """
    Compare found vs required skills for a target role.

    Args:
        found_skills: list of skill names extracted from the resume.
        required_skills: list of skill names required for the target role.

    Returns:
        dict with "matched" and "missing" skill lists.
    """
    found_set = {s.lower().strip() for s in found_skills}
    required_set = {s.lower().strip() for s in required_skills}

    matched = sorted(required_set & found_set)
    missing = sorted(required_set - found_set)

    return {"matched": matched, "missing": missing}


def generate_roadmap(missing_skills: list[str], weeks_per_skill: int = 1) -> list[dict]:
    """
    Build a simple week-by-week roadmap for the missing skills.

    Args:
        missing_skills: list of skill names not found in the resume.
        weeks_per_skill: how many weeks to allocate per missing skill.

    Returns:
        List of {"week": int, "skill": str, "focus": str} entries.
    """
    roadmap = []
    for i, skill in enumerate(missing_skills):
        week_start = i * weeks_per_skill + 1
        roadmap.append(
            {
                "week": week_start,
                "skill": skill,
                "focus": LEARNING_HINTS.get(skill, DEFAULT_HINT),
            }
        )
    return roadmap


if __name__ == "__main__":
    found = ["python", "pandas", "machine learning", "scikit-learn", "sql"]
    required = ["python", "machine learning", "scikit-learn", "fastapi", "docker", "mlflow"]

    gap = analyze_skill_gap(found, required)
    print("Matched:", gap["matched"])
    print("Missing:", gap["missing"])

    roadmap = generate_roadmap(gap["missing"])
    for step in roadmap:
        print(f"Week {step['week']}: {step['skill']} — {step['focus']}")
