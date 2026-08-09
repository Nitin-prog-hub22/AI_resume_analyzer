"""
skill_extractor.py
-------------------
Module 3: Skill Extraction.

Loads a controlled skill dictionary (data/skill_dictionary.csv) and
searches cleaned resume text for those skills using keyword/phrase
matching. Skills are grouped into categories (programming, databases,
ml, cloud_devops, tools, etc.).

An LLM or spaCy-based extractor could be swapped in later as an
"advanced improvement" without changing the rest of the pipeline,
since both would just need to return the same
List[dict] structure from `extract_skills`.
"""

import re
from pathlib import Path

import pandas as pd

DEFAULT_SKILL_DICT_PATH = Path(__file__).parent / "data" / "skill_dictionary.csv"


def load_skill_dictionary(path: Path = DEFAULT_SKILL_DICT_PATH) -> pd.DataFrame:
    """Load the controlled skill list with its category labels."""
    df = pd.read_csv(path)
    # Defensive: drop any blank/malformed rows before they can turn into
    # NaN and break string operations downstream.
    df = df.dropna(subset=["skill", "category"]).reset_index(drop=True)
    df["skill"] = df["skill"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    return df


def _build_pattern(skill: str) -> re.Pattern:
    """
    Build a regex pattern for a skill that works for both plain words
    (with word boundaries) and symbol-heavy tokens like c++, c#, .net
    (where \\b word boundaries don't apply cleanly).
    """
    escaped = re.escape(skill)
    if re.match(r"^[a-z0-9\s]+$", skill):
        # Plain alphanumeric skill -> use strict word boundaries
        pattern = rf"\b{escaped}\b"
    else:
        # Symbol-containing skill (c++, c#, .net) -> boundary on left only,
        # loose on right since \b doesn't work next to punctuation.
        pattern = rf"(?<![a-z0-9]){escaped}"
    return re.compile(pattern, re.IGNORECASE)


def extract_skills(cleaned_text: str, skill_df: pd.DataFrame | None = None) -> list[dict]:
    """
    Search cleaned resume text for every skill in the dictionary.

    Args:
        cleaned_text: output of text_cleaner.clean_resume_text().
        skill_df: optional pre-loaded skill dictionary DataFrame.

    Returns:
        List of dicts: [{"skill": "python", "category": "programming"}, ...]
        for every skill found in the text, in dictionary order.
    """
    if skill_df is None:
        skill_df = load_skill_dictionary()

    found = []
    for _, row in skill_df.iterrows():
        skill, category = row["skill"], row["category"]
        pattern = _build_pattern(skill)
        if pattern.search(cleaned_text):
            found.append({"skill": skill, "category": category})

    return found


def group_by_category(found_skills: list[dict]) -> dict[str, list[str]]:
    """Group a flat list of found skills into {category: [skills]}."""
    grouped: dict[str, list[str]] = {}
    for item in found_skills:
        grouped.setdefault(item["category"], []).append(item["skill"])
    return grouped


if __name__ == "__main__":
    sample_cleaned = (
        "experienced python developer with sql pandas numpy and "
        "machine learning scikit-learn skills. familiar with c++ and .net"
    )
    skills = extract_skills(sample_cleaned)
    print("Found skills:", skills)
    print("Grouped:", group_by_category(skills))
