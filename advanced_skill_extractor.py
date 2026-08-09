"""
advanced_skill_extractor.py
-----------------------------
Advanced Approach: "spaCy entity and phrase extraction" (project guide,
Section 9). This is a drop-in alternative to skill_extractor.py's plain
regex/keyword matching — it returns the exact same output shape
(list of {"skill": ..., "category": ...} dicts), so app.py can switch
between the two freely.

Requires the spaCy library AND a downloaded language model:

    pip install spacy
    python -m spacy download en_core_web_sm

If the model isn't installed, extract_skills_spacy raises
SpacyModelNotFound with instructions — the dashboard catches this and
falls back to the keyword-based extractor automatically.
"""

import pandas as pd

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_INSTALLED = True
except ImportError:
    SPACY_INSTALLED = False

_NLP_CACHE: dict = {}


class SpacyModelNotFound(Exception):
    """Raised when spaCy is installed but the requested language model isn't."""


def _load_nlp(model_name: str = "en_core_web_sm"):
    if model_name in _NLP_CACHE:
        return _NLP_CACHE[model_name]
    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise SpacyModelNotFound(
            f"spaCy model '{model_name}' is not installed. Run:\n"
            f"    python -m spacy download {model_name}"
        ) from e
    _NLP_CACHE[model_name] = nlp
    return nlp


def extract_skills_spacy(
    cleaned_text: str,
    skill_df: pd.DataFrame,
    model_name: str = "en_core_web_sm",
) -> list[dict]:
    """
    Extract skills using spaCy's PhraseMatcher instead of plain regex.

    This catches multi-word skill phrases more robustly than regex word
    boundaries (e.g. handles tokenization edge cases spaCy already
    solves), and is the natural place to extend to NER-based extraction
    of company names, degrees, etc. if desired later.

    Args:
        cleaned_text: output of text_cleaner.clean_resume_text().
        skill_df: skill dictionary DataFrame (skill, category columns).
        model_name: spaCy model to use.

    Returns:
        Same shape as skill_extractor.extract_skills():
        [{"skill": "python", "category": "programming"}, ...]

    Raises:
        RuntimeError: if spaCy itself isn't installed.
        SpacyModelNotFound: if spaCy is installed but the model isn't.
    """
    if not SPACY_INSTALLED:
        raise RuntimeError("spaCy is not installed. Run: pip install spacy")

    nlp = _load_nlp(model_name)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    categories = dict(zip(skill_df["skill"], skill_df["category"]))
    patterns = [nlp.make_doc(skill) for skill in skill_df["skill"]]
    matcher.add("SKILLS", patterns)

    doc = nlp(cleaned_text)
    matches = matcher(doc)

    found, seen = [], set()
    for _, start, end in matches:
        span_text = doc[start:end].text.lower().strip()
        if span_text and span_text not in seen:
            seen.add(span_text)
            found.append({"skill": span_text, "category": categories.get(span_text, "other")})

    return found


if __name__ == "__main__":
    from skill_extractor import load_skill_dictionary

    sample = "experienced python developer with sql, pandas and machine learning skills"
    try:
        result = extract_skills_spacy(sample, load_skill_dictionary())
        print(result)
    except (RuntimeError, SpacyModelNotFound) as e:
        print(f"Skipped: {e}")
