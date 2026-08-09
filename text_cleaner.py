"""
text_cleaner.py
----------------
Module 2: Text Extraction and Cleaning (cleaning half).

Normalizes raw resume text so it can be reliably compared against
skill keywords and job descriptions, while preserving technical
tokens like C++, C#, and .NET that a naive cleaner would destroy.
"""

import re

# Tokens that must survive cleaning untouched (checked case-insensitively).
# We protect them with placeholders before stripping symbols, then restore them.
PROTECTED_TOKENS = ["c++", "c#", ".net", "node.js", "asp.net", "vue.js"]


def _protect_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Temporarily replace protected tokens with safe placeholders."""
    mapping = {}
    for i, token in enumerate(PROTECTED_TOKENS):
        # Alphanumeric-only placeholder so it survives symbol stripping untouched.
        placeholder = f"zzprotectedtoken{i}zz"
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(placeholder, text)
            mapping[placeholder] = token
    return text, mapping


def _restore_tokens(text: str, mapping: dict[str, str]) -> str:
    for placeholder, token in mapping.items():
        text = text.replace(placeholder, token)
    return text


def clean_resume_text(raw_text: str) -> str:
    """
    Clean and normalize resume text.

    Steps:
      1. Protect special technical tokens (C++, C#, .NET, etc.).
      2. Lowercase everything for case-insensitive comparison.
      3. Remove bullet characters, extra punctuation, and non-printable noise.
      4. Collapse repeated whitespace/newlines into single spaces.
      5. Restore the protected technical tokens.

    Args:
        raw_text: unprocessed text extracted from a resume file.

    Returns:
        Cleaned, lowercase, single-spaced text.
    """
    text = raw_text

    # Step 1: protect tokens that contain symbols we're about to strip
    text, mapping = _protect_tokens(text)

    # Step 2: lowercase
    text = text.lower()

    # Step 3: remove bullets / odd unicode punctuation, keep letters,
    # digits, spaces, and a small set of useful punctuation (+ # . / -)
    text = re.sub(r"[•●▪◦‣·]", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\#\.\/\-]", " ", text)

    # Step 4: collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 5: restore protected tokens
    text = _restore_tokens(text, mapping)

    return text


if __name__ == "__main__":
    sample = """
    • Proficient in Python, C++, and C#.
    - Built REST APIs using .NET and Node.js!!
    Email: someone@example.com    Phone: +91-9999999999
    """
    print(clean_resume_text(sample))
