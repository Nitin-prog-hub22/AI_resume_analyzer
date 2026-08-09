"""
ai_feedback.py
---------------
Optional Advanced Feature: "LLM-generated resume feedback using a
controlled prompt" .

This module is entirely optional. The core analyzer (skill extraction,
TF-IDF matching, skill-gap roadmap) works fully without it. If an API
key is present in `.env`, the dashboard shows an extra button that
calls the configured provider for a short, structured piece of
feedback text.

Supported providers :
    GROQ_API_KEY    -> Groq (fast, generous free tier)
    

We call each provider's plain REST endpoint directly with `requests`
instead of pulling in every provider's SDK, to keep requirements.txt
light and to make the actual HTTP call visible for the viva
("How would you call an LLM API?").
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ, if present

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

REQUEST_TIMEOUT = 30


class AIFeedbackError(Exception):
    """Raised when no provider is configured or the API call fails."""


def get_configured_provider() -> str | None:
    """Return the first configured provider name, or None if none is set."""
    if GROQ_API_KEY:
        return "groq"
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return None


def _build_prompt(resume_skills: list[str], missing_skills: list[str], target_role: str, match_score: float) -> str:
    """
    Controlled prompt: fixed structure, resume content passed only as a
    skill list (never the raw resume text) to keep the request small,
    predictable, and free of personally identifiable information.
    """
    return (
        "You are a career coach giving short, constructive feedback on a resume-to-role match. "
        f"Target role: {target_role}. Match score: {match_score:.0f}%. "
        f"Skills already on the resume: {', '.join(resume_skills) or 'none detected'}. "
        f"Skills missing for this role: {', '.join(missing_skills) or 'none'}. "
        "Write 3 short paragraphs, no headings, no markdown formatting: "
        "(1) one encouraging sentence about their current strengths, "
        "(2) concrete, practical advice on how to close the missing-skill gap, "
        "(3) one suggestion for how to better showcase existing skills on the resume itself. "
        "Keep it under 130 words total. Do not invent specific employers, degrees, or personal details."
    )


def _call_groq(prompt: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 300,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_openai(prompt: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 300,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


_PROVIDER_FUNCS = {
    "groq": _call_groq,
    "gemini": _call_gemini,
    "openai": _call_openai,
}


def get_ai_feedback(
    resume_skills: list[str],
    missing_skills: list[str],
    target_role: str,
    match_score: float,
    provider: str | None = None,
) -> str:
    """
    Generate short AI feedback text for the resume-to-role match.

    Args:
        resume_skills: skills already found in the resume.
        missing_skills: skills missing for the target role.
        target_role: the role the user selected.
        match_score: the computed match score (0-100).
        provider: "groq" | "gemini" | "openai". If None, auto-detects
            from whichever API key is set in `.env`.

    Returns:
        Feedback text from the LLM.

    Raises:
        AIFeedbackError: if no provider is configured, or the API call fails.
    """
    provider = provider or get_configured_provider()
    if provider is None:
        raise AIFeedbackError(
            "No AI provider configured. Add GROQ_API_KEY, GEMINI_API_KEY, "
            "or OPENAI_API_KEY to your .env file to enable this optional feature."
        )
    if provider not in _PROVIDER_FUNCS:
        raise AIFeedbackError(f"Unknown provider '{provider}'. Choose groq, gemini, or openai.")

    prompt = _build_prompt(resume_skills, missing_skills, target_role, match_score)

    try:
        return _PROVIDER_FUNCS[provider](prompt)
    except requests.exceptions.RequestException as e:
        raise AIFeedbackError(f"AI feedback request failed: {e}") from e
    except (KeyError, IndexError) as e:
        raise AIFeedbackError(f"Unexpected response format from {provider}: {e}") from e


if __name__ == "__main__":
    # Manual test hook: set an API key in .env, then run `python ai_feedback.py`
    provider = get_configured_provider()
    if provider is None:
        print("No provider configured — add an API key to .env to test this.")
    else:
        print(f"Using provider: {provider}")
        feedback = get_ai_feedback(
            resume_skills=["python", "sql", "excel"],
            missing_skills=["pandas", "tableau"],
            target_role="Data Analyst",
            match_score=52.0,
        )
        print(feedback)
