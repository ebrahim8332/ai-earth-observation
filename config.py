"""
config.py — Environment variable loading and app-wide defaults.
All other modules import from here. Never read os.environ directly elsewhere.
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file if one exists in the working directory or project root.
# If no .env file is present, os.getenv() calls below simply return the defaults.
load_dotenv()

# AI provider keys — both are optional. The app degrades gracefully if absent.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Mapbox token — used if the app adds a Mapbox tile layer in a future day.
MAPBOX_TOKEN: str = os.getenv("MAPBOX_TOKEN", "")

# Default LLM provider setting from env. Not used for logic here;
# the ai_assistant module detects available keys and picks the chain automatically.
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "none")

# Groq model parameters — can be overridden via env without changing code.
GROQ_MAX_COMPLETION_TOKENS: int = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "2500"))
GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))


def has_groq() -> bool:
    """Return True if a Groq API key is available."""
    return bool(GROQ_API_KEY)


def has_gemini() -> bool:
    """Return True if a Gemini API key is available."""
    return bool(GEMINI_API_KEY)


def has_any_key() -> bool:
    """Return True if at least one AI provider key is configured."""
    return has_groq() or has_gemini()
