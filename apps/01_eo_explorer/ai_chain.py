"""
ai_chain.py — Multi-model AI fallback chain for the EOIL portal.

Tries providers in order of quality. Locks to the first model that succeeds
and reuses it for the rest of the session. If the locked model fails
mid-session (rate limit or server error), the chain continues from that
point and re-locks to the next working model.

Chain order when both keys are present (best quality first):
  [0]  gemini-2.5-pro          — best reasoning, highest quality
  [1]  gemini-2.5-flash        — fast, high quality
  [2]  gemini-2.0-flash        — reliable, good quality
  [3]  gemini-2.0-flash-lite   — lighter, still strong
  [4]  gemini-2.5-flash-lite   — lightest Gemini option
  [5]  llama-3.3-70b-versatile — Groq Tier 1
  [6]  llama-4-scout-17b       — Groq Tier 2, high limits
  [7]  qwen3-32b               — Groq Tier 3
  [8]  llama-3.1-8b-instant    — Groq last resort, very high RPD

If only GROQ_API_KEY is set, Gemini tiers are skipped.
If neither key is set, complete() returns (None, None) and the caller
uses its own substantive fallback text.

Usage:
    from ai_chain import complete

    text, model = complete(prompt, groq_key=config.GROQ_API_KEY, gemini_key=config.GEMINI_API_KEY)
    if text:
        st.markdown(text)
        st.caption(f"AI response from {model}")
    else:
        st.markdown(fallback_text)
"""

import streamlit as st

# Session state key for the locked provider index.
# One lock is shared across all modules in a session — once a model works
# it is reused everywhere.
_LOCK_KEY = "ai_chain_locked_index"

# ---------------------------------------------------------------------------
# Chain definition — ordered list of (provider, model_name) tuples
# Built at call time from whichever keys are available.
# ---------------------------------------------------------------------------

_GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
]

_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]


def _build_chain(groq_key, gemini_key):
    """Return the ordered list of (provider_fn, model_name) tuples.

    provider_fn(prompt, model_name, key) -> str
    Raises an exception on any failure so the chain can continue.
    """
    chain = []

    if gemini_key:
        for model in _GEMINI_MODELS:
            chain.append(("gemini", model, gemini_key))

    if groq_key:
        for model in _GROQ_MODELS:
            chain.append(("groq", model, groq_key))

    return chain


# ---------------------------------------------------------------------------
# Provider call functions — one per API
# Each raises an exception on failure; the chain catches it and moves on.
# ---------------------------------------------------------------------------

def _call_gemini(prompt, model_name, api_key):
    """Call a Gemini text model and return the response string.

    Uses google-genai (the new SDK, not google-generativeai).
    Plain text response — no JSON mode needed for EOIL interpretations.
    """
    from google import genai
    from google.genai import types

    client   = genai.Client(api_key=api_key)
    config   = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=1024,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=config,
    )
    text = response.text
    if not text or not text.strip():
        raise ValueError(f"Empty response from {model_name}")
    return text.strip()


def _call_groq(prompt, model_name, api_key):
    """Call a Groq model and return the response string.

    Plain text — no response_format JSON mode needed here.
    """
    from groq import Groq

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise ValueError(f"Empty response from {model_name}")
    return text.strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def complete(prompt, groq_key="", gemini_key=""):
    """Try each provider in order and return (response_text, model_name).

    Locks to the first model that succeeds. Re-locks if the locked model
    fails and a later one succeeds.

    Returns (None, None) if no keys are provided or all models fail.
    The caller is responsible for showing fallback text in that case.
    """
    chain = _build_chain(groq_key or "", gemini_key or "")

    if not chain:
        return None, None

    start = st.session_state.get(_LOCK_KEY, 0)
    # Guard against a stale lock pointing past the end of this chain
    start = min(start, len(chain) - 1)

    errors = []
    for i in range(start, len(chain)):
        provider, model_name, key = chain[i]

        try:
            if provider == "gemini":
                text = _call_gemini(prompt, model_name, key)
            else:
                text = _call_groq(prompt, model_name, key)

            # Lock (or re-lock) to this working model for the rest of the session
            st.session_state[_LOCK_KEY] = i
            return text, model_name

        except Exception as e:
            errors.append(f"{model_name}: {type(e).__name__}")
            continue

    # All models exhausted
    return None, None


def active_model():
    """Return the name of the currently locked model, or None if not yet locked."""
    locked = st.session_state.get(_LOCK_KEY)
    return None if locked is None else f"chain position {locked}"
