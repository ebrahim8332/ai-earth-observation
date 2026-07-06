"""
ai_chain.py — Multi-model AI fallback chain for the EOIL portal.

Tries providers in order of quality. Locks to the first model that succeeds
and reuses it for the rest of the session. If the locked model fails
mid-session (rate limit or server error), the chain continues from that
point and re-locks to the next working model.

Chain order when both keys are present (best quality first):
  [0]  gemini-2.5-pro          — highest quality Gemini model
  [1]  gemini-3-flash-preview  — matches 2.5 Pro quality, 3x faster
  [2]  gemini-3.1-flash-lite   — best speed-to-quality in Gemini tier
  [3]  gemini-2.5-flash        — reliable all-rounder
  [4]  gemini-2.5-flash-lite   — lighter variant of 2.5 Flash
  [5]  gemini-2.0-flash        — deprecated June 2026, 8K output cap
  [6]  gemini-2.0-flash-lite   — deprecated June 2026, 8K output cap
  [7]  qwen/qwen3.6-27b         — Groq Tier 1 (replaces llama-3.3-70b, deprecated Jul 2 2026)
  [8]  qwen/qwen3-32b           — Groq Tier 2
  [10] gpt-oss-120b            — Groq Tier 4, 200K TPD
  [11] llama-3.1-8b-instant    — Groq Tier 5, very high RPD
  [12] gpt-oss-20b             — Groq Tier 6, fast, lower nuance
  [13] gemini-flash-latest     — unstable alias, absolute last resort

If only GROQ_API_KEY is set, Gemini tiers are skipped.
If neither key is set, complete() returns (None, None) and the caller
uses its own substantive fallback text.

Output token ceilings come from config.py (GEMINI_MAX_OUTPUT_TOKENS,
GROQ_MAX_COMPLETION_TOKENS) — the same values ai_assistant.py's chain uses —
rather than a hardcoded number here, so there is one place to raise or lower
the ceiling for the whole portal instead of two independently-drifting ones.

Usage:
    from ai_chain import complete

    text, model = complete(prompt, groq_key=config.GROQ_API_KEY, gemini_key=config.GEMINI_API_KEY)
    if text:
        st.markdown(text)
        st.caption(f"AI response from {model}")
    else:
        st.markdown(fallback_text)
"""

import config

# ---------------------------------------------------------------------------
# Chain definition — ordered list of (provider, model_name) tuples
# Built at call time from whichever keys are available.
# ---------------------------------------------------------------------------

_GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    # gemini-flash-latest is an unstable alias — appended after all Groq models in _build_chain()
]

_GROQ_MODELS = [
    # llama-3.3-70b-versatile deprecated July 2, 2026; decommission Aug 16, 2026.
    # qwen3.6-27b promoted to first Groq slot as Groq-recommended replacement.
    "qwen/qwen3.6-27b",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",           # smaller/faster sibling to 120B, last Groq resort
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

    # gemini-flash-latest is an unstable alias — it goes last, after all Groq models
    if gemini_key:
        chain.append(("gemini", "gemini-flash-latest", gemini_key))

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

    client     = genai.Client(api_key=api_key)
    gen_config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=gen_config,
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
        max_tokens=config.GROQ_MAX_COMPLETION_TOKENS,
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

    Each call starts from position 0 — this app makes one AI call per user
    interaction, so there is no operation-level lock to maintain.

    Returns (None, None) if no keys are provided or all models fail.
    The caller is responsible for showing fallback text in that case.
    """
    chain = _build_chain(groq_key or "", gemini_key or "")

    if not chain:
        return None, None

    errors = []
    for provider, model_name, key in chain:
        try:
            if provider == "gemini":
                text = _call_gemini(prompt, model_name, key)
            else:
                text = _call_groq(prompt, model_name, key)
            return text, model_name

        except Exception as e:
            errors.append(f"{model_name}: {type(e).__name__}")
            continue

    # All models exhausted
    return None, None
