"""
imagery_interpreter.py — AI Imagery Interpreter portal module (Day 10)

Fetches a Sentinel-2 true-color chip from Planetary Computer for a user-specified
location and date, then passes the image bytes to a vision AI model for plain-language
interpretation.

Vision chain (8 models, multimodal only):
  [0] gemini-2.5-pro          Google — best reasoning
  [1] gemini-2.5-flash        Google — fast, confirmed working
  [2] gemini-2.0-flash        Google
  [3] gemini-2.0-flash-lite   Google
  [4] gemini-2.5-flash-lite   Google
  [5] gemini-flash-latest     Google — alias
  [6] llama-4-scout-17b       Groq — confirmed working
  [7] llama-4-maverick        Groq — Llama 4 multimodal

Text-only models (llama-3.3-70b, qwen3-32b, gpt-oss-120b, llama-3.1-8b) are not
included — they cannot receive images.

Reuses spectral_explorer.py for all Planetary Computer fetch/render logic.
No duplication of STAC search or COG rendering code.
"""

import io
import base64
from datetime import datetime, timedelta

import numpy as np
import streamlit as st
from PIL import Image as PILImage

import spectral_explorer

# ---------------------------------------------------------------------------
# Vision chain definition
# ---------------------------------------------------------------------------

# Only multimodal-capable models. Ordered best-to-fastest within each provider.
_VISION_GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
]

_VISION_GROQ_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

# Session state key for the vision chain lock.
# Separate from the text chain lock so the two do not interfere.
_VISION_LOCK_KEY = "vision_chain_locked_index"

# ---------------------------------------------------------------------------
# Structured prompt
# ---------------------------------------------------------------------------

_VISION_PROMPT = """\
You are an earth observation analyst reviewing a Sentinel-2 true-color satellite image.

Analyze what you see and write 3-4 paragraphs covering:

1. Land cover types visible — vegetation, water, urban areas, bare soil, \
agriculture, forest, wetlands, snow or ice
2. Notable features or patterns — rivers, coastlines, roads, field boundaries, \
deforestation edges, urban sprawl, industrial sites
3. Seasonal or ecological state — green vs dry season, vegetation density, \
water levels, cloud or snow cover
4. What this image could inform — land use monitoring, environmental assessment, \
agricultural planning, infrastructure review, or disaster response

Be specific about what is actually visible in the image. Do not guess or invent \
features that are not clearly present.

Region: {location}
Image date: {date}
"""

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_chip(bbox: list, date_str: str) -> tuple:
    """
    Fetch the best available Sentinel-2 true-color chip from Planetary Computer.

    Searches a 60-day window centred on date_str (±30 days). Picks the scene
    with the best spatial coverage and cloud cover below 20%.
    Falls back to 50% cloud tolerance if no clean scene is found.

    Returns:
        (image_array, metadata) where image_array is a uint8 numpy H×W×3 array
        and metadata is a dict with keys: date, cloud_cover, scene_id.
        Returns (None, None) if no usable scene is found.
    """
    target = datetime.strptime(date_str, "%Y-%m-%d")
    start  = (target - timedelta(days=30)).strftime("%Y-%m-%d")
    end    = (target + timedelta(days=30)).strftime("%Y-%m-%d")
    date_range = f"{start}/{end}"

    try:
        catalog = spectral_explorer.get_catalog()

        # First attempt: clean scenes only
        items = spectral_explorer.search_scenes(
            catalog,
            collection  = "sentinel-2-l2a",
            bbox        = bbox,
            date_range  = date_range,
            max_cloud   = 20,
            cloud_field = "eo:cloud_cover",
        )

        # Second attempt: relax cloud tolerance
        if not items:
            items = spectral_explorer.search_scenes(
                catalog,
                collection  = "sentinel-2-l2a",
                bbox        = bbox,
                date_range  = date_range,
                max_cloud   = 50,
                cloud_field = "eo:cloud_cover",
            )

        if not items:
            return None, None

        # Pick best scene by spatial coverage
        best_item, _, _ = spectral_explorer.find_best_scene(
            items, "B04", "B03", "B02", "Sentinel-2 L2A", max_to_check=5
        )
        if best_item is None:
            best_item = items[0]

        # Render true-color: B4 = Red, B3 = Green, B2 = Blue
        arr = spectral_explorer.render_combination(
            best_item, "B04", "B03", "B02", "Sentinel-2 L2A", width=600
        )

        if arr is None or arr.max() == 0:
            return None, None

        metadata = {
            "date":        best_item.datetime.strftime("%Y-%m-%d"),
            "cloud_cover": best_item.properties.get("eo:cloud_cover", 0),
            "scene_id":    best_item.id,
        }
        return arr, metadata

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def array_to_jpeg_bytes(arr: np.ndarray) -> bytes:
    """Convert a uint8 H×W×3 numpy array to compressed JPEG bytes."""
    img = PILImage.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Vision AI chain
# ---------------------------------------------------------------------------

def interpret_image(
    image_bytes: bytes,
    location:    str,
    date_str:    str,
    gemini_key:  str = "",
    groq_key:    str = "",
) -> tuple:
    """
    Send image bytes to the vision chain and return (interpretation_text, model_name).

    Tries models in order. Locks to first success for the rest of the session.
    Returns (None, None) if all models fail — caller shows fallback text.
    """
    prompt = _VISION_PROMPT.format(location=location, date=date_str)
    return _complete_vision(prompt, image_bytes, gemini_key=gemini_key, groq_key=groq_key)


def _complete_vision(
    prompt:      str,
    image_bytes: bytes,
    gemini_key:  str = "",
    groq_key:    str = "",
) -> tuple:
    """
    Internal vision fallback chain.
    Mirrors the text chain in ai_chain.py but sends image + prompt together.
    Uses a separate session lock key so vision and text locks are independent.
    """
    chain = []
    if gemini_key:
        for model in _VISION_GEMINI_MODELS:
            chain.append(("gemini", model, gemini_key))
    if groq_key:
        for model in _VISION_GROQ_MODELS:
            chain.append(("groq", model, groq_key))

    if not chain:
        return None, None

    start = st.session_state.get(_VISION_LOCK_KEY, 0)
    start = min(start, len(chain) - 1)

    for i in range(start, len(chain)):
        provider, model_name, key = chain[i]
        try:
            if provider == "gemini":
                text = _call_gemini_vision(prompt, image_bytes, model_name, key)
            else:
                text = _call_groq_vision(prompt, image_bytes, model_name, key)

            # Lock to this model for the rest of the session
            st.session_state[_VISION_LOCK_KEY] = i
            return text, model_name

        except Exception:
            continue

    return None, None


def _call_gemini_vision(
    prompt:      str,
    image_bytes: bytes,
    model_name:  str,
    api_key:     str,
) -> str:
    """
    Call a Gemini multimodal model with image bytes and a text prompt.
    Uses the google-genai SDK (new SDK, not google-generativeai).
    Image is sent as base64-encoded inline data.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    b64    = base64.b64encode(image_bytes).decode()

    response = client.models.generate_content(
        model    = model_name,
        contents = [types.Content(role="user", parts=[
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=b64)),
            types.Part(text=prompt),
        ])],
        config = types.GenerateContentConfig(
            temperature       = 0.3,
            max_output_tokens = 1024,
        ),
    )

    text = response.text
    if not text or not text.strip():
        raise ValueError(f"Empty response from {model_name}")
    return text.strip()


def _call_groq_vision(
    prompt:      str,
    image_bytes: bytes,
    model_name:  str,
    api_key:     str,
) -> str:
    """
    Call a Groq vision model with image bytes and a text prompt.
    Groq vision requires image as a base64 data URL in the message content.
    """
    from groq import Groq

    client   = Groq(api_key=api_key)
    b64      = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    response = client.chat.completions.create(
        model    = model_name,
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text",      "text": prompt},
        ]}],
        max_tokens  = 1024,
        temperature = 0.3,
    )

    text = response.choices[0].message.content
    if not text or not text.strip():
        raise ValueError(f"Empty response from {model_name}")
    return text.strip()


# ---------------------------------------------------------------------------
# Fallback interpretation
# ---------------------------------------------------------------------------

def get_fallback_interpretation(location: str, date_str: str) -> str:
    """
    Substantive fallback shown when no vision model is available.
    Gives the user a reading guide for the true-color image rather than
    an error message or placeholder text.
    """
    month = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %Y")

    return (
        f"No vision AI model is available to interpret this image. "
        f"The image shows a Sentinel-2 true-color composite of **{location}** "
        f"from **{month}**.\n\n"
        "**How to read a true-color satellite image:**\n\n"
        "- **Dark green to bright green** areas are healthy vegetation. "
        "Dense forest appears darker. Crops and grassland appear lighter.\n"
        "- **Dark blue to black** areas are water. Deep water absorbs almost all light "
        "and appears very dark. Shallow water may appear lighter or blue-green.\n"
        "- **Grey to white** areas are urban surfaces or bare concrete. "
        "Road networks appear as thin grey lines.\n"
        "- **Brown or tan** areas are bare soil, dry grassland, or recently harvested fields.\n"
        "- **Bright white patches** are either cloud cover or snow and ice.\n\n"
        "To enable AI vision interpretation, add a `GEMINI_API_KEY` or `GROQ_API_KEY` "
        "to the app's environment variables."
    )
