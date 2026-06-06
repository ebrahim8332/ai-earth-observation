"""
ai_assistant.py — AI provider chain and fallback logic.

Primary path: tries Gemini models first (if GEMINI_API_KEY set),
then Groq models (if GROQ_API_KEY set), in priority order.
Falls back to local predefined functions if all API calls fail or no keys are set.

The app never crashes due to a missing or failed API key.
"""

import json
import time
import config

# --- Provider chain definition ---
# Each entry: (provider, model_id, display_label)
# Tried in order. Locks to first model that succeeds.
# If a locked model fails mid-session, continues from that point.
PROVIDER_CHAIN = []

if config.has_gemini():
    PROVIDER_CHAIN += [
        ("gemini", "gemini-2.5-pro",        "Gemini 2.5 Pro"),
        ("gemini", "gemini-2.5-flash",       "Gemini 2.5 Flash"),
        ("gemini", "gemini-2.0-flash",       "Gemini 2.0 Flash"),
        ("gemini", "gemini-2.0-flash-lite",  "Gemini 2.0 Flash Lite"),
        ("gemini", "gemini-2.5-flash-lite",  "Gemini 2.5 Flash Lite"),
        ("gemini", "gemini-flash-latest",    "Gemini Flash Latest"),
    ]

if config.has_groq():
    PROVIDER_CHAIN += [
        ("groq", "llama-3.3-70b-versatile",                        "Groq Llama 3.3 70B"),
        ("groq", "meta-llama/llama-4-scout-17b-16e-instruct",      "Groq Llama 4 Scout"),
        ("groq", "qwen/qwen3-32b",                                  "Groq Qwen3 32B"),
        ("groq", "openai/gpt-oss-120b",                             "Groq GPT-OSS 120B"),
        ("groq", "llama-3.1-8b-instant",                            "Groq Llama 3.1 8B"),
    ]

# Session state for chain locking: tracks which index we last succeeded at.
# Using a mutable dict so the lock persists across function calls in one session.
_chain_state = {"locked_index": 0}


def get_provider_status() -> str:
    """Return a short string describing which AI mode is active, for display in the sidebar."""
    if not PROVIDER_CHAIN:
        return "Fallback mode (no API keys)"
    label = PROVIDER_CHAIN[_chain_state["locked_index"]][2]
    return f"API ({label})"


def ask(question: str, theme: str, dataset: str, location: str, mode: str) -> str:
    """
    Handle a user-submitted question.
    Tries the API chain first. Falls back to local functions if the chain is empty or fails.
    """
    if not PROVIDER_CHAIN:
        return _fallback_for_mode(mode, theme, dataset)

    system_prompt = (
        "You are an Earth Observation tutor, AI strategist, and geospatial advisor. "
        "The user is a technology and AI advisor rebuilding their EO knowledge. "
        "Respond clearly and concisely. Be technically accurate but accessible. "
        "Avoid academic language. Use plain, direct sentences. "
        f"Current context: theme={theme}, dataset={dataset}, location={location}."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    response = _call_chain(messages)
    return response if response else _fallback_for_mode(mode, theme, dataset)


def auto_explain(theme: str, dataset: str, location: str, mode: str) -> str:
    """
    Generate an automatic explanation based on the active AI mode, without a user question.
    Used to populate the AI panel when the user has not submitted a question yet.
    """
    prompts = {
        "Explain selected theme":   f"Give a clear, practical explanation of the '{theme}' theme in Earth Observation. Cover what it is, what questions it answers, and one concrete business application. Plain language only.",
        "Explain selected dataset": f"Explain the '{dataset}' dataset. Cover what it measures, the sensor that produces it, its resolution and revisit frequency, and its main limitation. Plain language only.",
        "Explain business use case":f"Give one concrete business use case for '{dataset}' in the context of '{theme}'. State the decision it supports, the data it requires, and the business value. Plain language only.",
        "Explain limitations":      f"What are the key limitations of '{dataset}' for the '{theme}' use case? Be specific about what the data cannot do and why. Plain language only.",
        "Suggest next analysis":    f"Given the '{theme}' theme and '{dataset}' dataset, what should the analyst do next? Suggest one specific follow-on analysis, the data it needs, and why it adds value. Plain language only.",
    }

    user_msg = prompts.get(mode, f"Explain {theme} and {dataset} briefly.")

    if not PROVIDER_CHAIN:
        return _fallback_for_mode(mode, theme, dataset)

    system_prompt = (
        "You are an Earth Observation tutor and geospatial advisor. "
        "Respond in plain, direct language. No academic tone. No filler. "
        "Answers should be 100-200 words maximum."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]

    response = _call_chain(messages)
    return response if response else _fallback_for_mode(mode, theme, dataset)


# --- Internal chain execution ---

def _call_chain(messages: list) -> str:
    """
    Walk the provider chain starting from the locked index.
    On success, lock to that index and return the response text.
    On retriable error (429, 503, 413, timeout), advance to the next provider.
    On auth error (401, 403), raise immediately — no fallback.
    Returns empty string if all providers fail.
    """
    start = _chain_state["locked_index"]
    for i in range(start, len(PROVIDER_CHAIN)):
        provider, model_id, label = PROVIDER_CHAIN[i]
        try:
            if provider == "groq":
                result = _call_groq(model_id, messages)
            else:
                result = _call_gemini(model_id, messages)
            # Success — lock to this index for the remainder of the session
            _chain_state["locked_index"] = i
            return result
        except _AuthError:
            # Auth errors are not retried — surface them immediately
            raise
        except Exception:
            # Any other error (rate limit, timeout, service error) — try next
            continue
    return ""


def _call_groq(model_id: str, messages: list) -> str:
    """
    Call the Groq API with the given model and messages.
    Returns the response content as a plain string.
    Raises _AuthError on 401/403. Raises generic Exception on all other failures.
    """
    from groq import Groq, AuthenticationError, RateLimitError

    client = Groq(api_key=config.GROQ_API_KEY)
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.3,
            max_tokens=config.GROQ_MAX_COMPLETION_TOKENS,
            timeout=60,
        )
        return completion.choices[0].message.content or ""
    except AuthenticationError as e:
        raise _AuthError(str(e))
    except Exception as e:
        raise e


def _call_gemini(model_id: str, messages: list) -> str:
    """
    Call the Gemini API using the new google.genai SDK.
    Matches the pattern used in deck-studio-coach/src/providers/gemini_provider.py.
    Converts OpenAI-style messages to Gemini format internally.
    Raises _AuthError on 401/403. Raises generic Exception on all other failures.
    """
    from google import genai
    from google.genai import types, errors as gemini_errors

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # Extract system message — Gemini takes it as a separate config field
    system_text = ""
    gemini_contents = []
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_text = content
        else:
            gemini_role = "model" if role == "assistant" else "user"
            gemini_contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part(text=content)],
                )
            )

    gen_config = types.GenerateContentConfig(
        system_instruction=system_text if system_text else None,
        temperature=0.3,
        max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
    )

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=gemini_contents,
            config=gen_config,
        )
        return response.text or ""
    except gemini_errors.ClientError as e:
        if e.code in (401, 403):
            raise _AuthError(str(e))
        raise e
    except Exception as e:
        raise e


# --- Fallback functions ---
# These run when no API key is configured or all API calls fail.
# Responses are substantive — not placeholder text.

def _fallback_for_mode(mode: str, theme: str, dataset: str) -> str:
    """Route to the correct fallback function based on active AI mode."""
    if mode == "Explain selected theme":
        return explain_theme(theme)
    elif mode == "Explain selected dataset":
        return explain_dataset(dataset)
    elif mode == "Explain business use case":
        return explain_use_case(theme, dataset)
    elif mode == "Explain limitations":
        return explain_limitations(dataset)
    else:
        return suggest_next_analysis(theme)


def explain_theme(theme: str) -> str:
    """Return a detailed plain-English explanation of the given EO theme."""
    explanations = {
        "EO Basics": (
            "**Earth Observation (EO) — What it is**\n\n"
            "EO is the use of sensors on satellites, aircraft, and drones to collect "
            "information about the Earth's surface and atmosphere. The data is used to "
            "detect change, monitor conditions, and answer questions that ground surveys "
            "cannot answer at scale or speed.\n\n"
            "The core sensor types are:\n"
            "- **Optical/multispectral** — measure reflected sunlight across multiple wavelength bands. "
            "Sentinel-2 and Landsat are the primary free sources.\n"
            "- **Radar (SAR)** — active sensor, works at night and through cloud cover. Sentinel-1 is free.\n"
            "- **Thermal** — measures heat emission. Landsat TIRS provides free thermal data at 100m resolution.\n"
            "- **Atmospheric spectrometers** — measure gas concentrations. Sentinel-5P TROPOMI is the free source.\n"
            "- **LiDAR** — measures vertical structure with laser pulses. GEDI on the ISS is publicly available.\n\n"
            "All of these are freely accessible through ESA, USGS, and NASA. No licensing cost."
        ),
        "Vegetation and Agriculture": (
            "**Vegetation and Agriculture — What it is**\n\n"
            "Plants reflect near-infrared (NIR) light strongly and red light weakly. "
            "Healthy vegetation has a high NIR-to-red ratio. Stressed or dying vegetation shows the reverse.\n\n"
            "The NDVI (Normalized Difference Vegetation Index) quantifies this ratio as a number "
            "between -1 and +1. Values above 0.4 indicate healthy green vegetation. "
            "Values near 0 indicate bare soil. Negative values indicate water or snow.\n\n"
            "Key applications:\n"
            "- **Crop health monitoring** — detect stress before it becomes visible to the eye\n"
            "- **Irrigation management** — identify under- or over-watered zones in large fields\n"
            "- **Yield forecasting** — NDVI trends correlate with end-of-season yield\n"
            "- **Deforestation detection** — sudden NDVI drop flags clearing events\n\n"
            "Sentinel-2 at 10m resolution provides NDVI updates every 5 days at no cost."
        ),
        "Water and Environment": (
            "**Water and Environment — What it is**\n\n"
            "Water absorbs near-infrared light almost completely. This makes it easy to "
            "distinguish from land in satellite imagery. The NDWI (Normalized Difference "
            "Water Index) uses green and NIR bands to map surface water extent.\n\n"
            "Key applications:\n"
            "- **Flood mapping** — detect inundation extent within hours of a flood event\n"
            "- **Reservoir monitoring** — track water level and storage volume trends\n"
            "- **Water quality** — turbidity and algal bloom detection using visible and NIR bands\n"
            "- **Wetland health** — track seasonal variation in wetland extent and condition\n\n"
            "Sentinel-2 and Landsat both provide the bands needed for water analysis at no cost. "
            "Landsat adds a 50-year archive for long-term trend analysis."
        ),
        "Atmosphere and Methane": (
            "**Atmosphere and Methane — What it is**\n\n"
            "Sentinel-5P TROPOMI measures the concentration of trace gases in the lower atmosphere "
            "by analyzing how sunlight is absorbed and re-emitted at specific wavelengths. "
            "Different gases absorb at different wavelength signatures — this is how the sensor "
            "distinguishes methane from NO2 from CO.\n\n"
            "Methane (CH4) is the primary concern for energy operators. TROPOMI detects "
            "methane column concentrations at roughly 5 km resolution, daily, globally.\n\n"
            "Key applications:\n"
            "- **Basin-level leak detection** — identify anomalous CH4 concentrations above operating areas\n"
            "- **Regulatory exposure assessment** — compare reported vs. satellite-detected emissions\n"
            "- **Investor disclosure** — support TCFD and SEC climate rule reporting\n\n"
            "Limitation: 5 km pixels cannot pinpoint individual equipment failures. "
            "Attribution requires dispersion modeling on top of the TROPOMI data."
        ),
        "Infrastructure Monitoring": (
            "**Infrastructure Monitoring — What it is**\n\n"
            "EO supports infrastructure monitoring by detecting physical changes in and around "
            "asset corridors. Key techniques:\n\n"
            "- **Change detection** — compare two images from different dates to flag new construction, "
            "clearing, or disturbance within a right-of-way buffer\n"
            "- **SAR coherence** — Sentinel-1 radar tracks millimeter-scale ground movement, "
            "useful for detecting subsidence near facilities or pipelines\n"
            "- **Vegetation encroachment** — NDVI analysis along corridors flags trees or shrubs "
            "approaching clearance limits\n"
            "- **Thermal anomaly detection** — Landsat TIRS can detect heat signatures from "
            "flares, leaks, or uncontrolled combustion\n\n"
            "All of these techniques use free data. The value is in automating routine monitoring "
            "that currently requires expensive aerial surveys or ground patrols."
        ),
        "Climate and Risk": (
            "**Climate and Risk — What it is**\n\n"
            "Physical climate risk analysis uses EO to quantify how often and how severely "
            "an asset is exposed to climate-related hazards. Key risk categories:\n\n"
            "- **Flood risk** — historical inundation frequency from Sentinel-1 SAR and Landsat\n"
            "- **Wildfire risk** — fuel load (NDVI-derived), drought index, and historical burn scars\n"
            "- **Heat stress** — Landsat thermal data maps surface temperature to identify heat islands\n"
            "- **Coastal erosion** — multispectral time series tracks shoreline position change\n\n"
            "EO-derived risk scores are increasingly used in:\n"
            "- Insurance underwriting and pricing\n"
            "- Portfolio stress testing under IPCC scenarios\n"
            "- Regulatory disclosure under TCFD and EU Taxonomy frameworks\n\n"
            "Landsat's 50-year archive is particularly valuable here — it provides the historical "
            "baseline needed to assess return periods for extreme events."
        ),
    }
    return explanations.get(theme, f"No fallback explanation available for theme: {theme}")


def explain_dataset(dataset: str) -> str:
    """Return a detailed plain-English explanation of the given EO dataset."""
    explanations = {
        "Sentinel-2 Multispectral Imagery": (
            "**Sentinel-2 — What it is**\n\n"
            "Sentinel-2 is a pair of satellites operated by the European Space Agency (ESA) "
            "as part of the Copernicus programme. They carry the MultiSpectral Instrument (MSI), "
            "which measures reflected sunlight across 13 spectral bands.\n\n"
            "Key facts:\n"
            "- **Resolution:** 10 meters in the visible and near-infrared bands\n"
            "- **Revisit:** 5 days with both satellites combined\n"
            "- **Coverage:** global, between 84°N and 56°S\n"
            "- **Cost:** free to download via ESA Copernicus Open Access Hub or Microsoft Planetary Computer\n\n"
            "Primary uses: vegetation health (NDVI), water body mapping (NDWI), "
            "urban change detection, wildfire burn scar mapping.\n\n"
            "Key limitation: optical sensor — cloud cover blocks the signal. "
            "Not usable under persistent cloud without combining with SAR data."
        ),
        "Landsat 8/9 Imagery": (
            "**Landsat 8/9 — What it is**\n\n"
            "Landsat is a joint USGS/NASA program that has operated continuously since 1972. "
            "Landsat 8 and 9 carry two instruments: the Operational Land Imager (OLI) for "
            "multispectral bands, and the Thermal Infrared Sensor (TIRS) for heat mapping.\n\n"
            "Key facts:\n"
            "- **Resolution:** 30 meters multispectral, 100 meters thermal\n"
            "- **Revisit:** 16 days per satellite; 8 days combined with Landsat 8 and 9\n"
            "- **Archive depth:** data back to 1972 for trend analysis\n"
            "- **Cost:** free via USGS EarthExplorer or Google Earth Engine\n\n"
            "Primary uses: long-term land cover change, urban heat island mapping, "
            "surface temperature monitoring near industrial facilities.\n\n"
            "Key limitation: 30-meter resolution is coarser than Sentinel-2 for "
            "agriculture applications. 16-day revisit can miss fast events."
        ),
        "Sentinel-5P TROPOMI Atmospheric Data": (
            "**Sentinel-5P TROPOMI — What it is**\n\n"
            "Sentinel-5P carries TROPOMI, a spectrometer that measures trace gas concentrations "
            "in the atmosphere. It works by analyzing how sunlight is absorbed at different "
            "wavelengths as it passes through the atmosphere.\n\n"
            "Key facts:\n"
            "- **Resolution:** 5.5 x 3.5 km per pixel\n"
            "- **Revisit:** daily global coverage\n"
            "- **Gases detected:** CH4, NO2, O3, SO2, CO, HCHO\n"
            "- **Cost:** free via ESA Copernicus or Google Earth Engine\n\n"
            "Primary use for energy operators: methane (CH4) monitoring at basin scale. "
            "NO2 is also used as a proxy for industrial activity and diesel combustion.\n\n"
            "Key limitation: 5 km pixels cannot attribute emissions to a specific facility. "
            "Attribution requires atmospheric dispersion modeling."
        ),
        "GEDI Spaceborne LiDAR": (
            "**GEDI LiDAR — What it is**\n\n"
            "GEDI (Global Ecosystem Dynamics Investigation) is a NASA LiDAR instrument "
            "mounted on the International Space Station. It fires laser pulses downward and "
            "measures the time it takes for the reflected signal to return, building a "
            "vertical profile of vegetation structure.\n\n"
            "Key facts:\n"
            "- **Footprint:** 25-meter circular shots\n"
            "- **Shot spacing:** 600 meters along-track (not wall-to-wall coverage)\n"
            "- **Coverage:** 51.6°N to 51.6°S latitude\n"
            "- **Cost:** free via NASA Earthdata\n\n"
            "Primary uses: canopy height mapping, above-ground biomass estimation, "
            "vegetation encroachment detection along infrastructure corridors.\n\n"
            "Key limitation: sample-based, not continuous. Cannot generate a wall-to-wall "
            "canopy height map without spatial interpolation between shots."
        ),
        "Infrastructure Sample Layer": (
            "**Infrastructure Sample Layer — What it is**\n\n"
            "This is a hardcoded demonstration layer, not real operational data. "
            "It shows a sample gas transmission pipeline corridor in Georgia "
            "as an orange LineString on the map.\n\n"
            "Purpose: demonstrate how infrastructure geometries are overlaid on "
            "satellite base maps, and how spatial queries (buffer analysis, "
            "change detection within a corridor) would work against real pipeline data.\n\n"
            "In a production deployment, this layer would be replaced with actual "
            "pipeline geometry from an operator's GIS system, integrated via a "
            "secure data connection.\n\n"
            "Do not use this layer for any operational, engineering, or regulatory purpose."
        ),
        "Vegetation Health Layer": (
            "**Vegetation Health Layer — What it is**\n\n"
            "This is a sample NDVI-derived classification layer for demonstration purposes. "
            "In a real deployment, it would show three zones for the monitored area:\n\n"
            "- **Green (healthy):** NDVI > 0.4 — dense, active vegetation\n"
            "- **Yellow (stressed):** NDVI 0.2-0.4 — reduced vigor, possible water or nutrient stress\n"
            "- **Red (bare/sparse):** NDVI < 0.2 — minimal vegetation cover\n\n"
            "This layer would update automatically each time a new Sentinel-2 acquisition "
            "becomes available (every 5 days, weather permitting).\n\n"
            "In Day 3 of this program, the live Sentinel-2 pipeline will replace this "
            "static layer with real NDVI calculations from the Planetary Computer API."
        ),
    }
    return explanations.get(dataset, f"No fallback explanation available for dataset: {dataset}")


def explain_use_case(theme: str, dataset: str) -> str:
    """Return a concrete business use case for the given theme and dataset combination."""
    return (
        f"**Business Use Case: {dataset} for {theme}**\n\n"
        f"**Decision it supports:** Monitoring and managing physical conditions related to {theme.lower()} "
        f"using satellite-derived data from {dataset}.\n\n"
        "**How it works:**\n"
        f"1. Acquire free {dataset} data covering the area of interest\n"
        "2. Process the imagery to extract the relevant index or measurement\n"
        "3. Compare current conditions against a baseline or threshold\n"
        "4. Flag anomalies or changes for field verification or operational response\n\n"
        "**Business value:** Replaces or supplements expensive ground surveys with "
        "automated satellite monitoring. Increases the frequency of observation from "
        "quarterly or annual surveys to near-daily updates. Prioritizes where field "
        "resources should be deployed based on data rather than schedule.\n\n"
        "**What it requires:** A geographic boundary for the area of interest, "
        "access to the free satellite archive, and a processing pipeline to compute "
        "and compare the relevant metric over time."
    )


def explain_limitations(dataset: str) -> str:
    """Return a detailed plain-English explanation of the limitations of the given dataset."""
    from data_catalog import DATASETS
    dataset_info = DATASETS.get(dataset, {})
    if dataset_info:
        return (
            f"**Limitations of {dataset}**\n\n"
            f"{dataset_info['limitations']}\n\n"
            "**What this means for analysis:**\n"
            "- Do not rely on this dataset alone for decisions where these limitations matter\n"
            "- Combine with complementary data sources to fill the gaps\n"
            "- For optical sensors: use SAR (Sentinel-1) in areas with persistent cloud cover\n"
            "- For coarse atmospheric data: supplement with dispersion modeling for source attribution\n"
            "- For sample/demonstration layers: replace with real operational data before any production use"
        )
    return f"No limitation detail available for: {dataset}"


def suggest_next_analysis(theme: str) -> str:
    """Return a specific next-step analysis recommendation for the given theme."""
    suggestions = {
        "EO Basics": (
            "**Suggested next step: Run a band combination on Sentinel-2**\n\n"
            "The most practical first exercise in EO is to load a Sentinel-2 scene "
            "and view it in different band combinations.\n\n"
            "- **True color (B4, B3, B2):** what the eye would see from space\n"
            "- **False color NIR (B8, B4, B3):** vegetation appears bright red — useful for crop monitoring\n"
            "- **SWIR (B12, B8A, B4):** highlights burned areas, bare soil, and urban surfaces\n\n"
            "Data source: Microsoft Planetary Computer (free). "
            "Tool: the `pystac-client` and `planetary-computer` libraries introduced on Day 3 of this program."
        ),
        "Vegetation and Agriculture": (
            "**Suggested next step: Compute NDVI for a specific field boundary**\n\n"
            "Define a polygon around a field of interest and compute NDVI from the most recent "
            "Sentinel-2 scene. Compare it to the same field 30 days ago.\n\n"
            "A drop in mean NDVI of more than 0.1 over 30 days is a flag worth investigating. "
            "Causes could include drought, pest damage, disease, or irrigation failure.\n\n"
            "Data source: Sentinel-2 via Planetary Computer (free). "
            "Method: compute (B8 - B4) / (B8 + B4) for each pixel within the boundary."
        ),
        "Water and Environment": (
            "**Suggested next step: Map surface water extent for the past 12 months**\n\n"
            "Use the NDWI index (B3 - B8) / (B3 + B8) on monthly Sentinel-2 composites "
            "to track how the water body extent has changed over the past year.\n\n"
            "Look for: seasonal variation (expected), persistent loss (drought signal), "
            "sudden expansion (flood event).\n\n"
            "Data source: Sentinel-2 via Planetary Computer or Google Earth Engine (both free)."
        ),
        "Atmosphere and Methane": (
            "**Suggested next step: Download TROPOMI methane data for your operating area**\n\n"
            "Query the Sentinel-5P TROPOMI methane product (S5P_OFFL_L2__CH4) via "
            "the Copernicus Data Space API or Google Earth Engine for a 30-day window "
            "over your area of interest.\n\n"
            "Plot the daily CH4 column values. Look for: persistent elevation above the "
            "global background (~1900 ppb), episodic spikes, spatial pattern consistent "
            "with wind direction from a known facility.\n\n"
            "Cost: free. Data latency: approximately 3 hours after acquisition."
        ),
        "Infrastructure Monitoring": (
            "**Suggested next step: Run a change detection analysis on a pipeline corridor**\n\n"
            "Define a 500-meter buffer around the pipeline geometry. Query two Sentinel-2 scenes "
            "separated by 30 days. Compute the difference in NDVI within the buffer.\n\n"
            "Significant NDVI decrease within the buffer indicates ground disturbance — "
            "possible excavation, construction, or vegetation clearing.\n\n"
            "This is the core logic behind automated right-of-way monitoring systems. "
            "It can be run continuously on new Sentinel-2 acquisitions with a small amount of code."
        ),
        "Climate and Risk": (
            "**Suggested next step: Generate a flood frequency map using the Landsat archive**\n\n"
            "Use the MNDWI (Modified NDWI) on Landsat imagery from 1990 to present for your "
            "area of interest. Count how many times each pixel was classified as water across "
            "the archive. Pixels that show as water in 10% or more of scenes are in the "
            "flood-prone zone.\n\n"
            "This produces an empirical flood frequency map based on 30+ years of observation "
            "rather than modeled return periods. More credible for asset-level risk disclosure.\n\n"
            "Data source: Landsat Collection 2 via USGS EarthExplorer or Google Earth Engine (free)."
        ),
    }
    return suggestions.get(theme, f"No next-step suggestion available for theme: {theme}")


# --- Internal exception for authentication failures ---
class _AuthError(Exception):
    """Raised when an API key is present but rejected by the provider."""
    pass
