"""
spectral_explorer.py — Core logic for the Spectral Explorer tab.

Responsibilities:
  - Search Planetary Computer for scenes matching location, date, satellite, cloud cover
  - Render any RGB band combination using the PC rendering API
  - Build a contact sheet (all presets as thumbnails)
  - Build a scene timeline chart
  - Generate AI explanations for band combinations

The black image fix from the Day 2 notebook is baked in:
  - Individual bands (not the visual composite asset)
  - rescale per satellite from satellite_catalog
  - gamma correction
  - Coverage scoring to find the best scene
  - STAC query filtered to avoid tile boundary issues
"""

import requests
import numpy as np
import plotly.graph_objects as go
import pystac_client
import planetary_computer
from PIL import Image
from io import BytesIO
from datetime import datetime

import config
import satellite_catalog
import ai_assistant

# Planetary Computer STAC API endpoint
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Rendering API endpoint
PC_RENDER_URL = "https://planetarycomputer.microsoft.com/api/data/v1/item/preview"


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------

def get_catalog():
    """
    Open and return a signed connection to the Planetary Computer STAC catalog.
    Cached in session state by the caller to avoid reconnecting on every interaction.
    """
    return pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)


def search_scenes(catalog, collection: str, bbox: list, date_range: str,
                  max_cloud: int, cloud_field: str | None) -> list:
    """
    Search the STAC catalog for scenes matching the given parameters.
    Returns a list of STAC items sorted by cloud cover (ascending).
    If cloud_field is None (SAR), no cloud filter is applied.
    """
    query = {}
    if cloud_field:
        query[cloud_field] = {"lt": max_cloud}

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query=query if query else None,
    )

    items = list(search.items())

    # Sort by cloud cover if available, else by date
    if cloud_field:
        items.sort(key=lambda x: x.properties.get(cloud_field, 100))
    else:
        items.sort(key=lambda x: x.datetime, reverse=True)

    return items


def score_scene_coverage(item, r_band: str, g_band: str, b_band: str,
                          satellite_key: str, width: int = 200) -> float:
    """
    Fetch a small test render and count non-zero pixels.
    Returns coverage as a percentage (0-100).
    Used to rank scenes by actual spatial coverage, not just cloud metadata.
    """
    arr = render_combination(item, r_band, g_band, b_band, satellite_key, width=width)
    if arr is None:
        return 0.0
    valid = (arr.max(axis=2) > 5).sum()
    total = arr.shape[0] * arr.shape[1]
    return round(100.0 * valid / total, 1)


def find_best_scene(items: list, r_band: str, g_band: str, b_band: str,
                    satellite_key: str, max_to_check: int = 6) -> tuple:
    """
    Score the top N scenes by coverage and return the best one.
    Returns (item, coverage_pct) or (None, 0) if nothing has valid data.
    """
    best_item    = None
    best_pct     = 0.0
    best_results = []

    for item in items[:max_to_check]:
        pct = score_scene_coverage(item, r_band, g_band, b_band, satellite_key)
        date  = item.datetime.strftime("%Y-%m-%d")
        cloud = item.properties.get("eo:cloud_cover", 0)
        best_results.append((date, cloud, pct, item))
        if pct > best_pct:
            best_pct  = pct
            best_item = item

    return best_item, best_pct, best_results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_combination(item, r_band: str, g_band: str, b_band: str,
                        satellite_key: str, width: int = 600) -> np.ndarray | None:
    """
    Call the Planetary Computer rendering API to produce an RGB image from
    any three bands. Returns a numpy uint8 array or None on failure.

    The black image fix is applied here:
    - Individual named bands (not the visual composite)
    - Per-satellite rescale values from satellite_catalog
    - Gamma correction
    - Crop to valid (non-black) region
    """
    sat     = satellite_catalog.SATELLITES[satellite_key]
    rescale = sat["rescale"]
    gamma   = sat["color_formula"]

    # Build the request parameters. Repeated keys require a list of tuples.
    params = [
        ("collection", sat["collection"]),
        ("item",       item.id),
        ("assets",     r_band),
        ("assets",     g_band),
        ("assets",     b_band),
        ("asset_bidx", f"{r_band}|1"),
        ("asset_bidx", f"{g_band}|1"),
        ("asset_bidx", f"{b_band}|1"),
        ("rescale",    rescale),
        ("rescale",    rescale),
        ("rescale",    rescale),
        ("width",      str(width)),
        ("height",     str(width)),
    ]

    if gamma:
        params.append(("color_formula", gamma))

    try:
        resp = requests.get(PC_RENDER_URL, params=params, timeout=60)
        if resp.status_code != 200:
            return None

        img = Image.open(BytesIO(resp.content)).convert("RGB")
        arr = np.array(img)

        # Crop to the region with actual data (removes black nodata borders)
        arr = _crop_to_valid(arr)
        return arr

    except Exception:
        return None


def render_ndvi(item, satellite_key: str, width: int = 600) -> np.ndarray | None:
    """
    Render an NDVI map using the titiler expression endpoint.
    NDVI = (NIR - Red) / (NIR + Red).
    Uses a red-yellow-green colormap: red = bare/stressed, green = healthy vegetation.

    Only supported for optical satellites (Sentinel-2 and Landsat).
    """
    sat = satellite_catalog.SATELLITES[satellite_key]
    if sat.get("sar"):
        return None

    # Band names differ by satellite
    if satellite_key == "Sentinel-2 L2A":
        expression = "(B08-B04)/(B08+B04)"
    elif satellite_key == "Landsat 8/9":
        expression = "(nir08-red)/(nir08+red)"
    else:
        return None

    params = [
        ("collection",    sat["collection"]),
        ("item",          item.id),
        ("expression",    expression),
        ("colormap_name", "rdylgn"),
        ("rescale",       "-1,1"),
        ("width",         str(width)),
        ("height",        str(width)),
    ]

    try:
        resp = requests.get(PC_RENDER_URL, params=params, timeout=60)
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        arr = np.array(img)
        return _crop_to_valid(arr)
    except Exception:
        return None


def render_ndwi(item, satellite_key: str, width: int = 600) -> np.ndarray | None:
    """
    Render an NDWI map (water index).
    NDWI = (Green - NIR) / (Green + NIR).
    Uses a blue colormap: bright blue = open water.
    """
    sat = satellite_catalog.SATELLITES[satellite_key]
    if sat.get("sar"):
        return None

    if satellite_key == "Sentinel-2 L2A":
        expression = "(B03-B08)/(B03+B08)"
    elif satellite_key == "Landsat 8/9":
        expression = "(green-nir08)/(green+nir08)"
    else:
        return None

    params = [
        ("collection",    sat["collection"]),
        ("item",          item.id),
        ("expression",    expression),
        ("colormap_name", "blues"),
        ("rescale",       "-1,1"),
        ("width",         str(width)),
        ("height",        str(width)),
    ]

    try:
        resp = requests.get(PC_RENDER_URL, params=params, timeout=60)
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        arr = np.array(img)
        return _crop_to_valid(arr)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Contact sheet (all presets as thumbnails)
# ---------------------------------------------------------------------------

def render_contact_sheet(item, satellite_key: str,
                          include_ndvi: bool = True,
                          include_ndwi: bool = True,
                          thumb_size: int = 300) -> list:
    """
    Render every named preset for the selected satellite as a small thumbnail.
    Also renders NDVI and NDWI if requested and supported.

    Returns a list of dicts: [{label, note, array}, ...]
    Arrays are numpy uint8. None means the render failed.
    """
    sat     = satellite_catalog.SATELLITES[satellite_key]
    presets = sat["presets"]
    results = []

    for label, preset in presets.items():
        arr = render_combination(
            item,
            preset["r"], preset["g"], preset["b"],
            satellite_key,
            width=thumb_size,
        )
        results.append({
            "label": label,
            "note":  preset["note"],
            "array": arr,
            "type":  "preset",
        })

    if include_ndvi and not sat.get("sar"):
        arr = render_ndvi(item, satellite_key, width=thumb_size)
        results.append({
            "label": "NDVI",
            "note":  "Vegetation health index. Green = healthy. Red = bare or stressed.",
            "array": arr,
            "type":  "index",
        })

    if include_ndwi and not sat.get("sar"):
        arr = render_ndwi(item, satellite_key, width=thumb_size)
        results.append({
            "label": "NDWI",
            "note":  "Water body index. Bright blue = open water.",
            "array": arr,
            "type":  "index",
        })

    return results


# ---------------------------------------------------------------------------
# Scene timeline chart
# ---------------------------------------------------------------------------

def build_timeline(items: list, cloud_field: str | None,
                   coverage_scores: list = None) -> go.Figure:
    """
    Build a Plotly bar chart showing available scenes over time.
    Bars are coloured by cloud cover: green = clear, yellow = some cloud, red = cloudy.
    Hovering shows the scene date, cloud cover, and coverage score if available.
    """
    if not items:
        return go.Figure()

    scores_map = {}
    if coverage_scores:
        for date, cloud, pct, item in coverage_scores:
            scores_map[item.id] = pct

    dates  = []
    clouds = []
    colors = []
    texts  = []

    for item in items:
        date  = item.datetime.strftime("%Y-%m-%d")
        cloud = item.properties.get(cloud_field, 0) if cloud_field else 0
        pct   = scores_map.get(item.id, None)

        dates.append(date)
        clouds.append(cloud)

        # Colour by cloud cover
        if cloud < 5:
            colors.append("#2ecc71")     # green
        elif cloud < 15:
            colors.append("#f39c12")     # amber
        else:
            colors.append("#e74c3c")     # red

        hover = f"Date: {date}<br>Cloud: {cloud:.1f}%"
        if pct is not None:
            hover += f"<br>Coverage: {pct:.0f}%"
        texts.append(hover)

    fig = go.Figure(go.Bar(
        x=dates,
        y=[1] * len(dates),
        marker_color=colors,
        hovertext=texts,
        hoverinfo="text",
    ))

    fig.update_layout(
        title="Available scenes (green = clear, amber = some cloud, red = cloudy)",
        xaxis_title="Date",
        yaxis_visible=False,
        height=160,
        margin=dict(l=10, r=10, t=40, b=40),
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="#ffffff",
    )
    return fig


# ---------------------------------------------------------------------------
# AI explanation
# ---------------------------------------------------------------------------

def explain_combination(r_band: str, g_band: str, b_band: str,
                         satellite_key: str, location_name: str,
                         is_index: bool = False, index_name: str = "") -> str:
    """
    Generate an AI explanation of what the selected band combination reveals
    for the given location. Uses the same fallback chain as the rest of the app.
    """
    sat   = satellite_catalog.SATELLITES[satellite_key]
    bands = sat["bands"]

    if is_index:
        prompt = (
            f"Explain what the {index_name} index shows when applied to {satellite_key} "
            f"imagery over {location_name}. Cover: what the index measures, what the "
            f"colour scale means (bright vs dark), and one specific thing to look for "
            f"in this location. Plain language, under 150 words."
        )
    else:
        r_name = bands.get(r_band, {}).get("name", r_band)
        g_name = bands.get(g_band, {}).get("name", g_band)
        b_name = bands.get(b_band, {}).get("name", b_band)
        prompt = (
            f"Explain what a satellite image with Red={r_name}, Green={g_name}, "
            f"Blue={b_name} reveals when looking at {location_name} from {satellite_key}. "
            f"Cover: what stands out in each colour, what land cover types are identifiable, "
            f"and one specific insight for this location. Plain language, under 150 words."
        )

    return ai_assistant.auto_explain(
        theme="EO Basics",
        dataset=satellite_key,
        location=location_name,
        mode="Explain selected dataset",
    ) if not config.has_any_key() else ai_assistant.ask(
        question=prompt,
        theme="EO Basics",
        dataset=satellite_key,
        location=location_name,
        mode="Explain selected dataset",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _crop_to_valid(arr: np.ndarray, threshold: int = 5) -> np.ndarray:
    """
    Crop a numpy image array to the bounding box of non-black pixels.
    Removes the large black nodata borders that satellite tiles contain.
    If the entire image is black, returns the original array unchanged.
    """
    valid = arr.max(axis=2) > threshold
    if not valid.any():
        return arr
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    return arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
