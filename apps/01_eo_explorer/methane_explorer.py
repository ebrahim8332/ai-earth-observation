"""
methane_explorer.py — Emissions Explorer module (Day 12)
Fetches TROPOMI Sentinel-5P data from Google Earth Engine for four atmospheric gases.

This module provides:
- GAS_CONFIG: configuration for CH4, NO2, CO, and SO2
- fetch_tropomi_mosaic(): daily mean image from all orbital passes
- get_regional_mean(): regional mean concentration (bestEffort at native scale)
- build_emissions_map(): Folium map with GEE tile layer
- get_emissions_interpretation(): AI interpretation using the shared ai_chain
- build_interpretation_prompt(): structured four-section prompt for the AI
"""

import math
import ee
import folium
import branca.colormap as cm
import streamlit as st
from datetime import datetime, timedelta

from ai_chain import complete

# ---------------------------------------------------------------------------
# Gas configuration
# One entry per gas. All parameters confirmed against real TROPOMI data.
# ---------------------------------------------------------------------------

GAS_CONFIG = {
    "Methane (CH4)": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CH4",
        "band":       "CH4_column_volume_mixing_ratio_dry_air",
        "unit":       "ppb",
        "min_val":    1860,
        "max_val":    1960,
        "palette":    [
            "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
            "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
        ],
        "description": (
            "Methane column mixing ratio in dry air. "
            "Background atmosphere is ~1900 ppb. "
            "Oil and gas operations, coal mines, and landfills elevate readings locally."
        ),
        "utility_relevance": (
            "Gas pipeline leaks, LNG facilities, coal seam gas, landfill emissions. "
            "Readings above 1950 ppb over an industrial region indicate significant fugitive emissions."
        ),
    },
    "Nitrogen Dioxide (NO2)": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band":       "NO2_column_number_density",
        "unit":       "mol/m²",
        "min_val":    0,
        "max_val":    0.0002,
        "palette":    [
            "#ffffff", "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
            "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026",
        ],
        "description": (
            "NO2 tropospheric column density. "
            "Background is near zero. Industrial areas and cities show elevated values. "
            "Values above 0.0001 mol/m² are considered elevated. Above 0.0002 is high."
        ),
        "utility_relevance": (
            "Power plant combustion, fleet operations, and regulatory compliance monitoring. "
            "Useful for identifying which industrial zones are exceeding emission thresholds."
        ),
    },
    "Carbon Monoxide (CO)": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "band":       "CO_column_number_density",
        "unit":       "mol/m²",
        "min_val":    0.02,
        "max_val":    0.05,
        "palette":    [
            "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
            "#4292c6", "#2171b5", "#08519c", "#08306b",
        ],
        "description": (
            "CO total column density. "
            "Background levels are 0.02-0.03 mol/m². "
            "Wildfire plumes and industrial combustion push values to 0.05+ mol/m²."
        ),
        "utility_relevance": (
            "Industrial fires, wildfire risk to infrastructure, and emergency response planning. "
            "Elevated CO over a region indicates active combustion within or upwind of the area."
        ),
    },
    "Sulfur Dioxide (SO2)": {
        "collection": "COPERNICUS/S5P/OFFL/L3_SO2",
        "band":       "SO2_column_number_density",
        "unit":       "mol/m²",
        "min_val":    0,
        "max_val":    0.001,
        "palette":    [
            "#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679",
            "#41ab5d", "#238443", "#006837", "#004529",
        ],
        "description": (
            "SO2 total column density. "
            "Background is near zero. Volcanic degassing and coal plant stacks raise values. "
            "Values above 0.0003 mol/m² are notable."
        ),
        "utility_relevance": (
            "Industrial site emissions, coal plant stack monitoring, and compliance tracking. "
            "Also used for volcanic hazard monitoring when SO2 spikes precede eruptions."
        ),
    },
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Minimum bbox side in degrees for TROPOMI analysis.
# TROPOMI pixels are ~5-7 km. A 3-degree region (~300 km) gives enough
# coverage for meaningful statistics and a readable map.
MIN_SPAN_DEG = 3.0


def pad_bbox(bbox):
    """Expand a bbox to at least MIN_SPAN_DEG on each side.

    Geocoders often return a point or tiny bbox for industrial regions
    (e.g. 'Permian Basin, Texas' → a single coordinate). TROPOMI data
    at a point is meaningless — we need a regional view.
    """
    west, south, east, north = bbox
    cx = (west + east) / 2
    cy = (south + north) / 2
    half_w = max((east - west) / 2,  MIN_SPAN_DEG / 2)
    half_h = max((north - south) / 2, MIN_SPAN_DEG / 2)
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def bbox_to_geometry(bbox):
    """Convert a [west, south, east, north] bbox list to an ee.Geometry.Rectangle."""
    west, south, east, north = bbox
    return ee.Geometry.Rectangle([west, south, east, north])


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_tropomi_mosaic(gas_key, bbox, target_date, composite_days=7):
    """Fetch a TROPOMI composite image for the given gas and date.

    Aggregates all orbital passes within a composite_days window centred
    on target_date (default: 7 days). This gives solid regional coverage
    because 7 days × ~14 passes/day = ~98 passes. After quality masking,
    most areas on Earth have valid data within any 7-day window.

    Single-day composites leave large masked areas because TROPOMI's
    quality filter removes pixels with cloud contamination, and one day
    is rarely enough to fill a region completely.

    Returns:
        image        (ee.Image or None)  — the composite mean image
        window_label (str or None)       — date window label for display
        pass_count   (int)               — number of orbital passes in window
    """
    cfg  = GAS_CONFIG[gas_key]
    bbox = pad_bbox(bbox)   # ensure region is at least MIN_SPAN_DEG wide
    target_dt  = datetime.strptime(target_date, "%Y-%m-%d")
    half       = composite_days // 2
    win_start  = (target_dt - timedelta(days=half)).strftime("%Y-%m-%d")
    win_end    = (target_dt + timedelta(days=half + 1)).strftime("%Y-%m-%d")

    col = (
        ee.ImageCollection(cfg["collection"])
        .filterDate(win_start, win_end)
        .select(cfg["band"])
    )

    count = col.size().getInfo()
    if count > 0:
        window_label = f"{win_start} to {win_end}"
        return col.mean(), window_label, count

    return None, None, 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def get_regional_mean(image, bbox, band_name):
    """Compute the regional mean concentration using point sampling.

    Samples the image at a 3×3 grid of points spread across the bbox and
    returns the mean of all valid sample values. Point sampling bypasses
    the projection issues that affect reduceRegion on TROPOMI composites.

    Returns a float, or None if no valid samples are found.
    """
    try:
        bbox = pad_bbox(bbox)
        west, south, east, north = bbox
        # Build a 3x3 grid of sample points across the bbox
        lons = [west + (east  - west) * f for f in [0.25, 0.5, 0.75]]
        lats = [south + (north - south) * f for f in [0.25, 0.5, 0.75]]

        features = [
            ee.Feature(ee.Geometry.Point([lon, lat]))
            for lon in lons
            for lat in lats
        ]
        fc = ee.FeatureCollection(features)

        sampled = image.sampleRegions(
            collection = fc,
            scale      = 111320,
            geometries = False,
        )

        # Pull all sample values and average the non-null ones in Python
        rows = sampled.getInfo().get("features", [])
        values = [
            r["properties"][band_name]
            for r in rows
            if r["properties"].get(band_name) is not None
        ]

        if values:
            return sum(values) / len(values)
        return None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Map building
# ---------------------------------------------------------------------------

def build_emissions_map(image, gas_key, bbox, actual_date):
    """Build a Folium map with a TROPOMI tile layer.

    Uses GEE's getMapId() to get a tile URL with embedded auth token —
    the same pattern used in gee_sar.py and gee_change.py. The browser
    fetches tiles directly from GEE; no data passes through the app server.

    Returns:
        folium.Map with the gas layer added, or None on failure.
    """
    cfg  = GAS_CONFIG[gas_key]
    bbox = pad_bbox(bbox)

    try:
        vis_params = {
            "min":     cfg["min_val"],
            "max":     cfg["max_val"],
            "palette": cfg["palette"],
        }

        map_id    = image.getMapId(vis_params)
        tile_url  = map_id["tile_fetcher"].url_format

        west, south, east, north = bbox
        cx = (west + east) / 2
        cy = (south + north) / 2

        # Calculate zoom from bbox span so the region fills the map frame.
        # Calibrated so Texas (~11°) → zoom 5, a basin (~5°) → zoom 6,
        # a city region (~1°) → zoom 8.
        span = max(east - west, north - south)
        zoom = max(4, min(8, round(7 - math.log2(max(span, 0.5)))))
        m = folium.Map(location=[cy, cx], zoom_start=zoom, tiles="CartoDB positron")

        folium.TileLayer(
            tiles   = tile_url,
            attr    = "Google Earth Engine / Copernicus TROPOMI",
            name    = f"{gas_key} — {actual_date}",
            overlay = True,
            control = True,
            opacity = 0.85,
        ).add_to(m)

        # Colorbar legend — default branca positioning (bottom-right of map)
        colormap = cm.LinearColormap(
            colors   = cfg["palette"],
            vmin     = cfg["min_val"],
            vmax     = cfg["max_val"],
            caption  = f"{gas_key} ({cfg['unit']})",
        )
        colormap.width = 300
        colormap.add_to(m)

        folium.LayerControl().add_to(m)

        return m

    except Exception:
        return None


# ---------------------------------------------------------------------------
# AI interpretation
# ---------------------------------------------------------------------------

def build_interpretation_prompt(gas_key, mean_val, region, actual_date):
    """Build a structured four-section prompt for the AI.

    The four sections are designed to deliver operational value:
    PATTERN / SOURCE ATTRIBUTION / REGULATORY CONTEXT / UTILITY ACTION
    """
    cfg = GAS_CONFIG[gas_key]

    if mean_val is not None:
        val_str = f"{mean_val:.4f} {cfg['unit']}"
    else:
        val_str = f"data unavailable ({cfg['unit']})"

    prompt = f"""You are an environmental data analyst interpreting satellite atmospheric measurements for a utility or infrastructure operator.

TROPOMI Sentinel-5P data — {gas_key}
Region: {region}
Period: {actual_date} (7-day composite)
Regional average: {val_str}
Data source: Copernicus Sentinel-5P TROPOMI (European Space Agency)

{cfg['description']}

Relevance to utilities and industry:
{cfg['utility_relevance']}

Write a structured interpretation with exactly four labeled sections:

PATTERN
Describe what the regional average value indicates. Is it elevated, normal, or low compared to background levels? What does the spatial pattern visible in the data suggest about where concentrations are highest?

SOURCE ATTRIBUTION
What are the most likely emission sources producing this reading at this location and time of year? Be specific to the region and the gas. Consider industrial activity, seasonal factors, and natural sources.

REGULATORY CONTEXT
What emission standards or reporting obligations apply to this gas in this type of region? What would trigger a regulatory notification or required investigation?

UTILITY ACTION
What specific action should a utility operator, infrastructure manager, or environmental compliance team take based on this reading? Be concrete and operational.

Keep each section to 3-5 sentences. Use plain language. No bullet points within sections."""

    return prompt


def get_emissions_interpretation(gas_key, mean_val, region, actual_date,
                                  groq_key="", gemini_key=""):
    """Call the shared AI chain and return (interpretation_text, model_name).

    Falls back to built-in text if no API keys are configured.
    """
    prompt = build_interpretation_prompt(gas_key, mean_val, region, actual_date)
    text, model = complete(prompt, groq_key=groq_key, gemini_key=gemini_key)

    if text:
        return text, model

    # Built-in fallback — substantive, not a placeholder
    cfg = GAS_CONFIG[gas_key]
    fallback = (
        f"**{gas_key} — {region} — {actual_date}**\n\n"
        f"Regional average: {mean_val:.4f} {cfg['unit']} (if available).\n\n"
        f"{cfg['description']}\n\n"
        f"**Utility relevance:** {cfg['utility_relevance']}\n\n"
        "Add a GROQ_API_KEY or GEMINI_API_KEY to Streamlit secrets to enable full AI interpretation."
    )
    return fallback, None
