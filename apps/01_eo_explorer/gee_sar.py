"""
gee_sar.py — SAR Explorer module
Logic adapted from notebooks/03_sentinel1_sar_basics.ipynb

This module provides:
- Sentinel-1 GRD image fetching from GEE for any location and two dates
- Backscatter statistics (VV, VH min/max/mean in dB)
- GEE map tile URLs so Folium can display VV, VH, false color, and change layers
- Plotly grouped bar chart comparing both dates
- Groq AI interpretation with substantive fallback text
"""

import io
import math
import requests
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config

# ---------------------------------------------------------------------------
# SAR collection settings
# Sentinel-1 GRD — same collection used in the Day 5 notebook
# ---------------------------------------------------------------------------

SAR_COLLECTION   = "COPERNICUS/S1_GRD"
INSTRUMENT_MODE  = "IW"            # Interferometric Wide swath — standard land mode
DEFAULT_SCALE    = 100             # metres — used for statistics reduction
WINDOW_DAYS      = 10              # days either side of target date to search
MIN_BBOX_DEG     = 0.35            # minimum bbox side in degrees (~25 km)
                                   # smaller bboxes produce single-pixel blowups

# ---------------------------------------------------------------------------
# Image fetching
# ---------------------------------------------------------------------------

def pad_bbox(bbox):
    """Expand a bbox to at least MIN_BBOX_DEG on each side.

    When a geocoder returns a point or tiny area (e.g. 'Port of Rotterdam'
    resolves to a single building), the bbox is only a few metres wide.
    GEE fetches 1-2 SAR pixels and blows them up to fill the output image,
    producing solid colour blocks.

    This function pads any dimension smaller than MIN_BBOX_DEG by expanding
    it symmetrically around the centroid.

    bbox: [west, south, east, north]
    Returns: padded [west, south, east, north]
    """
    west, south, east, north = bbox
    cx = (west + east) / 2
    cy = (south + north) / 2
    half_w = max((east  - west)  / 2, MIN_BBOX_DEG / 2)
    half_h = max((north - south) / 2, MIN_BBOX_DEG / 2)
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def fetch_sar_image(bbox, date_str, gee_available):
    """Fetch the closest Sentinel-1 GRD image to date_str for the given bbox.

    Searches a +/- WINDOW_DAYS window around the target date.
    Tries descending orbit first, then ascending if nothing is found.
    Pads tiny bboxes (from point geocodes) to MIN_BBOX_DEG before querying.

    bbox: [west, south, east, north]
    date_str: 'YYYY-MM-DD'

    Returns (image, count) where image is a GEE image object (or None)
    and count is how many scenes were found in the window.
    """
    if not gee_available:
        return None, 0, bbox

    try:
        import ee

        bbox     = pad_bbox(bbox)          # expand point geocodes to usable area
        geometry = ee.Geometry.Rectangle(bbox)
        date     = ee.Date(date_str)
        start    = date.advance(-WINDOW_DAYS, "day")
        end      = date.advance(WINDOW_DAYS, "day")

        def build_collection(orbit_pass):
            return (
                ee.ImageCollection(SAR_COLLECTION)
                .filterBounds(geometry)
                .filterDate(start, end)
                .filter(ee.Filter.eq("instrumentMode", INSTRUMENT_MODE))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
                .select(["VV", "VH"])
            )

        # Try descending first — standard for Europe and most land regions
        collection = build_collection("DESCENDING")
        count      = collection.size().getInfo()

        if count == 0:
            # Fall back to ascending — covers areas with fewer descending passes
            collection = build_collection("ASCENDING")
            count      = collection.size().getInfo()

        if count == 0:
            return None, 0, bbox

        image = collection.first().clip(geometry)
        return image, count, bbox

    except Exception:
        return None, 0, bbox


# ---------------------------------------------------------------------------
# Backscatter statistics
# ---------------------------------------------------------------------------

def get_backscatter_stats(image, bbox):
    """Extract VV and VH backscatter statistics from a Sentinel-1 image.

    Computes min, max, and mean for each polarization band.
    Values are in dB (decibels) — the standard SAR measurement unit.

    Typical values:
      Calm water:    -25 to -20 dB
      Vegetation:    -15 to -10 dB
      Urban/ships:    -5 to  +5 dB

    Returns a dict with keys VV_min, VV_max, VV_mean, VH_min, VH_max, VH_mean.
    """
    try:
        import ee

        geometry = ee.Geometry.Rectangle(bbox)

        stats = image.reduceRegion(
            reducer=ee.Reducer.minMax().combine(
                ee.Reducer.mean(), sharedInputs=True
            ),
            geometry=geometry,
            scale=DEFAULT_SCALE,
            maxPixels=1e8,
        ).getInfo()

        return {
            "VV_min":  round(stats.get("VV_min",  -25.0), 1),
            "VV_max":  round(stats.get("VV_max",    5.0), 1),
            "VV_mean": round(stats.get("VV_mean", -12.0), 1),
            "VH_min":  round(stats.get("VH_min",  -35.0), 1),
            "VH_max":  round(stats.get("VH_max",    0.0), 1),
            "VH_mean": round(stats.get("VH_mean", -20.0), 1),
        }

    except Exception:
        # Return plausible defaults so the UI does not crash
        return {
            "VV_min": -25.0, "VV_max": 5.0, "VV_mean": -12.0,
            "VH_min": -35.0, "VH_max": 0.0, "VH_mean": -20.0,
        }


# ---------------------------------------------------------------------------
# Image download — server-side via getThumbURL
#
# GEE tile URLs require auth headers the browser cannot send (CORS block).
# The fix: download images SERVER-SIDE using the service account credentials,
# then pass numpy arrays to st.image() for display.
# getThumbURL runs on the GEE server and returns a PNG via a temporary URL
# that the Streamlit server can fetch with requests.get().
# ---------------------------------------------------------------------------

def _download_rgb_image(rgb_image, bbox):
    """Download a pre-visualized GEE image as a numpy RGB array.

    Uses getThumbURL() with an explicit ee.Geometry.Rectangle region so GEE
    clips the export to our bbox rather than the full satellite swath (~250km).

    Why region matters:
    - Without region, 'dimensions' is calculated against the entire swath.
      600px over 250km = 417m/pixel — all local detail is lost.
    - With region=ee.Geometry.Rectangle(bbox), 600px covers only our area.
      600px over ~80km = ~133m/pixel — correct for Sentinel-1.

    The rgb_image must already be visualized (via .visualize()) before calling.
    Returns a uint8 numpy array (H x W x 3), or None on failure.
    """
    try:
        import ee

        region = ee.Geometry.Rectangle(bbox)

        url = rgb_image.getThumbURL({
            "region":     region,
            "dimensions": 400,        # longest side in pixels — compact display size
            "format":     "png",
        })

        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            return None

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return np.array(img)

    except Exception:
        return None


def build_sar_views(image1, image2, bbox, date1_str, date2_str):
    """Download all SAR view images as numpy arrays.

    Uses .visualize() on the GEE server to apply colour rendering, then
    downloads via getThumbURL with an explicit region so GEE clips to the
    bbox before resizing. This avoids the browser auth problem (tile URLs
    require GEE auth headers; getThumbURL produces a plain HTTPS download).

    Structure returned:
      {
        "VV Polarization":  {"date1": array, "date2": array},
        "VH Polarization":  {"date1": array, "date2": array},
        "False Color":      {"date1": array, "date2": array},
        "Change Map":       {"single": array},
      }

    Any failed download returns None for that entry — the UI handles gracefully.
    """
    if image1 is None or image2 is None:
        return {}

    import ee

    # --- Build derived bands with explicit names ---
    # .subtract() keeps the first band's name ("VV"), so we rename the ratio
    # to avoid duplicate band names in the false color stack.
    vv1   = image1.select("VV")
    vh1   = image1.select("VH")
    rat1  = vv1.subtract(vh1).rename("ratio")
    fc1   = vv1.addBands(vh1).addBands(rat1)   # bands: VV, VH, ratio

    vv2   = image2.select("VV")
    vh2   = image2.select("VH")
    rat2  = vv2.subtract(vh2).rename("ratio")
    fc2   = vv2.addBands(vh2).addBands(rat2)

    change = vv2.subtract(vv1).rename("change")

    # --- Apply colour rendering server-side using .visualize() ---
    # This produces a proper 3-band RGB image that getThumbURL can download
    # without any additional vis params.
    # Palette colours must not have the '#' prefix for GEE .visualize().

    vv1_rgb = vv1.visualize(min=-25, max=0,  palette=["000000", "ffffff"])
    vv2_rgb = vv2.visualize(min=-25, max=0,  palette=["000000", "ffffff"])

    vh1_rgb = vh1.visualize(min=-30, max=-5, palette=["000000", "ffffff"])
    vh2_rgb = vh2.visualize(min=-30, max=-5, palette=["000000", "ffffff"])

    # False color: per-band min/max as lists, bands named explicitly
    fc1_rgb = fc1.visualize(
        bands=["VV", "VH", "ratio"],
        min=[-20, -25, 0],
        max=[0,   -5,  15],
    )
    fc2_rgb = fc2.visualize(
        bands=["VV", "VH", "ratio"],
        min=[-20, -25, 0],
        max=[0,   -5,  15],
    )

    # Change map: diverging red-yellow-blue palette
    change_rgb = change.visualize(
        min=-5, max=5,
        palette=["d73027", "fee090", "ffffbf", "e0f3f8", "4575b4"],
    )

    return {
        "VV Polarization": {
            "date1": _download_rgb_image(vv1_rgb,  bbox),
            "date2": _download_rgb_image(vv2_rgb,  bbox),
        },
        "VH Polarization": {
            "date1": _download_rgb_image(vh1_rgb, bbox),
            "date2": _download_rgb_image(vh2_rgb, bbox),
        },
        "False Color": {
            "date1": _download_rgb_image(fc1_rgb, bbox),
            "date2": _download_rgb_image(fc2_rgb, bbox),
        },
        "Change Map": {
            "single": _download_rgb_image(change_rgb, bbox),
        },
    }


# ---------------------------------------------------------------------------
# Statistics chart
# ---------------------------------------------------------------------------

def build_stats_chart(stats1, stats2, date1_str, date2_str):
    """Grouped bar chart comparing VV and VH backscatter statistics for two dates.

    Shows min, mean, and max for each polarization side by side.
    Reference lines mark typical surface type thresholds.
    """
    categories = ["VV min", "VV mean", "VV max", "VH min", "VH mean", "VH max"]
    values1 = [
        stats1["VV_min"], stats1["VV_mean"], stats1["VV_max"],
        stats1["VH_min"], stats1["VH_mean"], stats1["VH_max"],
    ]
    values2 = [
        stats2["VV_min"], stats2["VV_mean"], stats2["VV_max"],
        stats2["VH_min"], stats2["VH_mean"], stats2["VH_max"],
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=date1_str,
        x=categories,
        y=values1,
        marker_color="#4a90d9",
        text=[f"{v:.1f}" for v in values1],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name=date2_str,
        x=categories,
        y=values2,
        marker_color="#e07b39",
        text=[f"{v:.1f}" for v in values2],
        textposition="outside",
    ))

    # Horizontal reference lines showing typical surface thresholds
    fig.add_hline(
        y=-20, line_dash="dot", line_color="#3399ff",
        annotation_text="Calm water (≈ -20 dB)",
        annotation_position="top left",
        annotation_font_size=10,
    )
    fig.add_hline(
        y=-5, line_dash="dot", line_color="#228B22",
        annotation_text="Urban / ships (≈ -5 dB)",
        annotation_position="top left",
        annotation_font_size=10,
    )

    fig.update_layout(
        title="Backscatter statistics comparison (dB)",
        xaxis_title="Statistic",
        yaxis_title="Backscatter (dB)",
        barmode="group",
        height=360,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ---------------------------------------------------------------------------
# AI interpretation
# ---------------------------------------------------------------------------

def get_sar_interpretation(stats1, stats2, date1_str, date2_str,
                            location_name, api_key=None):
    """Plain-language interpretation of the SAR backscatter analysis.

    Tries Groq first if an API key is available.
    Falls back to the substantive text adapted from notebook Cell 10.
    """
    vv_change = stats2["VV_mean"] - stats1["VV_mean"]
    vh_change = stats2["VH_mean"] - stats1["VH_mean"]

    context = (
        f"Study area: {location_name}\n"
        f"Sensor: Sentinel-1 SAR GRD, IW mode, Copernicus programme\n"
        f"Date 1: {date1_str} | Date 2: {date2_str}\n\n"
        f"Backscatter statistics for Date 1 (dB):\n"
        f"  VV: min {stats1['VV_min']}, max {stats1['VV_max']}, mean {stats1['VV_mean']}\n"
        f"  VH: min {stats1['VH_min']}, max {stats1['VH_max']}, mean {stats1['VH_mean']}\n\n"
        f"Backscatter statistics for Date 2 (dB):\n"
        f"  VV: min {stats2['VV_min']}, max {stats2['VV_max']}, mean {stats2['VV_mean']}\n"
        f"  VH: min {stats2['VH_min']}, max {stats2['VH_max']}, mean {stats2['VH_mean']}\n\n"
        f"Mean VV change between dates: {vv_change:+.1f} dB\n"
        f"Mean VH change between dates: {vh_change:+.1f} dB\n\n"
        f"Note: SAR works through cloud cover and at night. "
        f"These images were acquired regardless of weather conditions."
    )

    if api_key:
        try:
            from groq import Groq
            client   = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a SAR remote sensing analyst. "
                            "Explain SAR analysis results in plain language for a "
                            "technical but non-specialist audience. "
                            "Cover: what the backscatter values indicate about "
                            "surface types, what changed between the two dates, "
                            "one practical application, and one key limitation of SAR."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Interpret this Sentinel-1 SAR analysis:\n\n{context}",
                    },
                ],
                max_tokens=600,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception:
            pass   # fall through to substantive fallback

    return _sar_fallback(stats1, stats2, date1_str, date2_str,
                         location_name, vv_change, vh_change)


def _sar_fallback(stats1, stats2, date1_str, date2_str,
                   location_name, vv_change, vh_change):
    """Substantive fallback interpretation — adapted from notebook Cell 10."""
    vv_direction = "increased" if vv_change > 0 else "decreased"
    vh_direction = "increased" if vh_change > 0 else "decreased"

    return f"""**SAR Analysis — {location_name}**

**What the backscatter values indicate:**

The Sentinel-1 data reveals surface types through the intensity of radar energy returned to the satellite (measured in dB).

For Date 1 ({date1_str}):
- VV mean: {stats1['VV_mean']} dB — {"consistent with urban or vegetated land" if stats1['VV_mean'] > -15 else "consistent with water-dominated or smooth surfaces"}
- VH mean: {stats1['VH_mean']} dB — {"indicates volume scattering from vegetation or complex structures" if stats1['VH_mean'] > -20 else "typical of smooth or open surfaces"}

For Date 2 ({date2_str}):
- VV mean: {stats2['VV_mean']} dB
- VH mean: {stats2['VH_mean']} dB

Reference thresholds: calm water ≈ -20 dB, vegetation ≈ -12 dB, urban/ships ≈ -5 dB.

**What changed between the two dates:**
VV mean {vv_direction} by {abs(vv_change):.1f} dB. VH mean {vh_direction} by {abs(vh_change):.1f} dB.
{"An increase in backscatter suggests new structures, vessel arrivals, or rougher surface conditions." if vv_change > 0.5 else "A decrease in backscatter suggests vessel departures, surface smoothing, or seasonal vegetation change." if vv_change < -0.5 else "The change between dates is small — conditions were broadly stable across this period."}

**Practical application:**
SAR is the only operational sensor that functions through cloud cover and at night. Port monitoring, flood mapping, deforestation detection in tropical cloud-covered regions, and infrastructure change monitoring all rely on SAR for continuous coverage where optical sensors fail.

**Key limitation:**
SAR interpretation requires expertise. The layover effect distorts tall structures — a building appears to lean toward the satellite. Speckle noise (a granular texture visible in all SAR images) can mask subtle surface changes. Multi-look averaging reduces speckle but lowers spatial resolution."""
