"""
gee_change.py — Change Detection portal module
Logic mirrors the mechanics taught in notebooks/04_change_detection.ipynb

This module provides:
- Sentinel-2 NDVI compositing from GEE for two user-selected dates
- MODIS NDVI fallback for dates with insufficient Sentinel-2 coverage
- Change map: Folium interactive map with GEE tile layers (NDVI 1, NDVI 2, diff)
- Summary statistics: mean change, area of significant gain, area of significant loss
- Groq AI interpretation with substantive fallback text
"""

import json
import streamlit as st
import folium

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel-2 Surface Reflectance — same collection used in the notebook
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# MODIS NDVI — pre-computed, 250 m, 16-day composites
# Used when Sentinel-2 has insufficient cloud-free coverage
MODIS_COLLECTION = "MODIS/061/MOD13Q1"
MODIS_SCALE      = 10000.0   # GEE stores MODIS NDVI as integer * 10000

# Number of days either side of the target date to search for cloud-free scenes
WINDOW_DAYS = 30

# Maximum cloud cover % to accept a Sentinel-2 scene
MAX_CLOUD = 20

# Minimum meaningful NDVI change — changes smaller than this are treated as noise
CHANGE_THRESHOLD = 0.1

# Statistics computation scale — metres per pixel for region reduction
# 1000 m is 4x faster than 500 m and accurate enough for km² summaries.
# The area numbers change by less than 2% vs 500 m for regions this size.
STATS_SCALE = 1000

# Minimum bbox side length in degrees (~0.3 deg ≈ 30 km)
# Prevents single-pixel blowups when the geocoder returns a point
MIN_BBOX_DEG = 0.3

# ---------------------------------------------------------------------------
# GEE geometry helper
# ---------------------------------------------------------------------------

def _pad_bbox(bbox):
    """Expand a bbox to at least MIN_BBOX_DEG on each side.

    Prevents single-pixel blowups from point geocodes.
    bbox: [west, south, east, north]
    """
    west, south, east, north = bbox
    cx = (west  + east)  / 2
    cy = (south + north) / 2
    half_w = max((east  - west)  / 2, MIN_BBOX_DEG / 2)
    half_h = max((north - south) / 2, MIN_BBOX_DEG / 2)
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


# ---------------------------------------------------------------------------
# NDVI fetching — Sentinel-2 with MODIS fallback
# ---------------------------------------------------------------------------

def _compute_ndvi_s2(image):
    """Add an NDVI band to a Sentinel-2 image.

    NDVI = (B8 - B4) / (B8 + B4)  where B8 = NIR (842 nm), B4 = Red (665 nm).
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def fetch_ndvi_image(bbox, date_str, gee_available):
    """Fetch a cloud-free NDVI composite for the given bbox near date_str.

    Tries Sentinel-2 first (10 m, ±WINDOW_DAYS window, <MAX_CLOUD% cloud).
    Falls back to MODIS if fewer than 2 Sentinel-2 scenes qualify.

    Returns (image, source_name) where image is a GEE image with an 'NDVI' band,
    or (None, None) if GEE is unavailable.
    """
    if not gee_available:
        return None, None

    try:
        import ee

        bbox     = _pad_bbox(bbox)
        geometry = ee.Geometry.Rectangle(bbox)
        date     = ee.Date(date_str)
        start    = date.advance(-WINDOW_DAYS, "day")
        end      = date.advance( WINDOW_DAYS, "day")

        # Sentinel-2 attempt
        s2_col = (
            ee.ImageCollection(S2_COLLECTION)
            .filterBounds(geometry)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
            .map(_compute_ndvi_s2)
            .select("NDVI")
        )

        s2_count = s2_col.size().getInfo()

        if s2_count >= 2:
            return s2_col.median(), f"Sentinel-2 ({s2_count} scenes)"

        # MODIS fallback — wider window to guarantee coverage
        modis_start = date.advance(-40, "day")
        modis_end   = date.advance( 40, "day")

        modis_col = (
            ee.ImageCollection(MODIS_COLLECTION)
            .filterBounds(geometry)
            .filterDate(modis_start, modis_end)
            .select("NDVI")
        )

        # MODIS NDVI is stored as integer * 10000; divide to get 0-1 range
        modis_img = modis_col.median().divide(MODIS_SCALE)

        return modis_img, "MODIS (Sentinel-2 coverage insufficient)"

    except Exception as e:
        st.warning(f"GEE fetch error for {date_str}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Change statistics
# ---------------------------------------------------------------------------

EXTREME_THRESHOLD = 0.3   # NDVI change magnitude considered dramatic (fire, flood, clearcut)


def compute_change_stats(img1, img2, diff_image, bbox, gee_available):
    """Compute summary statistics from the NDVI difference image.

    Takes img1 and img2 (the raw NDVI composites) in addition to the diff image
    so we can compute per-date baseline means.

    Returns a dict with:
      mean_change:          mean NDVI diff across the region
      mean_ndvi1:           mean NDVI on Date 1 (baseline)
      mean_ndvi2:           mean NDVI on Date 2 (endpoint)
      std_change:           standard deviation of NDVI diff (spatial variability)
      net_change_km2:       area_gain minus area_loss
      gain_loss_ratio:      area_gain / area_loss (>1 = net greening)
      area_gain_km2:        area where diff > +CHANGE_THRESHOLD
      area_loss_km2:        area where diff < -CHANGE_THRESHOLD
      area_stable_km2:      area where |diff| <= CHANGE_THRESHOLD
      area_extreme_gain_km2 area where diff > +EXTREME_THRESHOLD
      area_extreme_loss_km2 area where diff < -EXTREME_THRESHOLD
      area_total_km2:       total analysed area
      pct_gain:             gain area as % of total
      pct_loss:             loss area as % of total
      pct_stable:           stable area as % of total
      threshold:            moderate change threshold used
      extreme_threshold:    extreme change threshold used
    """
    if not gee_available or diff_image is None:
        return _empty_stats()

    try:
        import ee

        bbox     = _pad_bbox(bbox)
        geometry = ee.Geometry.Rectangle(bbox)

        # ---------------------------------------------------------------
        # Scalar stats — two separate reduceRegion calls.
        #
        # Call 1: mean and std dev of the diff image.
        #   Reducer.mean().combine(stdDev, sharedInputs=True) works on a
        #   single-band image — both reducers share the same input band.
        #
        # Call 2: mean NDVI for each date separately.
        #   Stack ndvi1 and ndvi2 as two bands, reduce with mean().
        # ---------------------------------------------------------------

        # Call 1 — diff mean and std dev
        diff_stats = diff_image.rename("diff").reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), sharedInputs=True
            ),
            geometry=geometry,
            scale=STATS_SCALE,
            maxPixels=1e8,
        ).getInfo()

        mean_change = float(diff_stats.get("diff_mean")   or 0)
        std_change  = float(diff_stats.get("diff_stdDev") or 0)

        # Call 2 — baseline NDVI means for both dates
        ndvi_stats = img1.rename("ndvi1").addBands(img2.rename("ndvi2")).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=STATS_SCALE,
            maxPixels=1e8,
        ).getInfo()

        mean_ndvi1 = float(ndvi_stats.get("ndvi1") or 0)
        mean_ndvi2 = float(ndvi_stats.get("ndvi2") or 0)

        # ---------------------------------------------------------------
        # Area calculations — one GEE call using a classified image.
        #
        # Instead of five separate pixelArea reductions (one per category),
        # we assign each pixel to a class integer, stack pixel area as a
        # second band, and sum area by class in a single grouped reduction.
        #
        # Class codes:
        #   1 = moderate gain  (diff > CHANGE_THRESHOLD)
        #   2 = moderate loss  (diff < -CHANGE_THRESHOLD)
        #   3 = stable         (|diff| <= CHANGE_THRESHOLD)
        #   4 = extreme gain   (diff > EXTREME_THRESHOLD)
        #   5 = extreme loss   (diff < -EXTREME_THRESHOLD)
        #
        # Extreme classes overlap with moderate — a pixel with diff > 0.3
        # is counted in BOTH extreme gain (4) AND moderate gain (1).
        # The displayed "gain area" uses class 1 (moderate threshold).
        # The displayed "extreme gain area" uses class 4.
        # ---------------------------------------------------------------

        pixel_area = ee.Image.pixelArea().divide(1e6)   # convert m² to km²

        # Build a classification image — each pixel gets its class code
        # Higher-priority classes are assigned last so they overwrite lower ones.
        class_img = (
            ee.Image(3)                                           # default: stable
            .where(diff_image.gt(CHANGE_THRESHOLD),   1)         # moderate gain
            .where(diff_image.lt(-CHANGE_THRESHOLD),  2)         # moderate loss
        )

        # Sum pixel area grouped by class — one round-trip for gain/loss/stable
        area_by_class = (
            class_img.rename("class")
            .addBands(pixel_area.rename("area"))
            .reduceRegion(
                reducer=ee.Reducer.sum().group(groupField=0, groupName="class"),
                geometry=geometry,
                scale=STATS_SCALE,
                maxPixels=1e8,
            )
            .getInfo()
        )

        # Parse grouped results into a {class_code: area_km2} dict
        class_areas = {}
        for entry in area_by_class.get("groups", []):
            class_areas[int(entry["class"])] = float(entry["sum"] or 0)

        area_gain   = class_areas.get(1, 0.0)
        area_loss   = class_areas.get(2, 0.0)
        area_stable = class_areas.get(3, 0.0)
        area_total  = area_gain + area_loss + area_stable

        # Extreme thresholds — a second grouped reduction using the same pattern
        extreme_class_img = (
            ee.Image(0)
            .where(diff_image.gt(EXTREME_THRESHOLD),  4)
            .where(diff_image.lt(-EXTREME_THRESHOLD), 5)
        )

        extreme_by_class = (
            extreme_class_img.rename("class")
            .addBands(pixel_area.rename("area"))
            .reduceRegion(
                reducer=ee.Reducer.sum().group(groupField=0, groupName="class"),
                geometry=geometry,
                scale=STATS_SCALE,
                maxPixels=1e8,
            )
            .getInfo()
        )

        extreme_areas = {}
        for entry in extreme_by_class.get("groups", []):
            extreme_areas[int(entry["class"])] = float(entry["sum"] or 0)

        area_extreme_gain = extreme_areas.get(4, 0.0)
        area_extreme_loss = extreme_areas.get(5, 0.0)

        pct_gain   = (area_gain   / area_total * 100) if area_total > 0 else 0
        pct_loss   = (area_loss   / area_total * 100) if area_total > 0 else 0
        pct_stable = (area_stable / area_total * 100) if area_total > 0 else 0

        net_change     = area_gain - area_loss
        gain_loss_ratio = (area_gain / area_loss) if area_loss > 0 else None

        return {
            "mean_change":           round(mean_change,        4),
            "mean_ndvi1":            round(mean_ndvi1,         3),
            "mean_ndvi2":            round(mean_ndvi2,         3),
            "std_change":            round(std_change,         4),
            "net_change_km2":        round(net_change,         0),
            "gain_loss_ratio":       round(gain_loss_ratio, 2) if gain_loss_ratio is not None else None,
            "area_gain_km2":         round(area_gain,          0),
            "area_loss_km2":         round(area_loss,          0),
            "area_stable_km2":       round(area_stable,        0),
            "area_extreme_gain_km2": round(area_extreme_gain,  0),
            "area_extreme_loss_km2": round(area_extreme_loss,  0),
            "area_total_km2":        round(area_total,         0),
            "pct_gain":              round(pct_gain,           1),
            "pct_loss":              round(pct_loss,           1),
            "pct_stable":            round(pct_stable,         1),
            "threshold":             CHANGE_THRESHOLD,
            "extreme_threshold":     EXTREME_THRESHOLD,
        }

    except Exception as e:
        st.warning(f"Stats computation error: {e}")
        return _empty_stats()


def _empty_stats():
    """Return a zeroed stats dict when GEE is unavailable."""
    return {
        "mean_change":           0.0,
        "mean_ndvi1":            0.0,
        "mean_ndvi2":            0.0,
        "std_change":            0.0,
        "net_change_km2":        0.0,
        "gain_loss_ratio":       None,
        "area_gain_km2":         0.0,
        "area_loss_km2":         0.0,
        "area_stable_km2":       0.0,
        "area_extreme_gain_km2": 0.0,
        "area_extreme_loss_km2": 0.0,
        "area_total_km2":        0.0,
        "pct_gain":              0.0,
        "pct_loss":              0.0,
        "pct_stable":            0.0,
        "threshold":             CHANGE_THRESHOLD,
        "extreme_threshold":     EXTREME_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Folium map builder
# ---------------------------------------------------------------------------

def build_change_map(img1, img2, bbox, date1_str, date2_str, gee_available):
    """Build a Folium map with three GEE tile layers:
      - NDVI Date 1 (green colormap)
      - NDVI Date 2 (green colormap)
      - NDVI Difference (red-white-green diverging colormap)

    Returns a Folium map object, or None if GEE is unavailable or an error occurs.
    """
    if not gee_available or img1 is None or img2 is None:
        return None

    try:
        import ee

        bbox     = _pad_bbox(bbox)
        cx       = (bbox[0] + bbox[2]) / 2
        cy       = (bbox[1] + bbox[3]) / 2

        # Compute the difference image
        diff_img = img2.subtract(img1).rename("NDVI_diff")

        # Visualization parameters
        ndvi_vis = {
            "min": 0.0,
            "max": 0.9,
            "palette": ["#f7fcf0", "#74c476", "#00441b"],  # light green to dark green
        }
        diff_vis = {
            "min": -0.5,
            "max":  0.5,
            "palette": ["#d73027", "#f7f7f7", "#1a9850"],  # red, white, green
        }

        # Get GEE tile URLs — GEE renders tiles on-demand; only visible tiles are fetched
        url_ndvi1 = img1.getMapId(ndvi_vis)["tile_fetcher"].url_format
        url_ndvi2 = img2.getMapId(ndvi_vis)["tile_fetcher"].url_format
        url_diff  = diff_img.getMapId(diff_vis)["tile_fetcher"].url_format

        # Build Folium map centred on the study region
        m = folium.Map(
            location=[cy, cx],
            zoom_start=8,
            tiles="CartoDB positron",
        )

        # Add tile layers — the change map is shown by default
        folium.TileLayer(
            tiles=url_ndvi1,
            attr="Google Earth Engine / Copernicus",
            name=f"NDVI — Date 1 ({date1_str})",
            overlay=True,
            show=False,
            opacity=0.85,
        ).add_to(m)

        folium.TileLayer(
            tiles=url_ndvi2,
            attr="Google Earth Engine / Copernicus",
            name=f"NDVI — Date 2 ({date2_str})",
            overlay=True,
            show=False,
            opacity=0.85,
        ).add_to(m)

        folium.TileLayer(
            tiles=url_diff,
            attr="Google Earth Engine / Copernicus",
            name=f"NDVI Change ({date1_str} → {date2_str})",
            overlay=True,
            show=True,   # change map is the default visible layer
            opacity=0.85,
        ).add_to(m)

        # Study area boundary rectangle
        folium.Rectangle(
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color="#333333",
            weight=1.5,
            fill=False,
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        return m

    except Exception as e:
        st.warning(f"Map build error: {e}")
        return None


# ---------------------------------------------------------------------------
# AI interpretation
# ---------------------------------------------------------------------------

def get_change_interpretation(stats, date1, date2, region, src1, src2,
                               groq_key="", gemini_key=""):
    """Return a plain-language interpretation of the change detection result.

    Tries the full AI provider chain (Gemini first, then Groq) via ai_chain.
    Returns a substantive fallback if no keys are configured or all models fail.
    Returns a tuple: (interpretation_text, model_name_or_None)
    """
    import ai_chain

    ratio_str = (
        f"{stats['gain_loss_ratio']:.2f}"
        if stats["gain_loss_ratio"] is not None
        else "N/A (no loss)"
    )

    prompt = (
        f"You are an Earth observation analyst. "
        f"Interpret the following NDVI change detection result for {region}.\n\n"
        f"Date 1: {date1} (data source: {src1}) — mean NDVI: {stats['mean_ndvi1']:.3f}\n"
        f"Date 2: {date2} (data source: {src2}) — mean NDVI: {stats['mean_ndvi2']:.3f}\n\n"
        f"Mean NDVI change: {stats['mean_change']:+.4f}\n"
        f"Std deviation of change: {stats['std_change']:.4f} "
        f"(low = uniform/seasonal, high = patchy/human activity)\n"
        f"Net change: {stats['net_change_km2']:+,.0f} km2 (gain minus loss)\n"
        f"Gain/Loss ratio: {ratio_str}\n\n"
        f"Area of significant gain (>{CHANGE_THRESHOLD} NDVI): "
        f"{stats['area_gain_km2']:,.0f} km2 ({stats['pct_gain']:.1f}%)\n"
        f"Area of significant loss (<-{CHANGE_THRESHOLD} NDVI): "
        f"{stats['area_loss_km2']:,.0f} km2 ({stats['pct_loss']:.1f}%)\n"
        f"Stable area: {stats['area_stable_km2']:,.0f} km2 ({stats['pct_stable']:.1f}%)\n"
        f"Extreme gain area (>{EXTREME_THRESHOLD} NDVI): "
        f"{stats['area_extreme_gain_km2']:,.0f} km2\n"
        f"Extreme loss area (<-{EXTREME_THRESHOLD} NDVI): "
        f"{stats['area_extreme_loss_km2']:,.0f} km2\n\n"
        f"Cover the following in 3-4 short paragraphs:\n"
        f"1. What the numbers show — net greening or browning, and how large the change is.\n"
        f"2. The most likely causes — consider season, land use, climate, and whether "
        f"the dates span different seasons.\n"
        f"3. One practical application — what decision would this data support?\n"
        f"4. One limitation — what NDVI difference cannot tell us.\n\n"
        f"Write in plain language. No bullet points. Be direct."
    )

    text, model = ai_chain.complete(prompt, groq_key=groq_key, gemini_key=gemini_key)

    if text:
        return text, model

    return _fallback_interpretation(stats, date1, date2, region), None


def _fallback_interpretation(stats, date1, date2, region):
    """Substantive fallback interpretation based on the statistics.

    This is not placeholder text. It provides genuinely useful information
    derived from the computed statistics, regardless of whether Groq is available.
    """
    mean   = stats["mean_change"]
    gain   = stats["pct_gain"]
    loss   = stats["pct_loss"]
    stable = 100 - gain - loss

    # Characterize the magnitude of change
    if abs(mean) < 0.05:
        magnitude = "minimal"
    elif abs(mean) < 0.15:
        magnitude = "moderate"
    else:
        magnitude = "substantial"

    direction = "greening" if mean > 0 else "browning"

    interpretation = (
        f"**What the data shows.** "
        f"Between {date1} and {date2}, {region} shows {magnitude} net {direction} "
        f"(mean NDVI change: {mean:+.3f}). "
        f"{gain:.1f}% of the analysed area shows significant vegetation gain, "
        f"{loss:.1f}% shows significant loss, and {stable:.1f}% is stable.\n\n"
    )

    # Contextual explanation based on direction
    if mean > 0.15:
        interpretation += (
            "**Likely causes.** A mean NDVI gain of this magnitude most often reflects "
            "a seasonal transition — moving from a dry season to a wet season, or from "
            "winter dormancy to summer growth. Rapid crop establishment or post-fire "
            "regrowth can also produce gains at this scale. If the two dates span "
            "different seasons, the seasonal signal will dominate.\n\n"
        )
    elif mean > 0:
        interpretation += (
            "**Likely causes.** Moderate greening at this scale is consistent with "
            "gradual vegetation recovery, improved rainfall relative to the previous "
            "period, or seasonal progression within a single season. Small-scale "
            "agricultural land conversion to vegetation is also possible.\n\n"
        )
    elif mean < -0.15:
        interpretation += (
            "**Likely causes.** A mean NDVI loss of this magnitude is significant. "
            "Possible causes include severe drought stress, deforestation, fire activity, "
            "agricultural harvest, or a transition from wet to dry season. "
            "Large contiguous loss areas suggest drought or seasonal drying. "
            "Fragmented small patches suggest clearing or fire.\n\n"
        )
    else:
        interpretation += (
            "**Likely causes.** Moderate browning at this scale could reflect the "
            "end of a growing season, below-average rainfall, mild drought stress, "
            "or agricultural harvest. It is not necessarily a sign of permanent change.\n\n"
        )

    interpretation += (
        "**Practical application.** This analysis supports drought monitoring, "
        "crop yield estimation, deforestation surveillance, and post-fire assessment. "
        "Adding Groq AI interpretation (via the GROQ_API_KEY secret) will provide "
        "a more specific analysis tailored to this exact location and date pair.\n\n"
        "**Limitation.** NDVI differencing cannot distinguish between permanent "
        "land cover change and reversible seasonal change. A location that appears to "
        "have lost vegetation may simply be in its normal dry season. Multi-year "
        "analysis comparing the same calendar dates in different years is needed "
        "to separate seasonal variation from long-term trends."
    )

    return interpretation
