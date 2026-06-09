"""
land_cover.py — Land Cover Classification module for the EOIL portal.

Arc 1: Optical Intelligence and Land Cover.

Three-layer architecture:
  Layer 1: Fetch five Sentinel-2 bands from Planetary Computer
  Layer 2: K-means clustering (unsupervised) + Random Forest (weakly supervised)
  Layer 3: Groq/Gemini interprets classification results

Results are cached in Streamlit session state so switching between tabs
does not re-fetch or re-compute.
"""

import requests
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pystac_client
import planetary_computer

from PIL import Image
from io import BytesIO

import config
import geocoder
import map_picker
import ai_chain

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PC_RENDER_URL = "https://planetarycomputer.microsoft.com/api/data/v1/item/preview"
PC_STAC_URL   = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _pc_search_with_retry(search, retries=3, delay=5):
    """
    Call list(search.items()) with up to `retries` attempts.
    Planetary Computer occasionally returns a timeout on the free tier.
    Waiting `delay` seconds between attempts clears most transient errors.
    Raises the last exception if all attempts fail.
    """
    import time
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return list(search.items())
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay)
    raise last_exc

BANDS = ["B02", "B03", "B04", "B08", "B11"]

# Land cover color scheme — shared by K-means and Random Forest displays
LAND_COVER_COLORS = {
    "Water":               "#2166ac",
    "Dense vegetation":    "#1a9850",
    "Crops / sparse veg":  "#a6d96a",
    "Urban / built-up":    "#d7191c",
    "Desert / bare soil":  "#d6a85a",
}

LABEL_TO_INT = {
    "Water": 1, "Dense vegetation": 2, "Crops / sparse veg": 3,
    "Urban / built-up": 4, "Desert / bare soil": 5,
}

IMG_WIDTH = 256   # pixel width for all fetched images

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _compute_img_height(bbox):
    """Compute image height in pixels preserving the geographic aspect ratio."""
    lon_span = bbox[2] - bbox[0]
    lat_span = bbox[3] - bbox[1]
    lat_mid  = (bbox[1] + bbox[3]) / 2.0
    lon_km   = lon_span * 111.0 * np.cos(np.radians(lat_mid))
    lat_km   = lat_span * 111.0
    ratio    = lat_km / lon_km if lon_km > 0 else 1.0
    return max(64, int(IMG_WIDTH * ratio))


def _fetch_band(item, band_name, bbox, width, height):
    """Fetch one Sentinel-2 band as a 2D float array (values 0-1).

    Uses the PC rendering API with geographic bbox so only the study area
    is returned, not the full 100x100 km satellite tile.
    """
    params = [
        ("collection", "sentinel-2-l2a"),
        ("item",       item.id),
        ("assets",     band_name),
        ("asset_bidx", f"{band_name}|1"),
        ("rescale",    "0,10000"),
        ("width",      str(width)),
        ("height",     str(height)),
        ("bbox",       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"),
    ]
    try:
        resp = requests.get(PC_RENDER_URL, params=params, timeout=60)
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content)).convert("L")
            return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        pass
    return None


def _fetch_rgb(item, bbox, width, height):
    """Fetch true-color RGB composite exactly as the notebook does.

    Uses the PC multi-band rendering API with rescale=0,3000 and Gamma RGB 2.2.
    This is a single API call that returns a ready-made RGB PNG — the server
    composites the three bands and applies gamma in one step.

    rescale=0,3000 (not 0,10000) is intentional: it exposes the low-reflectance
    range that land surfaces occupy, making the image visually bright and correct.
    """
    params = [
        ("collection",    "sentinel-2-l2a"),
        ("item",          item.id),
        ("assets",        "B04"), ("assets", "B03"), ("assets", "B02"),
        ("asset_bidx",    "B04|1"), ("asset_bidx", "B03|1"), ("asset_bidx", "B02|1"),
        ("rescale",       "0,3000"), ("rescale", "0,3000"), ("rescale", "0,3000"),
        ("color_formula", "Gamma RGB 2.2"),
        ("width",         str(width)),
        ("height",        str(height)),
        ("bbox",          f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"),
    ]
    try:
        resp = requests.get(PC_RENDER_URL, params=params, timeout=60)
        if resp.status_code == 200:
            return np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
    except Exception:
        pass
    return None


def _build_rgb_from_bands(band_arrays, valid_mask=None):
    """Build true-color RGB from already-fetched B04/B03/B02 band arrays.

    valid_mask: boolean array (H, W). True = real satellite data. False = NoData
    (pixel is outside the satellite tile footprint — all bands returned 0).
    NoData pixels are rendered as light grey so they are clearly non-data.

    Uses 2nd-98th percentile stretching on valid pixels only, so the image
    is bright and readable regardless of scene brightness.
    """
    r = band_arrays.get("B04")
    g = band_arrays.get("B03")
    b = band_arrays.get("B02")
    if r is None or g is None or b is None:
        return None

    if valid_mask is None:
        valid_mask = (r > 0) | (g > 0) | (b > 0)

    def _stretch(band):
        """Stretch band to 0-1 using 2nd-98th percentile of valid pixels only."""
        vals = band[valid_mask]
        if vals.size == 0:
            return np.zeros_like(band, dtype=np.float32)
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        if hi <= lo:
            hi = lo + 1e-6
        return np.clip((band - lo) / (hi - lo), 0, 1)

    rgb = np.stack([_stretch(r), _stretch(g), _stretch(b)], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    # NoData pixels → light grey (not black, which looks like a render failure)
    rgb[~valid_mask] = [210, 210, 210]
    return rgb


def search_scenes(bbox, date_range, max_cloud=10):
    """Search Planetary Computer and return a list of available scenes.

    Returns a list of dicts — one per scene — sorted by date descending.
    Each dict has: item_id, date, cloud_cover, label (for the selectbox).
    Does not fetch any band data — that happens only after the user picks a scene.
    """
    try:
        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        items = _pc_search_with_retry(search)
        if not items:
            return [], "No scenes found. Try widening the date range or increasing max cloud %."

        items.sort(key=lambda x: x.datetime, reverse=True)
        scenes = []
        for item in items:
            cloud = item.properties.get("eo:cloud_cover", 0)
            date  = item.datetime.strftime("%Y-%m-%d")
            scenes.append({
                "item_id":     item.id,
                "date":        date,
                "cloud_cover": round(cloud, 1),
                "label":       f"{date}  —  cloud {cloud:.1f}%",
            })
        return scenes, None

    except Exception as e:
        return [], str(e)


def fetch_scene(bbox, item_id, max_cloud=10):
    """Fetch five Sentinel-2 bands for a specific scene (item_id).

    item_id comes from search_scenes() — the user chose it from the selectbox.
    Returns a dict with keys: rgb, bands, ndvi, ndwi, ndbi, valid_mask,
    scene_date, scene_cloud, item_id, shape.
    """
    try:
        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace
        )
        # Fetch the specific item by ID
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            ids=[item_id],
        )
        items = _pc_search_with_retry(search)
        if not items:
            return None, f"Could not retrieve scene {item_id}."

        item = items[0]
        h    = _compute_img_height(bbox)

        # Fetch five bands for ML algorithms
        band_arrays = {}
        for band in BANDS:
            arr = _fetch_band(item, band, bbox, IMG_WIDTH, h)
            if arr is not None:
                band_arrays[band] = arr

        if len(band_arrays) < 5:
            return None, f"Only {len(band_arrays)} of 5 bands returned. Try a different date range."

        # Valid pixel mask — pixels where every band is zero are outside the
        # satellite tile footprint (NoData). They must be excluded from classification
        # and shown as grey in all output images.
        b02 = band_arrays["B02"]
        valid_mask = (
            (band_arrays["B02"] > 0) |
            (band_arrays["B03"] > 0) |
            (band_arrays["B04"] > 0) |
            (band_arrays["B08"] > 0) |
            (band_arrays["B11"] > 0)
        )

        # True-color RGB — built from the same band arrays used by the ML algorithms.
        rgb = _build_rgb_from_bands(band_arrays, valid_mask)

        # Spectral indices — only meaningful on valid pixels; set NoData to 0
        def safe_index(a, b):
            denom = a + b
            denom = np.where(denom == 0, 1e-6, denom)
            result = (a - b) / denom
            result[~valid_mask] = 0.0
            return result

        ndvi = safe_index(band_arrays["B08"], band_arrays["B04"])
        ndwi = safe_index(band_arrays["B03"], band_arrays["B08"])
        ndbi = safe_index(band_arrays["B11"], band_arrays["B08"])

        return {
            "rgb":         rgb,
            "bands":       band_arrays,
            "ndvi":        ndvi,
            "ndwi":        ndwi,
            "ndbi":        ndbi,
            "valid_mask":  valid_mask,
            "scene_date":  item.datetime.strftime("%Y-%m-%d"),
            "scene_cloud": item.properties.get("eo:cloud_cover", 0),
            "item_id":     item.id,
            "shape":       (band_arrays["B02"].shape[0], band_arrays["B02"].shape[1]),
        }, None

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# ML Layer 2: K-means
# ---------------------------------------------------------------------------

def run_kmeans(scene, n_clusters=5):
    """Run K-means clustering on the 8-feature stack and return results dict.

    Features: 5 bands + NDVI + NDWI + NDBI = 8 features per pixel.
    Only valid pixels (inside the satellite tile footprint) are clustered.
    NoData pixels are assigned cluster -1 and shown as grey in the output.
    """
    h, w  = scene["shape"]
    bands = scene["bands"]
    valid = scene.get("valid_mask", np.ones((h, w), dtype=bool))

    feature_stack = np.stack([
        bands["B02"], bands["B03"], bands["B04"],
        bands["B08"], bands["B11"],
        scene["ndvi"], scene["ndwi"], scene["ndbi"],
    ], axis=-1)

    X        = feature_stack.reshape(h * w, 8)
    valid_flat = valid.flatten()

    # Fit and predict only on valid pixels
    scaler   = StandardScaler()
    X_valid  = scaler.fit_transform(X[valid_flat])
    kmeans   = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels_valid = kmeans.fit_predict(X_valid)

    # Build full cluster map: -1 = NoData, 0..n_clusters-1 = cluster id
    cluster_labels = np.full(h * w, -1, dtype=int)
    cluster_labels[valid_flat] = labels_valid
    cluster_map = cluster_labels.reshape(h, w)

    ndvi_flat = scene["ndvi"].flatten()
    ndwi_flat = scene["ndwi"].flatten()
    ndbi_flat = scene["ndbi"].flatten()

    cluster_stats = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_stats.append({
            "cluster": c,
            "count":   int(mask.sum()),
            "pct":     float(100.0 * mask.sum() / (h * w)),
            "ndvi":    float(ndvi_flat[mask].mean()),
            "ndwi":    float(ndwi_flat[mask].mean()),
            "ndbi":    float(ndbi_flat[mask].mean()),
        })

    # Assign land cover labels based on cluster spectral signature.
    # Urban threshold is NDBI > 0.15 — matches the Random Forest pseudo-label rule.
    # NDBI > 0.05 is too loose: arid sandy soil also triggers it, causing the whole
    # Nile Delta desert to be mis-labelled Urban. 0.15 requires a genuinely high
    # built-surface signal (concrete, asphalt) to qualify.
    label_map = {}
    for stats in cluster_stats:
        if stats["ndwi"] > 0.05:
            label_map[stats["cluster"]] = "Water"
        elif stats["ndvi"] > 0.4:
            label_map[stats["cluster"]] = "Dense vegetation"
        elif stats["ndvi"] > 0.15:
            label_map[stats["cluster"]] = "Crops / sparse veg"
        elif stats["ndbi"] > 0.25 and stats["ndvi"] < 0.10 and stats["ndwi"] < -0.20:
            label_map[stats["cluster"]] = "Urban / built-up"
        else:
            label_map[stats["cluster"]] = "Desert / bare soil"

    return {
        "cluster_map":   cluster_map,
        "cluster_stats": cluster_stats,
        "label_map":     label_map,
        "n_clusters":    n_clusters,
    }


def kmeans_to_color_image(kmeans_result):
    """Convert cluster map to an RGB uint8 image using LAND_COVER_COLORS.
    Pixels with cluster id -1 are NoData — rendered as light grey.
    """
    cluster_map = kmeans_result["cluster_map"]
    label_map   = kmeans_result["label_map"]
    h, w        = cluster_map.shape

    img = np.full((h, w, 3), 210, dtype=np.uint8)  # default = grey (NoData)
    for c, label in label_map.items():
        color = LAND_COVER_COLORS.get(label, "#888888")
        rgb   = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        img[cluster_map == c] = rgb

    return img


# ---------------------------------------------------------------------------
# ML Layer 2: Random Forest
# ---------------------------------------------------------------------------

def run_random_forest(scene, n_estimators=100):
    """Train Random Forest with pseudo-labels and predict valid pixels only.

    NoData pixels (outside the satellite tile footprint) are excluded from
    training and prediction. They are assigned label 0 in the output map
    and rendered as grey.
    """
    h, w      = scene["shape"]
    bands     = scene["bands"]
    valid     = scene.get("valid_mask", np.ones((h, w), dtype=bool))
    valid_flat = valid.flatten()
    n_pixels  = h * w

    ndvi_flat = scene["ndvi"].flatten()
    ndwi_flat = scene["ndwi"].flatten()
    ndbi_flat = scene["ndbi"].flatten()

    feature_stack = np.stack([
        bands["B02"], bands["B03"], bands["B04"],
        bands["B08"], bands["B11"],
        scene["ndvi"], scene["ndwi"], scene["ndbi"],
    ], axis=-1)

    X      = feature_stack.reshape(n_pixels, 8)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate pseudo-labels on valid pixels only
    pseudo = np.zeros(n_pixels, dtype=int)
    v = valid_flat  # shorthand
    pseudo[v & (ndwi_flat > 0.15) & (ndvi_flat < 0.05)]                       = 1  # Water
    pseudo[v & (ndvi_flat > 0.50)]                                              = 2  # Dense veg
    pseudo[v & (ndvi_flat > 0.20) & (ndvi_flat <= 0.50) & (ndwi_flat < 0.1)]  = 3  # Crops
    pseudo[v & (ndbi_flat > 0.25) & (ndvi_flat < 0.10) & (ndwi_flat < -0.20)] = 4  # Urban
    pseudo[v & (ndvi_flat < 0.08) & (ndwi_flat < -0.10) & (pseudo == 0)]      = 5  # Desert

    train_mask = pseudo > 0
    labeled    = int(train_mask.sum())

    if labeled < 10:
        return None, "Too few labeled pixels for training. Try a larger area."

    X_train = X_scaled[train_mask]
    y_train = pseudo[train_mask]

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=15,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Predict only on valid pixels — NoData pixels stay as 0 (unclassified)
    rf_labels = np.zeros(n_pixels, dtype=int)
    rf_labels[valid_flat] = rf.predict(X_scaled[valid_flat])
    rf_map = rf_labels.reshape(h, w)

    feature_names = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "NDBI"]
    importances   = rf.feature_importances_

    return {
        "rf_map":         rf_map,
        "importances":    importances,
        "feature_names":  feature_names,
        "labeled_pct":    float(100.0 * labeled / valid_flat.sum()),
        "class_counts": {
            "Water":             int((rf_labels == 1).sum()),
            "Dense vegetation":  int((rf_labels == 2).sum()),
            "Crops / sparse veg":int((rf_labels == 3).sum()),
            "Urban / built-up":  int((rf_labels == 4).sum()),
            "Desert / bare soil":int((rf_labels == 5).sum()),
        },
    }, None


def rf_to_color_image(rf_result):
    """Convert RF label map to an RGB uint8 image using LAND_COVER_COLORS.
    Pixels with label 0 are NoData — rendered as light grey.
    """
    rf_map = rf_result["rf_map"]
    h, w   = rf_map.shape

    id_to_label = {
        1: "Water", 2: "Dense vegetation", 3: "Crops / sparse veg",
        4: "Urban / built-up", 5: "Desert / bare soil",
    }
    img = np.full((h, w, 3), 210, dtype=np.uint8)  # default = grey (NoData)
    for label_id, label in id_to_label.items():
        color = LAND_COVER_COLORS.get(label, "#888888")
        rgb   = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        img[rf_map == label_id] = rgb

    return img


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_feature_importance_chart(rf_result):
    """Build a horizontal Plotly bar chart of Random Forest feature importances."""
    names       = rf_result["feature_names"]
    importances = rf_result["importances"]
    order       = np.argsort(importances)

    sorted_names = [names[i] for i in order]
    sorted_vals  = [importances[i] for i in order]

    colors = []
    for n in sorted_names:
        if n == "NDVI":
            colors.append("#1a9850")
        elif n == "NDWI":
            colors.append("#2166ac")
        elif n == "NDBI":
            colors.append("#d7191c")
        else:
            colors.append("#4d4d4d")

    fig = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in sorted_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature importance — what drove the classification",
        xaxis_title="Importance",
        height=320,
        margin=dict(l=10, r=60, t=40, b=30),
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="#ffffff",
    )
    return fig


def build_area_breakdown_chart(label_counts, title):
    """Build a horizontal bar chart showing area by land cover class."""
    total  = sum(label_counts.values())
    labels = list(label_counts.keys())
    pcts   = [100.0 * v / total for v in label_counts.values()]
    colors = [LAND_COVER_COLORS.get(l, "#888888") for l in labels]

    fig = go.Figure(go.Bar(
        x=pcts,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="% of area",
        height=280,
        margin=dict(l=10, r=60, t=40, b=30),
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="#ffffff",
    )
    return fig


# ---------------------------------------------------------------------------
# Legend HTML
# ---------------------------------------------------------------------------

def _legend_html():
    """Return an HTML legend for the land cover color scheme."""
    items = ""
    for label, color in LAND_COVER_COLORS.items():
        items += (
            f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color};border-radius:2px;margin-right:5px;"></span>'
            f'{label}</span>'
        )
    return f'<div style="font-size:0.82rem;padding:6px 0;">{items}</div>'


# ---------------------------------------------------------------------------
# AI Layer 3: interpretation
# ---------------------------------------------------------------------------

def get_ai_interpretation(scene, kmeans_result, rf_result):
    """Generate a structured interpretation from the classification results.

    Builds a prompt that includes:
    - K-means cluster breakdown with labels
    - Random Forest area breakdown
    - Top 3 features by importance
    Returns (text, model_name) tuple.
    """
    km_stats  = kmeans_result["cluster_stats"]
    km_labels = kmeans_result["label_map"]

    # Summarize K-means
    km_summary = []
    for stats in km_stats:
        label = km_labels[stats["cluster"]]
        km_summary.append(f"{label}: {stats['pct']:.1f}% of area")

    # Summarize RF area
    n_pixels   = rf_result["rf_map"].size
    rf_summary = [
        f"{label}: {100.0 * count / n_pixels:.1f}% of area"
        for label, count in rf_result["class_counts"].items()
        if count > 0
    ]

    # Top 3 features
    names    = rf_result["feature_names"]
    imp      = rf_result["importances"]
    top3_idx = np.argsort(imp)[::-1][:3]
    top3     = [f"{names[i]} ({imp[i]:.3f})" for i in top3_idx]

    prompt = f"""You are an Earth Observation analyst interpreting a land cover classification
of a Sentinel-2 satellite scene.

K-means classification (unsupervised, {kmeans_result['n_clusters']} clusters):
{chr(10).join(km_summary)}

Random Forest classification (weakly supervised):
{chr(10).join(rf_summary)}

Top 3 most important spectral features for the Random Forest:
{', '.join(top3)}

Scene date: {scene['scene_date']}
Cloud cover: {scene['scene_cloud']:.1f}%

Write a structured interpretation in 4 short paragraphs:
1. What the dominant land cover types are and what they indicate about this landscape.
2. What the top features tell us — why those spectral signals drove the classification.
3. Where K-means and Random Forest likely agree and where they may diverge.
4. One practical application of this classification (e.g. agriculture monitoring, urban expansion tracking, water resource mapping).

Be direct. No more than 200 words total. No filler phrases."""

    text, model = ai_chain.complete(
        prompt,
        groq_key=config.GROQ_API_KEY,
        gemini_key=config.GEMINI_API_KEY,
    )
    return text, model


def _fallback_interpretation(scene, kmeans_result, rf_result):
    """Return a substantive fallback interpretation when no AI key is available."""
    km_labels = kmeans_result["label_map"]
    label_counts_km = {}
    for stats in kmeans_result["cluster_stats"]:
        label = km_labels[stats["cluster"]]
        label_counts_km[label] = label_counts_km.get(label, 0) + stats["pct"]

    dominant = max(label_counts_km, key=label_counts_km.get)

    names = rf_result["feature_names"]
    imp   = rf_result["importances"]
    top1  = names[np.argmax(imp)]

    return f"""**Land cover classification — {scene['scene_date']}**

The dominant land cover class is **{dominant}** ({label_counts_km[dominant]:.1f}% of the scene by K-means). This reflects the spectral character of the study area.

The most important feature for the Random Forest classifier was **{top1}**. This indicates that the primary spectral contrast driving class separation is the {top1} signal.

K-means and Random Forest typically agree on pixels with strong spectral signatures (pure water, dense vegetation, bright desert) and diverge at class boundaries where pixels contain mixed land cover types.

To enable AI interpretation, add a GROQ_API_KEY or GEMINI_API_KEY to your .env file."""


# ---------------------------------------------------------------------------
# Main render function — called from app.py
# ---------------------------------------------------------------------------

def render(location_name, bbox):
    """Render the Land Cover Classification module.

    Called from app.py when the Land Cover module is selected.
    location_name: string label for display
    bbox: [west, south, east, north] bounding box
    """

    st.subheader("🌿 Land Cover Classification")
    st.caption(
        "Classify land cover using K-means clustering and Random Forest. "
        "Compare how unsupervised and supervised AI approaches see the same scene."
    )

    with st.expander("ℹ️ What does this module do?", expanded=False):
        st.markdown("""
**This module answers the question: what land cover types are present, and how confident are we in each classification?**

It fetches a real Sentinel-2 scene from Planetary Computer and runs two machine learning algorithms on the same imagery:

**K-means (unsupervised):** Groups pixels by spectral similarity. No training labels required. The algorithm finds structure in the data independently. You assign human-readable names to the clusters after the fact.

**Random Forest (weakly supervised):** A supervised classifier trained on automatically-generated labels derived from spectral thresholds. Learns a complex decision boundary from those labels and predicts every pixel.

**Three-layer architecture:**
- Layer 1: Five Sentinel-2 bands (B02, B03, B04, B08, B11) plus NDVI, NDWI, NDBI
- Layer 2: K-means and Random Forest applied to the same feature stack
- Layer 3: AI interpretation of what the classification reveals

**How to use it:**
1. Type a location and set a date range
2. Click Find Scenes — see all available Sentinel-2 scenes with date and cloud cover
3. Pick the scene you want from the list
4. Click Run Classification
5. Explore the K-means and Random Forest tabs
6. Click Get AI Interpretation for a structured analysis
        """)

    # --- Controls ---
    col_loc, col_d1, col_d2, col_cloud = st.columns([3, 1, 1, 1])

    with col_loc:
        lc_place = st.text_input(
            "Location",
            placeholder="e.g. Nile Delta, Egypt   |   Sacramento Valley   |   Mekong Delta",
            key="lc_place",
            value=location_name if location_name else "",
        )

    with col_d1:
        from datetime import date as _date, timedelta as _td
        lc_date_start = st.date_input(
            "From date",
            value=_date(2023, 3, 1),
            key="lc_date_start",
        )

    with col_d2:
        lc_date_end = st.date_input(
            "To date",
            value=_date(2023, 5, 31),
            key="lc_date_end",
        )

    with col_cloud:
        lc_max_cloud = st.slider("Max cloud %", 0, 30, 10, key="lc_cloud")

    col_k, col_find, col_run = st.columns([1, 2, 2])
    with col_k:
        n_clusters = st.slider("K-means clusters", 3, 8, 5, key="lc_k")
    with col_find:
        st.write("")
        find_btn = st.button(
            "🔍 Find Available Scenes",
            use_container_width=True,
            key="lc_find",
        )
    with col_run:
        st.write("")
        run_btn = st.button(
            "▶ Run Classification",
            type="primary",
            use_container_width=True,
            key="lc_run",
        )

    # --- Geocode ---
    lc_bbox        = bbox
    lc_region_name = location_name

    if lc_place.strip():
        cached_place = st.session_state.get("lc_geocoded_place", "")
        cached_bbox  = st.session_state.get("lc_geocoded_bbox", None)

        if lc_place.strip() != cached_place:
            map_picker.clear_click("lc")
            with st.spinner(f"Looking up '{lc_place}'..."):
                result_bbox = geocoder.geocode_place(lc_place)
            if result_bbox:
                st.session_state.lc_geocoded_place = lc_place.strip()
                st.session_state.lc_geocoded_bbox  = result_bbox
                cached_bbox = result_bbox
            else:
                st.error(f"Could not find '{lc_place}'. Try a different name.")
                cached_bbox = None

        if cached_bbox:
            lc_bbox        = cached_bbox
            lc_region_name = lc_place.strip()

    # Map picker
    if lc_bbox:
        with st.expander("📍 Refine location — click map to set exact area", expanded=False):
            picked = map_picker.render_map_picker(
                centre_bbox=lc_bbox,
                picker_key="lc",
                default_size_km=100,
            )
            if picked:
                lc_bbox = picked

    # Show bbox size so the user knows what area will be analysed
    if lc_bbox:
        lon_span = lc_bbox[2] - lc_bbox[0]
        lat_span = lc_bbox[3] - lc_bbox[1]
        import math
        mid_lat  = (lc_bbox[1] + lc_bbox[3]) / 2
        width_km = round(lon_span * 111.0 * math.cos(math.radians(mid_lat)))
        height_km= round(lat_span * 111.0)
        if width_km > 150 or height_km > 150:
            st.warning(
                f"⚠️ The geocoded area is **{width_km} × {height_km} km** — too large for a single "
                f"Sentinel-2 tile. This will produce black or grey NoData areas in the output. "
                f"Open **Refine location** above, click the map to centre a 100 km box on the "
                f"agricultural area you want to classify, then run again."
            )
        else:
            st.caption(f"📍 Analysis area: {width_km} × {height_km} km")

    # --- Step 1: Find available scenes ---
    if find_btn:
        if not lc_bbox:
            st.warning("Enter a location first.")
        else:
            date_range = f"{lc_date_start.isoformat()}/{lc_date_end.isoformat()}"
            with st.spinner(f"Searching for Sentinel-2 scenes over {lc_region_name}..."):
                scenes, err = search_scenes(lc_bbox, date_range, lc_max_cloud)
            if err:
                st.error(f"Search failed: {err}")
            elif not scenes:
                st.warning("No scenes found. Try a wider date range or higher max cloud %.")
            else:
                st.session_state.lc_available_scenes = scenes
                st.session_state.lc_search_bbox      = lc_bbox
                st.session_state.lc_search_region    = lc_region_name

    # --- Scene selector — shown after search ---
    selected_item_id = None
    if st.session_state.get("lc_available_scenes"):
        scenes = st.session_state.lc_available_scenes
        st.success(f"Found {len(scenes)} scene{'s' if len(scenes) != 1 else ''} — pick one below.")
        labels     = [s["label"] for s in scenes]
        chosen_idx = st.selectbox(
            "Select scene",
            range(len(labels)),
            format_func=lambda i: labels[i],
            key="lc_scene_selector",
        )
        selected_item_id = scenes[chosen_idx]["item_id"]

    # --- Step 2: Run classification on chosen scene ---
    if run_btn:
        if not lc_bbox:
            st.warning("Enter a location first.")
        elif not selected_item_id:
            st.warning("Click **Find Available Scenes** first, then pick a scene from the list.")
        else:
            st.session_state.lc_scene    = None
            st.session_state.lc_kmeans   = None
            st.session_state.lc_rf       = None
            st.session_state.lc_ai       = None
            st.session_state.lc_pending  = {
                "bbox":       st.session_state.get("lc_search_bbox", lc_bbox),
                "item_id":    selected_item_id,
                "max_cloud":  lc_max_cloud,
                "n_clusters": n_clusters,
                "region":     st.session_state.get("lc_search_region", lc_region_name),
            }
            st.rerun()

    # --- Execute pending run ---
    if st.session_state.get("lc_pending"):
        p = st.session_state.lc_pending
        st.session_state.lc_pending = None

        with st.spinner(f"Fetching Sentinel-2 bands for {p['region']} (~15 seconds)..."):
            scene, err = fetch_scene(p["bbox"], p["item_id"], p["max_cloud"])

        if err:
            st.error(f"Data fetch failed: {err}")
        else:
            with st.spinner("Running K-means clustering..."):
                km_result = run_kmeans(scene, p["n_clusters"])

            with st.spinner("Training Random Forest..."):
                rf_result, rf_err = run_random_forest(scene)

            if rf_err:
                st.warning(f"Random Forest: {rf_err}")
            else:
                st.session_state.lc_scene  = scene
                st.session_state.lc_kmeans = km_result
                st.session_state.lc_rf     = rf_result
                st.session_state.lc_region = p["region"]
                st.success(
                    f"Classification complete — scene from {scene['scene_date']}, "
                    f"cloud cover {scene['scene_cloud']:.1f}%."
                )

    # --- Display results ---
    if (st.session_state.get("lc_scene") is not None and
            st.session_state.get("lc_kmeans") is not None and
            st.session_state.get("lc_rf") is not None):

        scene      = st.session_state.lc_scene
        km_result  = st.session_state.lc_kmeans
        rf_result  = st.session_state.lc_rf
        region     = st.session_state.get("lc_region", "")

        km_img = kmeans_to_color_image(km_result)
        rf_img = rf_to_color_image(rf_result)

        st.divider()

        # --- Tabs ---
        tab1, tab2, tab3 = st.tabs(["🔵 K-means", "🌲 Random Forest", "🔍 Compare"])

        # --- Tab 1: K-means ---
        with tab1:
            st.markdown(f"**K-means Classification — {region} — {scene['scene_date']}**")
            st.markdown(
                "Unsupervised clustering. The algorithm grouped pixels by spectral similarity "
                "without any labels. Human-readable names were assigned after the fact based "
                "on each cluster's mean NDVI, NDWI, and NDBI values."
            )

            col_img, col_stats = st.columns([2, 1])

            with col_img:
                st.markdown(_legend_html(), unsafe_allow_html=True)
                st.image(km_img, caption="K-means classification", use_container_width=True)

            with col_stats:
                st.markdown("**Cluster breakdown**")
                km_labels = km_result["label_map"]
                for stats in sorted(km_result["cluster_stats"], key=lambda x: -x["pct"]):
                    label = km_labels[stats["cluster"]]
                    color = LAND_COVER_COLORS.get(label, "#888888")
                    st.markdown(
                        f'<span style="color:{color};font-weight:600;">{label}</span>: '
                        f'{stats["pct"]:.1f}%',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"NDVI {stats['ndvi']:+.3f}  "
                        f"NDWI {stats['ndwi']:+.3f}  "
                        f"NDBI {stats['ndbi']:+.3f}"
                    )

            # Area breakdown chart
            km_label_counts = {}
            for stats in km_result["cluster_stats"]:
                label = km_labels[stats["cluster"]]
                km_label_counts[label] = km_label_counts.get(label, 0.0) + stats["pct"]

            fig_km = build_area_breakdown_chart(km_label_counts, "K-means — area by class")
            st.plotly_chart(fig_km, use_container_width=True)

            st.caption(
                "**What is K-means?** The algorithm starts with random cluster centres, "
                "assigns every pixel to the nearest centre, then moves each centre to the "
                "mean of its pixels. It repeats until stable. The human assigns land cover "
                "labels based on each cluster's spectral profile — the machine never knew "
                "what 'vegetation' or 'water' means."
            )

        # --- Tab 2: Random Forest ---
        with tab2:
            st.markdown(f"**Random Forest Classification — {region} — {scene['scene_date']}**")
            st.markdown(
                f"Weakly supervised. Trained on {rf_result['labeled_pct']:.1f}% of pixels labeled "
                "by spectral thresholds. Predicted labels for all pixels including ambiguous ones."
            )

            col_img2, col_stats2 = st.columns([2, 1])

            with col_img2:
                st.markdown(_legend_html(), unsafe_allow_html=True)
                st.image(rf_img, caption="Random Forest classification", use_container_width=True)

            with col_stats2:
                st.markdown("**Class breakdown**")
                n_pixels = rf_result["rf_map"].size
                for label, count in rf_result["class_counts"].items():
                    if count > 0:
                        pct   = 100.0 * count / n_pixels
                        color = LAND_COVER_COLORS.get(label, "#888888")
                        st.markdown(
                            f'<span style="color:{color};font-weight:600;">{label}</span>: '
                            f'{pct:.1f}%',
                            unsafe_allow_html=True,
                        )

            # Feature importance chart
            fig_imp = build_feature_importance_chart(rf_result)
            st.plotly_chart(fig_imp, use_container_width=True)

            st.caption(
                "**Feature importance** shows which spectral signal drove the classification "
                "most. High importance = the feature was used early in many decision trees "
                "where the most information is gained. Low importance = the feature added "
                "little beyond what other features already captured."
            )

        # --- Tab 3: Compare ---
        with tab3:
            st.markdown(f"**Side-by-side comparison — {region} — {scene['scene_date']}**")

            col1, col2, col3 = st.columns(3)
            with col1:
                if scene.get("rgb") is not None:
                    st.image(scene["rgb"], caption="True color", use_container_width=True)
                else:
                    st.info("True color not available.")
            with col2:
                st.image(km_img, caption=f"K-means ({km_result['n_clusters']} clusters)", use_container_width=True)
            with col3:
                st.image(rf_img, caption="Random Forest", use_container_width=True)

            st.markdown(_legend_html(), unsafe_allow_html=True)

            # Agreement calculation
            km_int = np.zeros_like(km_result["cluster_map"], dtype=int)
            for c, label in km_result["label_map"].items():
                km_int[km_result["cluster_map"] == c] = LABEL_TO_INT.get(label, 0)

            agreement = int((km_int == rf_result["rf_map"]).sum())
            total     = km_int.size
            agree_pct = 100.0 * agreement / total

            st.metric(
                label="Agreement between K-means and Random Forest",
                value=f"{agree_pct:.1f}%",
                help="Percentage of pixels where both methods assigned the same land cover class.",
            )
            st.caption(
                f"Disagreement ({100 - agree_pct:.1f}%) occurs at class boundaries and in "
                "mixed pixels where the spectral signal does not clearly belong to one class."
            )

        # --- Layer 3: AI Interpretation ---
        st.divider()
        st.subheader("🤖 AI Interpretation")
        st.caption(
            "Layer 3: Groq or Gemini receives the classification statistics and explains "
            "what they mean. This layer interprets — it does not generate the analysis."
        )

        if st.button("Get AI Interpretation", type="primary", key="lc_ai_btn"):
            with st.spinner("Interpreting classification results..."):
                text, model = get_ai_interpretation(scene, km_result, rf_result)
                if text:
                    st.session_state.lc_ai       = text
                    st.session_state.lc_ai_model = model
                else:
                    st.session_state.lc_ai       = _fallback_interpretation(scene, km_result, rf_result)
                    st.session_state.lc_ai_model = "fallback"

        if st.session_state.get("lc_ai"):
            st.markdown(st.session_state.lc_ai)
            model = st.session_state.get("lc_ai_model", "")
            if model and model != "fallback":
                st.caption(f"AI response from {model}")

        # --- Export ---
        st.divider()
        st.subheader("📥 Export")

        import pandas as pd
        from PIL import Image as PILImage
        import io

        # Build classification stats table from RF results
        n_pixels = rf_result["rf_map"].size
        rows = []
        for label, count in rf_result["class_counts"].items():
            if count > 0:
                rows.append({
                    "Class":       label,
                    "Pixels":      count,
                    "Area_pct":    round(100.0 * count / n_pixels, 2),
                })
        export_df  = pd.DataFrame(rows).sort_values("Area_pct", ascending=False)
        csv_bytes  = export_df.to_csv(index=False).encode()
        safe_reg   = region.replace(" ", "_").replace(",", "")
        fname_csv  = f"land_cover_{safe_reg}_{scene['scene_date']}.csv"

        # Build PNG of the RF classified map
        rf_pil     = PILImage.fromarray(rf_img)
        png_buffer = io.BytesIO()
        rf_pil.save(png_buffer, format="PNG")
        png_bytes  = png_buffer.getvalue()
        fname_png  = f"land_cover_{safe_reg}_{scene['scene_date']}.png"

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="⬇️ Download classification stats as CSV",
                data=csv_bytes,
                file_name=fname_csv,
                mime="text/csv",
                key="lc_export_csv",
            )
            st.caption("Class name, pixel count, and area percentage from Random Forest.")
        with col_exp2:
            st.download_button(
                label="⬇️ Download classified map as PNG",
                data=png_bytes,
                file_name=fname_png,
                mime="image/png",
                key="lc_export_png",
            )
            st.caption("Random Forest classification map. Colour-coded by land cover class.")

    else:
        st.markdown("---")
        st.markdown(
            "**Type a location above, set a date range, and click Run Classification.**\n\n"
            "The module will fetch real Sentinel-2 imagery, run K-means and Random Forest "
            "on the same scene, and produce a structured AI interpretation."
        )
