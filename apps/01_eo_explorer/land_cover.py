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


def _build_rgb_from_bands(band_arrays):
    """Build a true-color uint8 RGB image from already-fetched B04/B03/B02 arrays.

    Uses the same bands the classification uses — no extra API call needed.
    Applies a simple gamma correction (power 0.5) to brighten the display.
    Sentinel-2 reflectance values at 0-1 float are typically low (0.05-0.3 for land)
    so the gamma lift makes the image readable without clipping.
    """
    r = band_arrays.get("B04")
    g = band_arrays.get("B03")
    b = band_arrays.get("B02")
    if r is None or g is None or b is None:
        return None

    # Stack to (H, W, 3), clip to 0-1, apply gamma, convert to uint8
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rgb = np.power(rgb, 0.5)          # gamma 0.5 — brightens without blowing out
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


def fetch_scene(bbox, date_range, max_cloud=10):
    """Search Planetary Computer and fetch five bands for the best scene.

    Returns a dict with keys: rgb, bands, ndvi, ndwi, ndbi, scene_date,
    scene_cloud, item_id. Returns None on failure.
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
        items = list(search.items())
        if not items:
            return None, "No scenes found for this location and date range."

        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        item  = items[0]
        h     = _compute_img_height(bbox)

        # Fetch five bands — RGB is built from these, no separate API call needed
        band_arrays = {}
        for band in BANDS:
            arr = _fetch_band(item, band, bbox, IMG_WIDTH, h)
            if arr is not None:
                band_arrays[band] = arr

        if len(band_arrays) < 5:
            return None, f"Only {len(band_arrays)} of 5 bands returned. Try a different date range."

        # Build true-color RGB from the fetched bands
        rgb = _build_rgb_from_bands(band_arrays)

        # Spectral indices
        def safe_index(a, b):
            denom = a + b
            denom = np.where(denom == 0, 1e-6, denom)
            return (a - b) / denom

        ndvi = safe_index(band_arrays["B08"], band_arrays["B04"])
        ndwi = safe_index(band_arrays["B03"], band_arrays["B08"])
        ndbi = safe_index(band_arrays["B11"], band_arrays["B08"])

        return {
            "rgb":         rgb,
            "bands":       band_arrays,
            "ndvi":        ndvi,
            "ndwi":        ndwi,
            "ndbi":        ndbi,
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
    Returns cluster_map, cluster_stats, label_map.
    """
    h, w = scene["shape"]
    bands = scene["bands"]

    feature_stack = np.stack([
        bands["B02"], bands["B03"], bands["B04"],
        bands["B08"], bands["B11"],
        scene["ndvi"], scene["ndwi"], scene["ndbi"],
    ], axis=-1)

    X = feature_stack.reshape(h * w, 8)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans        = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    cluster_map    = cluster_labels.reshape(h, w)

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
        elif stats["ndbi"] > 0.15 and stats["ndvi"] < 0.10:
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
    """Convert cluster map to an RGB uint8 image using LAND_COVER_COLORS."""
    cluster_map = kmeans_result["cluster_map"]
    label_map   = kmeans_result["label_map"]
    h, w        = cluster_map.shape

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c, label in label_map.items():
        color = LAND_COVER_COLORS.get(label, "#888888")
        rgb   = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        img[cluster_map == c] = rgb

    return img


# ---------------------------------------------------------------------------
# ML Layer 2: Random Forest
# ---------------------------------------------------------------------------

def run_random_forest(scene, n_estimators=100):
    """Train Random Forest with pseudo-labels and predict all pixels.

    Pseudo-labels are generated from strict spectral thresholds.
    Only high-confidence pixels are used as training data.
    """
    h, w      = scene["shape"]
    bands     = scene["bands"]
    ndvi_flat = scene["ndvi"].flatten()
    ndwi_flat = scene["ndwi"].flatten()
    ndbi_flat = scene["ndbi"].flatten()
    n_pixels  = h * w

    feature_stack = np.stack([
        bands["B02"], bands["B03"], bands["B04"],
        bands["B08"], bands["B11"],
        scene["ndvi"], scene["ndwi"], scene["ndbi"],
    ], axis=-1)

    X        = feature_stack.reshape(n_pixels, 8)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate pseudo-labels
    pseudo = np.zeros(n_pixels, dtype=int)
    pseudo[(ndwi_flat > 0.15) & (ndvi_flat < 0.05)]                               = 1  # Water
    pseudo[(ndvi_flat > 0.50)]                                                      = 2  # Dense veg
    pseudo[(ndvi_flat > 0.20) & (ndvi_flat <= 0.50) & (ndwi_flat < 0.1)]          = 3  # Crops
    pseudo[(ndbi_flat > 0.15) & (ndvi_flat < 0.10) & (ndwi_flat < 0.0)]           = 4  # Urban
    pseudo[(ndvi_flat < 0.08) & (ndwi_flat < -0.10) & (pseudo == 0)]              = 5  # Desert

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

    rf_labels = rf.predict(X_scaled)
    rf_map    = rf_labels.reshape(h, w)

    feature_names = ["B02", "B03", "B04", "B08", "B11", "NDVI", "NDWI", "NDBI"]
    importances   = rf.feature_importances_

    return {
        "rf_map":         rf_map,
        "importances":    importances,
        "feature_names":  feature_names,
        "labeled_pct":    float(100.0 * labeled / n_pixels),
        "class_counts": {
            "Water":             int((rf_labels == 1).sum()),
            "Dense vegetation":  int((rf_labels == 2).sum()),
            "Crops / sparse veg":int((rf_labels == 3).sum()),
            "Urban / built-up":  int((rf_labels == 4).sum()),
            "Desert / bare soil":int((rf_labels == 5).sum()),
        },
    }, None


def rf_to_color_image(rf_result):
    """Convert RF label map to an RGB uint8 image using LAND_COVER_COLORS."""
    rf_map = rf_result["rf_map"]
    h, w   = rf_map.shape

    id_to_label = {
        1: "Water", 2: "Dense vegetation", 3: "Crops / sparse veg",
        4: "Urban / built-up", 5: "Desert / bare soil",
    }
    img = np.zeros((h, w, 3), dtype=np.uint8)
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
2. Click Run Classification
3. Explore the K-means and Random Forest tabs
4. Click Get AI Interpretation for a structured analysis
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

    col_k, col_btn = st.columns([1, 3])
    with col_k:
        n_clusters = st.slider("K-means clusters", 3, 8, 5, key="lc_k")
    with col_btn:
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

    # --- Run ---
    if run_btn:
        if not lc_bbox:
            st.warning("Enter a location first.")
        else:
            date_range = f"{lc_date_start.isoformat()}/{lc_date_end.isoformat()}"
            st.session_state.lc_scene    = None
            st.session_state.lc_kmeans   = None
            st.session_state.lc_rf       = None
            st.session_state.lc_ai       = None
            st.session_state.lc_pending  = {
                "bbox":       lc_bbox,
                "date_range": date_range,
                "max_cloud":  lc_max_cloud,
                "n_clusters": n_clusters,
                "region":     lc_region_name,
            }
            st.rerun()

    # --- Execute pending run ---
    if st.session_state.get("lc_pending"):
        p = st.session_state.lc_pending
        st.session_state.lc_pending = None

        with st.spinner(f"Fetching Sentinel-2 bands for {p['region']} (5 bands, ~15 seconds)..."):
            scene, err = fetch_scene(p["bbox"], p["date_range"], p["max_cloud"])

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

    else:
        st.markdown("---")
        st.markdown(
            "**Type a location above, set a date range, and click Run Classification.**\n\n"
            "The module will fetch real Sentinel-2 imagery, run K-means and Random Forest "
            "on the same scene, and produce a structured AI interpretation."
        )
