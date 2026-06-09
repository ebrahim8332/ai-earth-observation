"""
corridor_risk.py — Corridor Risk Intelligence module for the EOIL portal.

Arc 2: Corridor and Vegetation Risk.

Three-layer architecture:
  Layer 1: Fetch six Sentinel-2 NDVI composites from GEE via getThumbURL()
  Layer 2: K-means, threshold classification, linear regression, Isolation Forest
  Layer 3: Groq/Gemini synthesises all four outputs into a risk brief

All algorithm logic is ported directly from notebook 07_corridor_risk.ipynb.
Results are cached in Streamlit session state so switching tabs does not re-run.
"""

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from PIL import Image
from io import BytesIO

import folium
from streamlit_folium import st_folium

import config
import ai_chain

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants — match the notebook exactly
# ---------------------------------------------------------------------------

# Pre-defined corridors — five geographies covering different vegetation regimes.
# Each bbox is [west, south, east, north] in decimal degrees, sized ~25 x 10 km
# so the 240x90 pixel grid stays at ~100m resolution.
#
# IMPORTANT — corridor alignment note:
# The NC corridor is the only validated study corridor. Its bbox was confirmed
# to produce clean NDVI results throughout notebook development.
# The other four corridors are ILLUSTRATIVE. Their bboxes are geographically
# plausible for the named region and vegetation type, but they were not traced
# to verified transmission line coordinates. For real client work, derive the
# bbox from actual line geometry (OpenStreetMap power=line, Global Energy
# Monitor, or utility GIS portals) and buffer by the ROW width.
#
# Seasonal note for Tanzania: April–September is the dry season in East Africa.
# NDVI slopes will be mostly negative (vegetation drying out). Vegetation that
# stays green through the dry season — water-dependent species, invasive acacias —
# is flagged as anomalous by Isolation Forest. That IS the encroachment signal.
CORRIDORS = {
    "Central NC, USA — Durham–Raleigh": {
        "bbox":    [-79.05, 35.89, -78.77, 35.98],
        "notes":   "24 km × 8 km  |  Mixed deciduous forest  |  Validated study corridor",
        "climate": "Temperate. NDVI rises April→September as canopy fills in. "
                   "Positive slopes = active encroachment toward the ROW.",
    },
    "East Midlands, UK — Nottinghamshire": {
        "bbox":    [-1.05, 52.90, -0.77, 52.99],
        "notes":   "24 km × 8 km  |  Mixed farmland and managed woodland  |  Illustrative corridor",
        "climate": "Temperate maritime. Growing season April–September. "
                   "Woodland edge encroachment on agricultural ROW corridors.",
    },
    "Hunter Valley, Australia — NSW": {
        "bbox":    [151.00, -32.85, 151.28, -32.76],
        "notes":   "25 km × 8 km  |  Eucalyptus scrub and dry sclerophyll forest  |  Illustrative corridor",
        "climate": "Southern Hemisphere. April–September is austral autumn/winter — "
                   "vegetation browning. Eucalyptus that stays green flags as anomalous.",
    },
    "Gauteng, South Africa — Johannesburg–Pretoria": {
        "bbox":    [27.90, -26.00, 28.18, -25.91],
        "notes":   "25 km × 8 km  |  Highveld grassland  |  Illustrative corridor",
        "climate": "Southern Hemisphere dry season April–September. "
                   "Invasive wattle and lantana maintain high NDVI — strong anomaly signal.",
    },
    "Dar es Salaam–Morogoro, Tanzania": {
        "bbox":    [38.20, -7.05, 38.48, -6.96],
        "notes":   "25 km × 8 km  |  Miombo woodland transition zone  |  Illustrative corridor",
        "climate": "East African dry season April–September. Most vegetation dries out. "
                   "Riparian forest and invasive acacias stay green — high anomaly confidence.",
    },
}

# Default corridor key (used on first load)
DEFAULT_CORRIDOR_KEY = "Central NC, USA — Durham–Raleigh"

# Fixed pixel dimensions — guarantees all six arrays are the same shape
IMG_W = 240
IMG_H = 90

NDVI_MIN = 0.0
NDVI_MAX = 1.0

# Six 45-day time windows across the 2023 growing season
TIME_STEPS = [
    ('2023-03-20', '2023-05-03', 'Apr 2023'),
    ('2023-04-18', '2023-06-01', 'May 2023'),
    ('2023-05-18', '2023-07-01', 'Jun 2023'),
    ('2023-06-17', '2023-07-31', 'Jul 2023'),
    ('2023-07-17', '2023-08-30', 'Aug 2023'),
    ('2023-08-16', '2023-09-29', 'Sep 2023'),
]
N_STEPS = len(TIME_STEPS)

# Risk class definitions
RISK_LABELS = {4: 'Critical', 3: 'Warning', 2: 'Monitor', 1: 'Clear'}
RISK_COLORS = {4: '#d73027', 3: '#fd8d3c', 2: '#ffffb3', 1: '#d9f0a3'}

SLOPE_GROWING  =  0.02   # NDVI/month — "actively growing"
SLOPE_FAST     =  0.03   # NDVI/month — "fast growing" (grass zone concern)

# Threshold classification NDVI cutoffs
THRESH_DENSE  = 0.60
THRESH_SHRUB  = 0.35
THRESH_GRASS  = 0.10

# K-means cluster count
K = 4
KM_LABELS_NAMED = ['Cleared ROW', 'Grass/forb zone', 'Shrub zone', 'Dense forest']
KM_COLORS       = ['#d4b483', '#ffffb3', '#a6d96a', '#1a9850']


# ---------------------------------------------------------------------------
# GEE helpers — ported from notebook cells 3, 4, 5, 6
# ---------------------------------------------------------------------------

def _init_gee():
    """Return an initialised GEE ee module, or None on failure.

    app.py calls gee_timeseries.init_gee() on startup, which runs ee.Initialize().
    If GEE is already initialised when this function runs, skip the init call —
    calling ee.Initialize() twice raises an exception that would otherwise make
    this function return None and incorrectly report that GEE is unavailable.
    """
    try:
        import ee
        import json
        import streamlit as st

        # If GEE is already initialised (by app.py startup), just return the module.
        try:
            ee.data.getInfo('projects/earthengine-public')
            return ee
        except ee.ee_exception.EEException:
            pass  # Not yet initialised — proceed with init below.

        secrets = st.secrets.get("gee", {})
        svc_json = secrets.get("GEE_SERVICE_ACCOUNT_JSON", "")
        project  = secrets.get("GEE_PROJECT", "")

        if svc_json:
            info        = json.loads(svc_json)
            credentials = ee.ServiceAccountCredentials(info["client_email"], key_data=svc_json)
            ee.Initialize(credentials=credentials, project=project)
        else:
            ee.Initialize(project=project)

        return ee
    except Exception:
        return None


def _mask_s2_clouds(ee, image):
    """QA60 pixel-level cloud mask — bits 10 and 11."""
    qa          = image.select('QA60')
    cloud_mask  = qa.bitwiseAnd(1 << 10).eq(0).And(
                  qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)


def _compute_ndvi(ee, image):
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')


def _build_composite(ee, start_date, end_date, geometry):
    """Build a cloud-masked median NDVI composite.

    80% scene-level filter is intentionally permissive — the QA60 mask
    removes actual cloudy pixels. 20% was discarding all May scenes over NC.
    """
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
        .map(lambda img: _mask_s2_clouds(ee, img))
        .map(lambda img: _compute_ndvi(ee, img))
    )
    return collection.median().clip(geometry)


def _download_ndvi_array(gee_image, geometry):
    """Download NDVI composite as PNG via getThumbURL and rescale to float.

    GEE maps [NDVI_MIN, NDVI_MAX] → grayscale [0, 255].
    We invert: ndvi = pixel / 255.0 * (MAX - MIN) + MIN.
    Fixed dimensions guarantee every array is (IMG_H, IMG_W).
    """
    try:
        thumb_params = {
            'min':        NDVI_MIN,
            'max':        NDVI_MAX,
            'dimensions': f'{IMG_W}x{IMG_H}',
            'region':     geometry,
            'format':     'png',
        }
        url  = gee_image.select('NDVI').getThumbURL(thumb_params)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        img  = Image.open(BytesIO(resp.content)).convert('L')
        arr  = np.array(img, dtype=np.float32) / 255.0
        arr  = arr * (NDVI_MAX - NDVI_MIN) + NDVI_MIN
        return arr
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_ndvi_arrays(_bbox):
    """Fetch all six NDVI composites from GEE and return as a list of arrays.

    Cached by bbox so re-running the portal does not re-fetch from GEE.
    _bbox is a tuple so it is hashable by st.cache_data.
    Returns (ndvi_arrays, labels, error_msg).
    """
    ee = _init_gee()
    if ee is None:
        return None, None, "GEE not available — check credentials in secrets."

    west, south, east, north = _bbox
    corridor_geom = ee.Geometry.Rectangle([west, south, east, north])

    ndvi_arrays = []
    labels      = []

    for start, end, label in TIME_STEPS:
        try:
            composite = _build_composite(ee, start, end, corridor_geom)
            arr       = _download_ndvi_array(composite, corridor_geom)
            if arr is None:
                arr = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            ndvi_arrays.append(arr)
            labels.append(label)
        except Exception as e:
            ndvi_arrays.append(np.zeros((IMG_H, IMG_W), dtype=np.float32))
            labels.append(label)

    return ndvi_arrays, labels, None


# ---------------------------------------------------------------------------
# Layer 2: Algorithm functions — ported from notebook cells 7–16
# ---------------------------------------------------------------------------

def run_algorithms(ndvi_arrays):
    """Run all four algorithms and return a results dict.

    Ported directly from notebook cells 9–16 (feature matrix through risk composite).
    """
    H, W = ndvi_arrays[0].shape
    n_pixels = H * W

    # Stack into 3D array: (H, W, N_STEPS)
    ndvi_stack = np.stack(ndvi_arrays, axis=2)
    ndvi_final = ndvi_arrays[-1]   # September snapshot

    # Flatten for ML
    ts_flat = ndvi_stack.reshape(n_pixels, N_STEPS)

    # --- Per-pixel slope (Algorithm 3 — linear regression) ---
    x = np.arange(N_STEPS, dtype=np.float32)
    slopes = np.apply_along_axis(
        lambda y: np.polyfit(x, y, 1)[0] if not np.any(np.isnan(y)) else 0.0,
        axis=1,
        arr=ts_flat
    )
    slope_map = slopes.reshape(H, W)

    # --- Feature matrix for Isolation Forest ---
    features = np.column_stack([
        ts_flat.mean(axis=1),
        ts_flat.std(axis=1),
        slopes,
        ts_flat.min(axis=1),
        ts_flat.max(axis=1),
        ts_flat[:, -1],
    ])

    # --- Algorithm 1: K-means on September snapshot ---
    X_snapshot  = ndvi_final.flatten().reshape(-1, 1)
    scaler_km   = StandardScaler()
    X_km_scaled = scaler_km.fit_transform(X_snapshot)

    kmeans      = KMeans(n_clusters=K, n_init=15, random_state=42)
    km_labels   = kmeans.fit_predict(X_km_scaled)

    cluster_stats = []
    for c in range(K):
        mask = km_labels == c
        cluster_stats.append({
            'cluster': c,
            'count':   int(mask.sum()),
            'mean_ndvi': float(ndvi_final.flatten()[mask].mean()),
        })
    cluster_stats.sort(key=lambda s: s['mean_ndvi'])

    km_cluster_label = {}
    km_cluster_color = {}
    for rank, stats in enumerate(cluster_stats):
        km_cluster_label[stats['cluster']] = KM_LABELS_NAMED[rank]
        km_cluster_color[stats['cluster']] = KM_COLORS[rank]

    km_map     = km_labels.reshape(H, W)
    km_int_map = np.vectorize(
        lambda c: KM_LABELS_NAMED.index(km_cluster_label[c])
    )(km_map)

    # --- Algorithm 2: Threshold classification ---
    thresh_map = np.zeros((H, W), dtype=np.int8)
    thresh_map[ndvi_final >= THRESH_DENSE]                                         = 3
    thresh_map[(ndvi_final >= THRESH_SHRUB) & (ndvi_final < THRESH_DENSE)]        = 2
    thresh_map[(ndvi_final >= THRESH_GRASS) & (ndvi_final < THRESH_SHRUB)]        = 1
    thresh_map[ndvi_final < THRESH_GRASS]                                          = 0

    km_thresh_agreement = float(100.0 * (km_int_map == thresh_map).mean())

    # --- Algorithm 4: Isolation Forest ---
    scaler_if   = StandardScaler()
    X_if_scaled = scaler_if.fit_transform(features)

    iso_forest  = IsolationForest(n_estimators=200, contamination=0.10, random_state=42, n_jobs=-1)
    iso_labels  = iso_forest.fit_predict(X_if_scaled)
    iso_scores  = iso_forest.decision_function(X_if_scaled)

    anomaly_flag = (iso_labels == -1)
    anomaly_map  = anomaly_flag.reshape(H, W)
    score_map    = iso_scores.reshape(H, W)

    # --- Combined risk composite ---
    thresh_flat  = thresh_map.flatten()
    risk_flat    = np.ones(n_pixels, dtype=np.int8)

    risk_flat[(thresh_flat == 2) & (slopes <= SLOPE_GROWING)]                          = 2
    risk_flat[(thresh_flat == 1) & (slopes > SLOPE_FAST)]                              = 2
    risk_flat[(thresh_flat == 2) & (slopes > SLOPE_GROWING)]                           = 3
    risk_flat[(thresh_flat == 3) & (slopes <= SLOPE_GROWING) & anomaly_flag]           = 3
    risk_flat[(thresh_flat == 3) & (slopes > SLOPE_GROWING)]                           = 4
    risk_flat[thresh_flat == 0]                                                         = 1

    risk_map = risk_flat.reshape(H, W)

    return {
        'H': H, 'W': W, 'n_pixels': n_pixels,
        'ndvi_final': ndvi_final,
        'slope_map': slope_map,
        'slopes': slopes,
        'thresh_map': thresh_map,
        'km_int_map': km_int_map,
        'km_cluster_label': km_cluster_label,
        'km_cluster_color': km_cluster_color,
        'cluster_stats': cluster_stats,
        'anomaly_map': anomaly_map,
        'anomaly_flag': anomaly_flag,
        'score_map': score_map,
        'risk_map': risk_map,
        'risk_flat': risk_flat,
        'km_thresh_agreement': km_thresh_agreement,
    }


# ---------------------------------------------------------------------------
# Matplotlib figure helpers — ported from notebook cells 7, 9, 10, 12, 14
# ---------------------------------------------------------------------------

def _fig_to_bytes(fig):
    """Convert a Matplotlib figure to PNG bytes for st.image()."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def build_ndvi_strip_fig(ndvi_arrays, labels):
    """Six-panel NDVI strip — notebook cell 7."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(
        'Layer 1 — Sentinel-2 NDVI Corridor Time Series\n'
        'April–September 2023  |  100 m resolution',
        fontsize=11, fontweight='bold'
    )
    for idx, (arr, label) in enumerate(zip(ndvi_arrays, labels)):
        r, c = divmod(idx, 3)
        ax   = axes[r][c]
        im   = ax.imshow(arr, cmap='RdYlGn', vmin=0.0, vmax=0.85, aspect='auto')
        ax.set_title(f'{label}\nmean NDVI: {arr.mean():.3f}', fontsize=9)
        ax.axis('off')
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0.0, 0.85)),
        ax=axes.ravel().tolist(),
        shrink=0.55, pad=0.02
    )
    cbar.set_label('NDVI', fontsize=9)
    plt.tight_layout()
    return fig


def build_classification_fig(results):
    """Three-panel K-means vs threshold — notebook cell 9."""
    zone_colors = KM_COLORS
    zone_labels = KM_LABELS_NAMED
    zone_cmap   = mcolors.ListedColormap(zone_colors)
    zone_norm   = mcolors.BoundaryNorm(np.arange(5) - 0.5, zone_cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        'Algorithms 1 & 2: K-means vs Threshold Classification\nSeptember 2023 NDVI snapshot',
        fontsize=10, fontweight='bold'
    )

    im0 = axes[0].imshow(results['ndvi_final'], cmap='RdYlGn', vmin=0, vmax=0.85, aspect='auto')
    axes[0].set_title('September NDVI', fontsize=9)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(results['km_int_map'], cmap=zone_cmap, norm=zone_norm, aspect='auto')
    axes[1].set_title('K-means (unsupervised)', fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(results['thresh_map'], cmap=zone_cmap, norm=zone_norm, aspect='auto')
    axes[2].set_title('Threshold (rule-based)', fontsize=9)
    axes[2].axis('off')

    patches = [mpatches.Patch(color=zone_colors[i], label=zone_labels[i]) for i in range(4)]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    return fig


def build_slope_fig(results):
    """Slope map + histogram — notebook cell 10."""
    slope_map = results['slope_map']
    slopes    = results['slopes']

    slope_abs = max(abs(np.percentile(slope_map, 5)), abs(np.percentile(slope_map, 95)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        'Algorithm 3: NDVI Growth Slope Per Pixel\nApril–September 2023  |  Units: NDVI change/month',
        fontsize=10, fontweight='bold'
    )

    im = axes[0].imshow(slope_map, cmap='RdYlGn', vmin=-slope_abs, vmax=slope_abs, aspect='auto')
    axes[0].set_title('Growth slope  (green=growing, red=receding)', fontsize=9)
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='NDVI/month')

    axes[1].hist(slopes, bins=60, color='#3a9e3a', edgecolor='none', alpha=0.75)
    axes[1].axvline(0, color='black', linewidth=1.2, linestyle='--', label='Zero')
    axes[1].axvline(slopes.mean(), color='red', linewidth=1.2,
                    label=f'Mean: {slopes.mean():.4f}')
    axes[1].set_xlabel('NDVI slope (units/month)', fontsize=9)
    axes[1].set_ylabel('Pixel count', fontsize=9)
    axes[1].set_title('Slope distribution', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    pct_positive = 100.0 * (slopes > 0).mean()
    axes[1].annotate(
        f'{pct_positive:.0f}% positive slope',
        xy=(0.62, 0.82), xycoords='axes fraction', fontsize=9, color='#1a9850',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1a9850', alpha=0.8)
    )
    plt.tight_layout()
    return fig


def build_anomaly_fig(results):
    """Isolation Forest anomaly map — notebook cell 12."""
    anomaly_map = results['anomaly_map']
    score_map   = results['score_map']
    ndvi_final  = results['ndvi_final']
    n_anomalous = results['anomaly_flag'].sum()
    n_pixels    = results['n_pixels']

    score_vabs = np.percentile(np.abs(score_map), 95)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        'Algorithm 4: Isolation Forest — Anomalous Growth Zones\n'
        '10% contamination rate  |  6 time series features',
        fontsize=10, fontweight='bold'
    )

    im_score = axes[0].imshow(score_map, cmap='coolwarm_r',
                               vmin=-score_vabs, vmax=score_vabs, aspect='auto')
    axes[0].set_title('Anomaly score  (red=more anomalous)', fontsize=9)
    axes[0].axis('off')
    plt.colorbar(im_score, ax=axes[0], fraction=0.046, pad=0.04, label='Score')

    axes[1].imshow(ndvi_final, cmap='Greys_r', vmin=0, vmax=0.85, aspect='auto', alpha=0.6)
    overlay = np.ma.masked_where(~anomaly_map, anomaly_map.astype(float))
    axes[1].imshow(overlay, cmap='Reds', vmin=0, vmax=1, aspect='auto', alpha=0.8)
    axes[1].set_title(
        f'Flagged pixels: {n_anomalous:,} ({100*n_anomalous/n_pixels:.0f}%)  '
        f'(red=anomalous, grey=NDVI background)', fontsize=9
    )
    axes[1].axis('off')
    plt.tight_layout()
    return fig


def build_risk_composite_fig(results):
    """Four-panel risk summary — notebook cell 14."""
    risk_flat  = results['risk_flat']
    risk_map   = results['risk_map']
    n_pixels   = results['n_pixels']
    slope_map  = results['slope_map']
    anomaly_map = results['anomaly_map']
    ndvi_final  = results['ndvi_final']

    risk_cmap = mcolors.ListedColormap([RISK_COLORS[i] for i in [1, 2, 3, 4]])
    risk_norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], risk_cmap.N)
    slope_abs = np.percentile(np.abs(slope_map), 95)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(
        'Layer 2 Summary: Four Algorithms → Combined Risk Composite\n'
        'Central NC Transmission Corridor  |  April–September 2023',
        fontsize=10, fontweight='bold'
    )

    axes[0].imshow(ndvi_final, cmap='RdYlGn', vmin=0, vmax=0.85, aspect='auto')
    axes[0].set_title('September NDVI\n(Algorithms 1 & 2 input)', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(slope_map, cmap='RdYlGn', vmin=-slope_abs, vmax=slope_abs, aspect='auto')
    axes[1].set_title('NDVI growth slope\n(Algorithm 3)', fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(ndvi_final, cmap='Greys_r', vmin=0, vmax=0.85, aspect='auto', alpha=0.5)
    anom_overlay = np.ma.masked_where(~anomaly_map, anomaly_map.astype(float))
    axes[2].imshow(anom_overlay, cmap='Reds', vmin=0, vmax=1, aspect='auto', alpha=0.85)
    axes[2].set_title('Isolation Forest anomalies\n(Algorithm 4)', fontsize=9)
    axes[2].axis('off')

    axes[3].imshow(risk_map, cmap=risk_cmap, norm=risk_norm, aspect='auto')
    axes[3].set_title('Combined risk composite\n(all four algorithms)', fontsize=9)
    axes[3].axis('off')

    risk_patches = [
        mpatches.Patch(
            color=RISK_COLORS[i],
            label=f'{RISK_LABELS[i]}  ({(risk_flat==i).sum():,} px)'
        )
        for i in [4, 3, 2, 1]
    ]
    fig.legend(handles=risk_patches, loc='lower center', ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Layer 3: AI brief — ported from notebook cells 19, 20
# ---------------------------------------------------------------------------

def build_context_string(results):
    """Build the structured algorithm summary string sent to the AI.

    Mirrors notebook cell 19 exactly.
    """
    rf  = results['risk_flat']
    n   = results['n_pixels']
    sl  = results['slopes']
    km  = results['km_int_map']
    th  = results['thresh_map']
    af  = results['anomaly_flag']
    ndvi_f = results['ndvi_final'].flatten()

    risk_pcts = {cls: 100.0 * (rf == cls).mean() for cls in [1, 2, 3, 4]}
    risk_slope_means = {
        cls: float(sl[rf == cls].mean()) if (rf == cls).any() else 0.0
        for cls in [1, 2, 3, 4]
    }

    anom_in_critical = 100.0 * af[rf == 4].mean() if (rf == 4).any() else 0.0
    anom_in_warning  = 100.0 * af[rf == 3].mean() if (rf == 3).any() else 0.0
    km_thresh_agr    = results['km_thresh_agreement']

    corridor_name = results.get('corridor_name', 'Utility transmission corridor')
    climate_note  = results.get('climate_note', '')

    context = f"""CORRIDOR RISK ANALYSIS SUMMARY
Study area: {corridor_name}
Seasonal context: {climate_note}
Data: Sentinel-2 NDVI, 100 m resolution, April–September 2023 (6 monthly composites)
Total pixels analysed: {n:,}

=== ALGORITHM OUTPUTS ===

Algorithm 1 — K-means clustering (k=4, September snapshot):
  Dense forest class area:  {100*(km==3).mean():.1f}%
  Shrub zone area:          {100*(km==2).mean():.1f}%
  Grass/forb zone area:     {100*(km==1).mean():.1f}%
  Cleared ROW area:         {100*(km==0).mean():.1f}%

Algorithm 2 — Threshold classification (fixed NDVI thresholds):
  Dense forest (NDVI >= 0.60):  {100*(th==3).mean():.1f}%
  Shrub zone (0.35–0.60):       {100*(th==2).mean():.1f}%
  Grass/forb zone (0.10–0.35):  {100*(th==1).mean():.1f}%
  Cleared ROW (< 0.10):         {100*(th==0).mean():.1f}%
  K-means vs threshold agreement: {km_thresh_agr:.1f}%

Algorithm 3 — Linear regression (NDVI slope per pixel, April–September):
  Mean slope across corridor:    {sl.mean():+.4f} NDVI/month
  Pixels with positive slope:    {100*(sl>0).mean():.1f}%
  Actively growing (>+0.02/mo):  {100*(sl>0.02).mean():.1f}%
  Rapidly growing (>+0.03/mo):   {100*(sl>0.03).mean():.1f}%
  Mean slope in dense forest:    {sl[th.flatten()==3].mean():+.4f} NDVI/month
  Mean slope in shrub zone:      {sl[th.flatten()==2].mean():+.4f} NDVI/month

Algorithm 4 — Isolation Forest (6 features: mean, std, slope, min, max, final NDVI):
  Anomalous pixels flagged:       {100*af.mean():.1f}%
  Anomaly rate in Critical zones: {anom_in_critical:.1f}%
  Anomaly rate in Warning zones:  {anom_in_warning:.1f}%

=== COMBINED RISK COMPOSITE ===

  CRITICAL — dense vegetation + active growth:      {risk_pcts[4]:.1f}%  (mean slope: {risk_slope_means[4]:+.4f}/mo)
  WARNING  — shrub growing OR dense + anomaly:      {risk_pcts[3]:.1f}%  (mean slope: {risk_slope_means[3]:+.4f}/mo)
  MONITOR  — shrub stable OR grass growing fast:    {risk_pcts[2]:.1f}%  (mean slope: {risk_slope_means[2]:+.4f}/mo)
  CLEAR    — ROW cleared or stable low NDVI:        {risk_pcts[1]:.1f}%  (mean slope: {risk_slope_means[1]:+.4f}/mo)
"""
    return context, risk_pcts, risk_slope_means


AI_SYSTEM_PROMPT = """You are a utility infrastructure analyst specialising in vegetation
encroachment risk for transmission corridors. You receive structured satellite analysis
outputs from four ML algorithms and produce a decision-ready inspection brief.

Your output must include three sections:
1. RISK TABLE — one row per risk level (Critical, Warning, Monitor, Clear) with:
   risk level, area percentage, mean NDVI slope, primary concern, and recommended action.
2. INSPECTION PRIORITY SCORING — rank the top 3 priority zones by weighted score
   (change magnitude 35% + proximity to ROW 30% + persistence across methods 20% +
   confidence 15%). Show the score and reasoning for each.
3. FIELD INSPECTION BRIEF — what inspection crews should look for, where to start,
   and what would confirm or refute the satellite findings.

Be specific. Every claim must reference a number from the input data.
Do not hedge excessively. Make a clear recommendation."""


def get_ai_brief(context_str):
    """Call ai_chain.complete() with the risk context. Returns (text, model)."""
    prompt = f"{AI_SYSTEM_PROMPT}\n\nProduce the three-section inspection brief:\n\n{context_str}"
    text, model = ai_chain.complete(
        prompt,
        groq_key=config.GROQ_API_KEY,
        gemini_key=config.GEMINI_API_KEY,
    )
    return text, model


# ---------------------------------------------------------------------------
# Plotly summary chart — risk class breakdown
# ---------------------------------------------------------------------------

def build_risk_breakdown_chart(risk_flat, n_pixels):
    """Horizontal bar chart of risk class percentages."""
    order  = [4, 3, 2, 1]
    labels = [RISK_LABELS[i] for i in order]
    pcts   = [100.0 * (risk_flat == i).sum() / n_pixels for i in order]
    colors = [RISK_COLORS[i] for i in order]

    fig = go.Figure(go.Bar(
        x=pcts,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1f}%' for p in pcts],
        textposition='outside',
    ))
    fig.update_layout(
        title='Combined risk composite — area by class',
        xaxis_title='% of corridor area',
        height=280,
        margin=dict(l=10, r=60, t=40, b=30),
        plot_bgcolor='#f8f8f8',
        paper_bgcolor='#ffffff',
    )
    return fig


# ---------------------------------------------------------------------------
# Main render function — called from app.py
# ---------------------------------------------------------------------------

def render():
    """Render the Corridor Risk Intelligence module.

    Called from app.py when the Corridor Risk module is selected.
    """

    st.subheader("🌿 Corridor Risk Intelligence")
    st.caption(
        "Multi-algorithm vegetation encroachment analysis for utility transmission corridors. "
        "Four ML algorithms on Sentinel-2 NDVI time series produce a prioritised risk composite."
    )

    # GEE connection status — shown at the top so it is visible before any interaction
    gee_ok = st.session_state.get("gee_available", False)
    if gee_ok:
        st.caption("🟢 GEE connected — live Sentinel-2 data active.")
    else:
        st.caption("🔴 GEE not connected. This module requires live GEE credentials.")

    with st.expander("ℹ️ What does this module do?", expanded=False):
        st.markdown("""
**This module answers two questions:**
1. Where is vegetation dense today?
2. Where is it growing toward the corridor?

A snapshot alone is not enough. Dense vegetation that has been stable for years may be low priority.
Thin vegetation growing fast toward the ROW is high priority even if it looks fine today.

**Four algorithms on the same data:**

| Algorithm | What it measures |
|---|---|
| K-means (unsupervised) | Groups pixels by spectral similarity. No labels needed. |
| Threshold classification | Fixed NDVI cutoffs — transparent and auditable. |
| Linear regression per pixel | Growth rate and direction across 6 monthly composites. |
| Isolation Forest | Flags pixels whose time series pattern is unusual. |

**Three-layer architecture:**
- **Layer 1:** GEE Sentinel-2 NDVI composites — six 45-day windows, April–September 2023
- **Layer 2:** All four algorithms applied to the same pixel arrays
- **Layer 3:** AI synthesis of algorithm outputs into a prioritised inspection brief

**Five pre-defined corridors** span North America, Europe, Australia, Africa, and East Africa.
Each corridor has a different vegetation type and climate regime — the same four algorithms
produce different risk patterns depending on local conditions.
        """)

    # --- Corridor selector ---
    corridor_key = st.selectbox(
        "Select study corridor",
        options=list(CORRIDORS.keys()),
        index=list(CORRIDORS.keys()).index(DEFAULT_CORRIDOR_KEY),
        key="cr_corridor_key",
    )
    corridor = CORRIDORS[corridor_key]

    st.info(
        f"**Corridor:** {corridor_key}  \n"
        f"**Area:** {corridor['notes']}  \n"
        f"**Seasonal context:** {corridor['climate']}  \n"
        f"**Data:** Sentinel-2 NDVI via Google Earth Engine  |  100 m resolution  |  April–September 2023  \n"
        f"{'⚠️ Illustrative corridor — bbox not traced to verified line geometry. For production use, derive from OpenStreetMap power lines or utility GIS data.' if corridor_key != DEFAULT_CORRIDOR_KEY else '✅ Validated study corridor — bbox confirmed through notebook development.'}"
    )

    # --- Corridor location map ---
    # Draws the analysis footprint on an interactive map.
    # Updates immediately when the corridor selectbox changes — no GEE required.
    west, south, east, north = corridor['bbox']
    centre_lat = (south + north) / 2
    centre_lon = (west  + east)  / 2

    # Tile layer toggle — two options covering the main use cases
    TILE_OPTIONS = {
        "Satellite": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attr":  "Esri World Imagery",
        },
        "Street map": {
            "tiles": "OpenStreetMap",
            "attr":  "OpenStreetMap contributors",
        },
    }

    tile_choice = st.radio(
        "Map view",
        options=list(TILE_OPTIONS.keys()),
        index=0,
        horizontal=True,
        key="cr_tile_choice",
    )
    tile_cfg = TILE_OPTIONS[tile_choice]

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=10,
        tiles=tile_cfg["tiles"],
        attr=tile_cfg["attr"],
    )

    # Draw the corridor bounding box as a rectangle
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color='#f7a200',
        weight=3,
        fill=True,
        fill_color='#f7a200',
        fill_opacity=0.15,
        tooltip=corridor_key,
    ).add_to(m)

    # Small crosshair at the centre so the corridor is easy to spot when zoomed out
    folium.Marker(
        location=[centre_lat, centre_lon],
        tooltip=corridor_key,
        icon=folium.DivIcon(
            html='<div style="font-size:18px; color:#f7a200; font-weight:bold; '
                 'text-shadow: 0 0 3px #000;">✦</div>',
            icon_size=(20, 20),
            icon_anchor=(10, 10),
        ),
    ).add_to(m)

    st_folium(m, height=420, use_container_width=True, returned_objects=[])

    # --- Run button ---
    run_btn = st.button(
        "▶ Run Analysis",
        type="primary",
        key="cr_run",
    )

    # --- Session state initialisation ---
    for k, v in [
        ("cr_ndvi_arrays", None),
        ("cr_labels",      None),
        ("cr_results",     None),
        ("cr_context",     None),
        ("cr_ai_text",     None),
        ("cr_ai_model",    None),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    # --- Trigger analysis ---
    if run_btn:
        if not gee_ok:
            st.error("GEE credentials required. Add GEE_SERVICE_ACCOUNT_JSON to Streamlit secrets.")
        else:
            st.session_state.cr_ndvi_arrays = None
            st.session_state.cr_results     = None
            st.session_state.cr_context     = None
            st.session_state.cr_ai_text     = None
            st.session_state.cr_pending     = True
            st.session_state.cr_pending_key = corridor_key   # capture the selected corridor
            st.rerun()

    # --- Execute pending run ---
    if st.session_state.get("cr_pending"):
        st.session_state.cr_pending = None
        # Read selected corridor from session state (set when run button was pressed)
        _selected_key  = st.session_state.get("cr_pending_key", DEFAULT_CORRIDOR_KEY)
        _selected_corr = CORRIDORS[_selected_key]
        bbox_tuple     = tuple(_selected_corr["bbox"])

        with st.spinner("Fetching Sentinel-2 NDVI composites from GEE (6 months — allow 60–90 seconds)..."):
            ndvi_arrays, labels, err = fetch_ndvi_arrays(bbox_tuple)

        if err or ndvi_arrays is None:
            st.error(f"GEE fetch failed: {err}")
        else:
            with st.spinner("Running four ML algorithms..."):
                results = run_algorithms(ndvi_arrays)

            # Inject corridor metadata so build_context_string() can name the location
            results['corridor_name'] = _selected_key
            results['climate_note']  = _selected_corr['climate']

            context_str, risk_pcts, risk_slope_means = build_context_string(results)

            st.session_state.cr_ndvi_arrays = ndvi_arrays
            st.session_state.cr_labels      = labels
            st.session_state.cr_results     = results
            st.session_state.cr_context     = context_str

            n_valid = sum(1 for a in ndvi_arrays if a.mean() > 0.01)
            st.success(
                f"Analysis complete — {n_valid}/6 composites with valid data, "
                f"{results['n_pixels']:,} pixels per time step."
            )

    # --- Display results ---
    if (st.session_state.cr_ndvi_arrays is not None and
            st.session_state.cr_results is not None):

        ndvi_arrays = st.session_state.cr_ndvi_arrays
        labels      = st.session_state.cr_labels
        results     = st.session_state.cr_results
        context_str = st.session_state.cr_context

        def section_break():
            st.markdown(
                '<hr style="border: none; border-top: 3px solid #d0d0d0; margin: 24px 0 16px 0;">',
                unsafe_allow_html=True,
            )

        # ===================================================================
        # LAYER 1: NDVI time series strip
        # ===================================================================
        section_break()
        st.subheader("📡 Layer 1 — Signal Processing")
        st.caption(
            "Six cloud-masked median NDVI composites from Sentinel-2. "
            "Each panel is a 45-day window. Red = bare/low NDVI. Green = dense vegetation."
        )

        fig_strip = build_ndvi_strip_fig(ndvi_arrays, labels)
        st.image(_fig_to_bytes(fig_strip), use_container_width=True)

        # NDVI summary metrics
        means = [a.mean() for a in ndvi_arrays]
        cols  = st.columns(6)
        for col, label, mean_val in zip(cols, labels, means):
            col.metric(label, f"{mean_val:.3f}")

        with st.expander("ℹ️ About Layer 1 — cloud handling", expanded=False):
            st.markdown("""
**Two-stage cloud handling:**

- **Stage 1 (scene-level):** Scenes with > 80% cloud cover are excluded. This filter is
  intentionally permissive. A scene that is 60% cloudy overall may still have clear pixels
  over our small corridor.

- **Stage 2 (pixel-level):** The Sentinel-2 QA60 band flags individual cloudy pixels.
  Flagged pixels are masked before the median composite is computed. Only clear-sky
  reflectance values contribute to each monthly composite.

The median then picks the middle value across all remaining clear pixels in the window.
This is robust against any residual cloud edges — a single cloudy pixel cannot pull the
composite away from the true surface signal.
            """)

        # ===================================================================
        # LAYER 2: Four algorithms
        # ===================================================================
        section_break()
        st.subheader("🤖 Layer 2 — Four ML Algorithms")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Classification",
            "📈 Growth Slope",
            "🔍 Anomaly Detection",
            "⚠️ Risk Composite",
            "📋 Statistics",
        ])

        # --- Tab 1: Classification (K-means + threshold) ---
        with tab1:
            st.markdown("**Algorithms 1 & 2: K-means vs Threshold Classification**")
            st.markdown(
                "Both algorithms classify the September snapshot into four vegetation zones. "
                "K-means learns the clusters from the data. Threshold applies fixed NDVI rules. "
                f"Agreement: **{results['km_thresh_agreement']:.1f}%** of pixels."
            )
            fig_class = build_classification_fig(results)
            st.image(_fig_to_bytes(fig_class), use_container_width=True)

            # Zone breakdown table
            thresh_map = results['thresh_map']
            n_pixels   = results['n_pixels']
            zone_names = {0: 'Cleared ROW', 1: 'Grass/forb zone', 2: 'Shrub zone', 3: 'Dense forest'}
            ranges_str = {0: '< 0.10', 1: '0.10–0.35', 2: '0.35–0.60', 3: '>= 0.60'}
            rows = []
            for cls_id in range(4):
                count = (thresh_map == cls_id).sum()
                rows.append({
                    'Zone':       zone_names[cls_id],
                    'NDVI range': ranges_str[cls_id],
                    'Pixels':     f"{count:,}",
                    '% Area':     f"{100.0*count/n_pixels:.1f}%",
                })
            import pandas as pd
            st.table(pd.DataFrame(rows))

        # --- Tab 2: Growth slope ---
        with tab2:
            st.markdown("**Algorithm 3: Linear Regression Per Pixel**")
            st.markdown(
                "NDVI slope across six monthly composites. "
                "Green = vegetation increasing. Red = decreasing. "
                f"Mean slope: **{results['slopes'].mean():+.4f}** NDVI/month."
            )
            fig_slope = build_slope_fig(results)
            st.image(_fig_to_bytes(fig_slope), use_container_width=True)

            slopes = results['slopes']
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Mean slope", f"{slopes.mean():+.4f}/mo")
            sc2.metric("Actively growing (>+0.02)", f"{100*(slopes>0.02).mean():.1f}%")
            sc3.metric("Rapidly growing (>+0.03)", f"{100*(slopes>0.03).mean():.1f}%")

        # --- Tab 3: Anomaly detection ---
        with tab3:
            st.markdown("**Algorithm 4: Isolation Forest**")
            n_anom = results['anomaly_flag'].sum()
            st.markdown(
                f"Flags pixels with unusual NDVI time series patterns. "
                f"**{n_anom:,} pixels** ({100*n_anom/n_pixels:.0f}%) flagged as anomalous. "
                "Does not require labelled data — learns the normal corridor pattern from the data."
            )
            fig_anom = build_anomaly_fig(results)
            st.image(_fig_to_bytes(fig_anom), use_container_width=True)

            with st.expander("ℹ️ What does Isolation Forest detect?", expanded=False):
                st.markdown("""
Isolation Forest builds an ensemble of random decision trees. For each pixel it
measures how quickly that pixel can be isolated from all others by random splits.
Anomalous pixels are isolated faster — their feature values are unusual.

In a corridor context, anomalous pixels often correspond to:
- Unusually rapid NDVI increase (fast-growing invasive species)
- High NDVI combined with a rising slope (large expanding vegetation mass)
- Unusual seasonality (evergreen species in a deciduous region)
- Disturbance recovery (cleared area regrowing fast)

The 10% contamination parameter tells the algorithm to expect approximately 1 in 10
pixels to be anomalous. Increasing this flags more pixels; decreasing it is stricter.
                """)

        # --- Tab 4: Risk composite ---
        with tab4:
            st.markdown("**Combined Risk Composite — All Four Algorithms**")
            st.markdown(
                "Risk is the combination of current density AND growth direction AND "
                "anomalous behaviour. A dense pixel with zero growth is lower risk than "
                "a shrub-zone pixel with a steep positive slope."
            )

            fig_risk = build_risk_composite_fig(results)
            st.image(_fig_to_bytes(fig_risk), use_container_width=True)

            fig_bar = build_risk_breakdown_chart(results['risk_flat'], results['n_pixels'])
            st.plotly_chart(fig_bar, use_container_width=True)

            risk_rules = {
                'Critical': 'Dense vegetation (NDVI ≥ 0.60) PLUS slope > +0.02/month',
                'Warning':  'Shrub zone + growing slope  OR  dense vegetation + anomaly flag',
                'Monitor':  'Shrub zone stable  OR  grass zone with slope > +0.03/month',
                'Clear':    'Cleared ROW (NDVI < 0.10)  OR  stable low vegetation',
            }
            for cls, rule in risk_rules.items():
                color = {'Critical':'#d73027','Warning':'#fd8d3c','Monitor':'#d4a017','Clear':'#1a9850'}[cls]
                st.markdown(
                    f'<span style="color:{color};font-weight:600;">{cls}:</span> {rule}',
                    unsafe_allow_html=True
                )

        # --- Tab 5: Statistics table ---
        with tab5:
            st.markdown("**Risk class statistics**")
            rf = results['risk_flat']
            sl = results['slopes']

            rows_risk = []
            for cls_id in [4, 3, 2, 1]:
                mask  = rf == cls_id
                count = mask.sum()
                rows_risk.append({
                    'Risk level':   RISK_LABELS[cls_id],
                    'Pixels':       f"{count:,}",
                    '% Area':       f"{100.0*count/n_pixels:.1f}%",
                    'Mean slope':   f"{sl[mask].mean():+.4f}/mo" if count > 0 else "—",
                })
            st.table(pd.DataFrame(rows_risk))

            st.markdown("**K-means cluster breakdown**")
            rows_km = []
            for rank, stats in enumerate(results['cluster_stats']):
                rows_km.append({
                    'Zone':       KM_LABELS_NAMED[rank],
                    'Pixels':     f"{stats['count']:,}",
                    '% Area':     f"{100.0*stats['count']/n_pixels:.1f}%",
                    'Mean NDVI':  f"{stats['mean_ndvi']:.3f}",
                })
            st.table(pd.DataFrame(rows_km))

        # ===================================================================
        # LAYER 3: AI risk brief
        # ===================================================================
        section_break()
        st.subheader("🤖 Layer 3 — AI Risk Synthesis")
        st.caption(
            "Layer 3 receives the structured algorithm summary and produces a "
            "decision-ready inspection brief. Every claim is traceable to a Layer 2 output."
        )

        if st.button("Get AI Risk Brief", type="primary", key="cr_ai_btn"):
            with st.spinner("Synthesising risk brief..."):
                text, model = get_ai_brief(context_str)
                if text:
                    st.session_state.cr_ai_text  = text
                    st.session_state.cr_ai_model = model
                else:
                    st.session_state.cr_ai_text  = "AI call failed. Check API keys."
                    st.session_state.cr_ai_model = None

        if st.session_state.get("cr_ai_text"):
            st.markdown(st.session_state.cr_ai_text)
            model = st.session_state.get("cr_ai_model", "")
            if model:
                st.caption(f"AI response from {model}")

        # ===================================================================
        # DATA QUALITY
        # ===================================================================
        section_break()
        st.subheader("🔍 Data Quality")

        n_valid = sum(1 for a in ndvi_arrays if a.mean() > 0.01)
        conf    = "High" if n_valid == 6 else f"Moderate — only {n_valid}/6 months have valid data"

        st.info(
            f"**Sensor:** Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED)  \n"
            f"**Spatial resolution:** 100 m (resampled from native 10 m)  \n"
            f"**Time steps:** 6 monthly composites — April–September 2023  \n"
            f"**Valid composites:** {n_valid}/6  \n"
            f"**Cloud handling:** Scene-level filter (80%) + QA60 pixel-level mask  \n"
            f"**Compositing method:** Median across all valid clear-sky observations per window  \n"
            f"**Classification methods:** K-means (k=4), threshold (fixed NDVI), "
            f"linear regression, Isolation Forest (contamination=10%)  \n"
            f"**Confidence:** {conf}"
        )

    else:
        st.markdown("---")
        st.markdown(
            "**Click Run Analysis to fetch satellite data and run all four algorithms.**\n\n"
            "The module will download six monthly NDVI composites from Google Earth Engine, "
            "run K-means, threshold classification, linear regression, and Isolation Forest "
            "on the same pixel arrays, then synthesise all outputs into a risk composite and "
            "AI inspection brief."
        )
