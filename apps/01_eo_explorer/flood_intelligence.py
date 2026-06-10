"""
flood_intelligence.py — Flood Detection and Impact Mapping module for the EOIL portal.

Arc 3: Flood Detection and Impact Mapping.

Three-layer architecture:
  Layer 1: Sentinel-1 SAR backscatter change, NDWI from Sentinel-2, SRTM slope mask,
            JRC Global Surface Water permanent water mask
  Layer 2: Combined flood extent classification — flooded km², permanent water km²,
            new flood-only km², confidence level
  Layer 3: Gemini/Groq structured impact brief — five-element flood intelligence output

All algorithm logic is ported from notebook 10_flood_detection.ipynb.
Results are cached in Streamlit session state so switching tabs does not re-run.

Educational core:
  SAR (C-band Sentinel-1) penetrates cloud cover because the radar wavelength (~5.6 cm)
  is far larger than cloud water droplets. Optical sensors (Sentinel-2) detect reflected
  sunlight, which clouds block. During a flood — which always involves heavy cloud —
  SAR is the only sensor that can map the inundation as it is happening.
"""

import io
import re
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import requests
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from PIL import Image
from io import BytesIO

import folium
from streamlit_folium import st_folium

from docx import Document
from docx.shared import Inches, Pt, RGBColor

import config
import ai_chain

# ---------------------------------------------------------------------------
# Pre-defined flood events
# Each entry has: name, bbox [W, S, E, N], before/after dates, context notes
# ---------------------------------------------------------------------------

FLOOD_EVENTS = {
    "Pakistan 2022 — Indus Floodplain, Sindh": {
        "bbox":         [67.0, 25.5, 69.5, 27.5],
        "center":       [26.5, 68.2],
        "zoom":         7,
        "before_start": "2022-06-01",
        "before_end":   "2022-07-31",
        "after_start":  "2022-08-20",
        "after_end":    "2022-09-30",
        "context": (
            "One-third of Pakistan submerged. 33 million displaced, 1,739 deaths, "
            "2 million homes damaged. Lower Indus floodplain, Sindh province. "
            "Heavy monsoon cloud cover throughout the event — optical imagery largely unusable."
        ),
        "sar_threshold": -3.0,
        "orbit": "DESCENDING",
    },
    "Bangladesh 2020 — Brahmaputra, Sylhet Division": {
        "bbox":         [90.5, 24.0, 92.5, 25.5],
        "center":       [24.8, 91.5],
        "zoom":         8,
        "before_start": "2020-06-01",
        "before_end":   "2020-06-30",
        "after_start":  "2020-07-01",
        "after_end":    "2020-08-15",
        "context": (
            "Severe monsoon flooding along the Brahmaputra and Meghna river systems. "
            "Sylhet, Mymensingh and Jamalpur divisions heavily affected. "
            "Cloud cover during monsoon season makes SAR the primary detection sensor."
        ),
        "sar_threshold": -3.0,
        "orbit": "ASCENDING",
    },
    "Nigeria 2022 — Niger-Benue Confluence, Lokoja": {
        "bbox":         [6.2, 7.5, 7.8, 9.0],
        "center":       [8.2, 6.9],
        "zoom":         8,
        "before_start": "2022-08-01",
        "before_end":   "2022-09-15",
        "after_start":  "2022-09-16",
        "after_end":    "2022-11-15",
        "context": (
            "Flooding at the confluence of the Niger and Benue rivers. "
            "Over 600 deaths, 1.3 million displaced. Anambra, Kogi, and Delta states affected. "
            "West African rainy season cloud cover limits optical sensor utility."
        ),
        "sar_threshold": -3.0,
        "orbit": "DESCENDING",
    },
}

# ---------------------------------------------------------------------------
# GEE helper functions — all computation is server-side
# ---------------------------------------------------------------------------

def _init_gee():
    """Return True if GEE is available in this session."""
    return st.session_state.get("gee_available", False)


def _run_flood_analysis(event_key: str) -> dict:
    """
    Run the full four-layer flood detection algorithm for a flood event.
    Returns a dict of results including statistics, thumbnail URLs, and flags.
    All computation is server-side on GEE.
    """
    import ee

    event = FLOOD_EVENTS[event_key]
    bbox  = event["bbox"]
    aoi   = ee.Geometry.Rectangle(bbox)

    # ------------------------------------------------------------------
    # Layer 1a: Sentinel-1 SAR before/after composites
    # ------------------------------------------------------------------
    def get_sar(start, end):
        return (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.eq('orbitProperties_pass', event["orbit"]))
            .select('VV')
            .median()
            .clip(aoi)
        )

    sar_before = get_sar(event["before_start"], event["before_end"])
    sar_after  = get_sar(event["after_start"],  event["after_end"])
    sar_change = sar_after.subtract(sar_before).rename('sar_change')

    # Count scenes for confidence
    def count_sar(start, end):
        return (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.eq('orbitProperties_pass', event["orbit"]))
            .size().getInfo()
        )

    before_count = count_sar(event["before_start"], event["before_end"])
    after_count  = count_sar(event["after_start"],  event["after_end"])

    # ------------------------------------------------------------------
    # Layer 1b: Sentinel-2 NDWI (where cloud-free optical exists)
    # ------------------------------------------------------------------
    def compute_ndwi(image):
        return image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    s2_col = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(event["after_start"], event["after_end"])
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    s2_count = s2_col.size().getInfo()

    if s2_count > 0:
        ndwi_image = s2_col.map(compute_ndwi).median().clip(aoi)
        ndwi_water = ndwi_image.gt(0.2).rename('ndwi_water')
    else:
        ndwi_image = ee.Image(0).rename('NDWI').clip(aoi)
        ndwi_water = ee.Image(0).rename('ndwi_water').clip(aoi)

    # ------------------------------------------------------------------
    # Layer 1c: SRTM slope mask
    # ------------------------------------------------------------------
    dem        = ee.Image('USGS/SRTMGL1_003').clip(aoi)
    slope      = ee.Terrain.slope(dem)
    slope_mask = slope.lt(5).rename('slope_mask')

    # ------------------------------------------------------------------
    # Layer 1d: JRC permanent water mask
    # ------------------------------------------------------------------
    gsw             = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(aoi)
    permanent_water = gsw.gt(75).rename('permanent_water')

    # ------------------------------------------------------------------
    # Layer 2: Combined flood classification
    # ------------------------------------------------------------------
    threshold        = event["sar_threshold"]
    sar_candidate    = sar_change.lt(threshold)
    sar_flat         = sar_candidate.And(slope_mask)
    sar_new          = sar_flat.And(permanent_water.Not())
    confirmed_flood  = sar_new.And(ndwi_water).multiply(2)
    sar_only_flood   = sar_new.And(ndwi_water.Not()).multiply(1)
    flood_class      = confirmed_flood.add(sar_only_flood).where(permanent_water, -1)
    flood_class      = flood_class.rename('flood_class').clip(aoi)

    # ------------------------------------------------------------------
    # Area statistics (500m scale for speed)
    # ------------------------------------------------------------------
    pixel_area = ee.Image.pixelArea().divide(1e6)

    def class_area(val):
        return (
            flood_class.eq(val).multiply(pixel_area)
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=500, maxPixels=1e9)
            .getInfo().get('flood_class', 0)
        )

    area_permanent   = class_area(-1)
    area_sar_flood   = class_area(1)
    area_confirmed   = class_area(2)
    area_total_flood = area_sar_flood + area_confirmed

    # Confidence classification
    if s2_count >= 3 and area_confirmed > 0:
        confidence = 'HIGH'
        confidence_note = 'SAR change and NDWI in agreement across multiple optical scenes.'
    elif s2_count > 0:
        confidence = 'MEDIUM'
        confidence_note = 'SAR primary detector. Limited optical confirmation available.'
    else:
        confidence = 'MEDIUM'
        confidence_note = (
            f'SAR only — no cloud-free optical scenes found during the flood period '
            f'(cloud cover filter: <20%). This is consistent with monsoon/storm conditions '
            f'that typically accompany major flood events. SAR is the appropriate primary '
            f'sensor for this scenario.'
        )

    # ------------------------------------------------------------------
    # Thumbnail URLs for the portal visualisation
    # ------------------------------------------------------------------
    region_coords = aoi.getInfo()['coordinates']
    thumb_params  = {'region': region_coords, 'dimensions': '300x300', 'format': 'png'}

    sar_vis    = {'min': -25, 'max': 0,   'palette': ['000000', 'ffffff']}
    change_vis = {'min': -10, 'max': 5,   'palette': ['0000ff', 'ffffff', 'ff0000']}
    flood_vis  = {
        'min': -1, 'max': 2,
        'palette': ['1E90FF', 'f0f0f0', 'FF8C00', 'FF0000'],
    }

    try:
        url_before = sar_before.getThumbURL({**sar_vis,    **thumb_params})
        url_after  = sar_after.getThumbURL( {**sar_vis,    **thumb_params})
        url_change = sar_change.getThumbURL({**change_vis, **thumb_params})
        url_flood  = flood_class.getThumbURL({**flood_vis, **thumb_params})
    except Exception:
        url_before = url_after = url_change = url_flood = None

    # ------------------------------------------------------------------
    # Cloud cover summary for Layer 3 context
    # ------------------------------------------------------------------
    s2_any = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(event["after_start"], event["after_end"])
        .size().getInfo()
    )
    avg_cloud = 'unknown'
    if s2_any > 0:
        cloud_col = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(event["after_start"], event["after_end"])
        )
        cloud_stats = cloud_col.aggregate_stats('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        mean_cloud = cloud_stats.get('mean', None)
        if mean_cloud is not None:
            avg_cloud = f'{mean_cloud:.0f}%'

    return {
        "event_key":        event_key,
        "event":            event,
        "before_count":     before_count,
        "after_count":      after_count,
        "s2_count":         s2_count,
        "avg_cloud":        avg_cloud,
        "area_permanent":   area_permanent,
        "area_sar_flood":   area_sar_flood,
        "area_confirmed":   area_confirmed,
        "area_total_flood": area_total_flood,
        "confidence":       confidence,
        "confidence_note":  confidence_note,
        "url_before":       url_before,
        "url_after":        url_after,
        "url_change":       url_change,
        "url_flood":        url_flood,
        "sar_threshold":    event["sar_threshold"],
        "threshold_label":  f'{event["sar_threshold"]:.0f} dB',
    }


def _fetch_png(url: str) -> bytes | None:
    """Download a PNG from a GEE thumbnail URL. Returns bytes or None on failure."""
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Layer 3: Impact brief prompts and fallback
# ---------------------------------------------------------------------------

_FIVE_ELEMENTS = [
    "Flood Extent",
    "Infrastructure Exposure",
    "Sensor Justification",
    "Response Recommendation",
    "Confidence and Limitations",
]

def _build_impact_prompt(results: dict) -> str:
    """Build the five-element structured prompt for Layer 3."""
    e       = results["event"]
    context = e["context"]
    r       = results

    return f"""You are an Earth observation analyst producing a structured flood impact brief for an emergency coordinator.

EVENT: {r['event_key']}
CONTEXT: {context}

LAYER 2 OUTPUTS (computed from Sentinel-1 SAR and supporting data):
- Permanent water baseline: {r['area_permanent']:.0f} km²
- SAR-detected new flood area: {r['area_sar_flood']:.0f} km²
- Optically confirmed flood area: {r['area_confirmed']:.0f} km²
- Total new flood area: {r['area_total_flood']:.0f} km²
- Confidence level: {r['confidence']}
- Confidence note: {r['confidence_note']}
- SAR scenes used (before period): {r['before_count']}
- SAR scenes used (after period): {r['after_count']}
- Cloud-free optical scenes available: {r['s2_count']}
- Average cloud cover over scene: {r['avg_cloud']}
- SAR backscatter change threshold applied: {r['threshold_label']}

Write a structured five-element impact brief. Use exactly these five headings. Be specific and quantitative.

## 1. Flood Extent
State the total new flood area in km². Distinguish permanent water from new inundation. State the total affected area in absolute terms. Do not use vague phrases — use the numbers provided.

## 2. Infrastructure Exposure
Based on the region, flood extent, and known land use, identify the infrastructure categories most likely present in the affected zone. Typical categories for this region: roads (national and rural), irrigation canals and water infrastructure, agricultural land (crop type if known), settlements and residential areas, power distribution infrastructure. Estimate exposure qualitatively — high, medium, or low — for each category, with a one-sentence justification.

## 3. Sensor Justification
Explain why SAR was the appropriate sensor for this specific event. Reference the cloud cover data provided. Explain the physical mechanism: C-band wavelength (~5.6 cm) vs cloud droplet size. Explain the backscatter change principle: why flooded land looks different from dry land to radar. Make this educational — a coordinator who does not know SAR should understand why this map exists and why optical imagery would have failed.

## 4. Response Recommendation
State what a field team or emergency coordinator should do with this output, and in what order. Be specific: which areas to prioritise for reconnaissance, what the SAR flood boundary implies for access route planning, how to use the permanent water baseline to distinguish new vs existing water in the field. Include at least one specific verification step (what to check on the ground that the satellite cannot confirm).

## 5. Confidence and Limitations
State clearly what this analysis cannot confirm without ground truth. Address: the SAR backscatter change threshold and its sensitivity, what may be overcounted (flooded vegetation, roughness changes from wind on floodwater), what may be undercounted (urban flood in built-up areas where SAR geometry is complex, shallow flooding below SAR detection threshold), and the spatial resolution limitation at 500m analysis scale vs actual field conditions.

End with a one-line data quality summary in this format:
DATA QUALITY: SAR scenes (before/after): {r['before_count']}/{r['after_count']} | Optical scenes: {r['s2_count']} | Cloud cover: {r['avg_cloud']} | Confidence: {r['confidence']}
"""


def _fallback_brief(results: dict) -> str:
    """Substantive fallback text used when no AI key is available."""
    e = results["event"]
    r = results
    return f"""## 1. Flood Extent

Total new flood area detected: **{r['area_total_flood']:.0f} km²** across the {r['event_key'].split('—')[1].strip()} study region. Permanent water baseline (rivers, canals, seasonal lakes): {r['area_permanent']:.0f} km². SAR-detected new inundation: {r['area_sar_flood']:.0f} km². Optically confirmed new water: {r['area_confirmed']:.0f} km². New flood water represents the inundation that did not exist in the pre-event baseline period.

## 2. Infrastructure Exposure

Based on regional land use context for this area:

- **Roads and transport:** HIGH exposure. Rural road networks in floodplain regions typically follow irrigation canal alignments. Inundation of {r['area_total_flood']:.0f} km² implies significant disruption to surface access.
- **Agricultural land:** HIGH exposure. Floodplain regions in this geography are primarily agricultural. Crop damage and soil erosion are the primary near-term impacts.
- **Settlements:** MEDIUM-HIGH exposure. Rural settlements in floodplain zones are vulnerable. Urban areas may have partial flood defences.
- **Power distribution:** MEDIUM exposure. Distribution lines typically follow road corridors. Substation flooding is a secondary risk requiring field verification.
- **Water infrastructure:** HIGH exposure. Irrigation canals may overflow, silting during recession. Drinking water infrastructure is a priority for field assessment.

## 3. Sensor Justification

Sentinel-1 SAR operates at C-band (5.4 GHz, ~5.6 cm wavelength). Cloud water droplets are typically 10–100 micrometres in diameter — more than 500 times smaller than the radar wavelength. The radar pulse passes through cloud without scattering. Optical sensors (Sentinel-2) detect reflected sunlight at wavelengths of 400–2400 nm. Cloud water absorbs and scatters this energy completely. During this event, average cloud cover was approximately {r['avg_cloud']} over the scene. {r['s2_count']} cloud-free optical scenes were available — {'none, making SAR the sole viable detection instrument' if r['s2_count'] == 0 else 'insufficient for complete coverage'}.

The flood detection signal in SAR is backscatter change. Dry land returns a strong VV signal because rough vegetation and soil scatter radar in multiple directions back toward the sensor. Flooded land returns a weak signal because smooth water reflects the radar pulse away from the sensor (specular reflection). A threshold of {r['threshold_label']} change was applied — this represents a halving of radar return power, a physically meaningful and widely validated flood detection criterion.

## 4. Response Recommendation

**Priority order for field teams:**

1. Confirm flood boundary at two to three accessible points along the SAR-detected flood edge. The satellite boundary is accurate to approximately 500m — field verification refines this for operational use.
2. Use the permanent water layer to distinguish new water from the pre-existing river and canal system. Do not route vehicles into areas flagged as new flood that have no historical water signature.
3. Prioritise reconnaissance of settlement areas within the flood boundary. The SAR analysis identifies water extent but cannot determine building occupancy or damage level.
4. Check road corridors at flood boundary crossings before committing vehicles. SAR flood mapping typically detects water above a depth threshold but does not measure depth.
5. Re-run analysis after 7–10 days using the next available SAR pass to assess whether flood extent is growing, stable, or receding.

## 5. Confidence and Limitations

**What this analysis cannot confirm without ground truth:**

- Flood depth is not measurable from SAR backscatter change alone. The analysis detects surface water extent, not depth.
- Urban flooding is systematically undercounted. SAR backscatter in built-up areas is dominated by building corner reflectors; shallow urban flooding does not produce the same backscatter drop as open floodplain flooding.
- Flooded vegetation (crops, forest) produces complex backscatter changes that can either mask or mimic flood signal depending on vegetation type and flood depth. The slope mask and permanent water mask reduce but do not eliminate this uncertainty.
- The {r['threshold_label']} threshold is conservative. Reducing to -2 dB would increase the detected area and increase false positive rate. Increasing to -5 dB would decrease detected area and reduce sensitivity to shallow flooding.
- Analysis resolution is 500m for statistics. SAR native resolution is ~10m. At 500m scale, mixed pixels at flood boundaries undercount total extent.

DATA QUALITY: SAR scenes (before/after): {r['before_count']}/{r['after_count']} | Optical scenes: {r['s2_count']} | Cloud cover: {r['avg_cloud']} | Confidence: {r['confidence']}
"""


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_extent_chart(results: dict) -> bytes:
    """Bar chart: permanent water, SAR-only flood, confirmed flood areas."""
    r = results
    categories = ['Permanent\nWater', 'SAR-Only\nFlood', 'Confirmed\nFlood\n(SAR+NDWI)']
    values     = [r['area_permanent'], r['area_sar_flood'], r['area_confirmed']]
    colors     = ['#1E90FF', '#FF8C00', '#CC2200']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)

    max_val = max(values) if max(values) > 0 else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.015,
            f'{val:,.0f} km²',
            ha='center', va='bottom', fontweight='bold', fontsize=10
        )

    ax.set_ylabel('Area (km²)', fontsize=10)
    ax.set_title(
        f"Flood Extent Classification\n"
        f"Total new flood: {r['area_total_flood']:,.0f} km²  |  Confidence: {r['confidence']}",
        fontsize=11, fontweight='bold'
    )
    ax.set_facecolor('#f8f8f8')
    fig.patch.set_facecolor('white')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_sar_thumbnails_fig(results: dict) -> bytes:
    """Three-panel SAR before / after / change figure."""
    panels = [
        (results['url_before'], f"SAR Before\n({results['event']['before_start'][:7]} – {results['event']['before_end'][:7]})\nVV backscatter median, {results['before_count']} scenes"),
        (results['url_after'],  f"SAR After\n({results['event']['after_start'][:7]} – {results['event']['after_end'][:7]})\nVV backscatter median, {results['after_count']} scenes"),
        (results['url_change'], f"Backscatter Change\nAfter minus Before (dB)\nBlue=increase  Red=decrease"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Layer 1 — Sentinel-1 SAR Analysis\n{results['event_key']}",
        fontsize=12, fontweight='bold'
    )

    for ax, (url, caption) in zip(axes, panels):
        raw = _fetch_png(url)
        if raw:
            try:
                img = Image.open(BytesIO(raw))
                ax.imshow(img)
            except Exception:
                ax.set_facecolor('#cccccc')
                ax.text(0.5, 0.5, 'Image\nunavailable', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
        else:
            ax.set_facecolor('#cccccc')
            ax.text(0.5, 0.5, 'Image\nunavailable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
        ax.set_title(caption, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Word document builder
# ---------------------------------------------------------------------------

def _build_word_doc(results: dict, brief_text: str, model_used: str) -> bytes:
    """Build a Word document with the flood intelligence brief and stats tables."""
    doc  = Document()
    r    = results

    # ---- Title ----
    title = doc.add_heading('Flood Intelligence Brief', level=0)
    title.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)

    doc.add_paragraph(r['event_key'])
    doc.add_paragraph(f"Generated by EOIL Portal — Arc 3 Flood Detection")
    doc.add_paragraph(r['event']['context'])
    doc.add_paragraph('')

    # ---- Layer 2 statistics table ----
    doc.add_heading('Layer 2 — Flood Extent Statistics', level=1)
    table = doc.add_table(rows=7, cols=2)
    table.style = 'Light Grid Accent 1'
    rows = [
        ('Metric',                  'Value'),
        ('Permanent water baseline', f"{r['area_permanent']:,.0f} km²"),
        ('SAR-only new flood',       f"{r['area_sar_flood']:,.0f} km²"),
        ('Confirmed flood (SAR+NDWI)', f"{r['area_confirmed']:,.0f} km²"),
        ('Total new flood area',     f"{r['area_total_flood']:,.0f} km²"),
        ('Confidence level',         r['confidence']),
        ('Confidence note',          r['confidence_note']),
    ]
    for i, (label, value) in enumerate(rows):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = value
        if i == 0:
            for cell in table.rows[i].cells:
                for run in cell.paragraphs[0].runs:
                    run.font.bold = True
    doc.add_paragraph('')

    # ---- Data quality table ----
    doc.add_heading('Data Quality', level=1)
    q_table = doc.add_table(rows=5, cols=2)
    q_table.style = 'Light Grid Accent 1'
    q_rows = [
        ('Parameter',                 'Value'),
        ('SAR scenes — before period', str(r['before_count'])),
        ('SAR scenes — after period',  str(r['after_count'])),
        ('Cloud-free optical scenes',  str(r['s2_count'])),
        ('Average cloud cover',        r['avg_cloud']),
    ]
    for i, (label, value) in enumerate(q_rows):
        q_table.rows[i].cells[0].text = label
        q_table.rows[i].cells[1].text = value
        if i == 0:
            for cell in q_table.rows[i].cells:
                for run in cell.paragraphs[0].runs:
                    run.font.bold = True
    doc.add_paragraph('')

    # ---- Thumbnails ----
    doc.add_heading('Layer 1 — SAR Imagery', level=1)
    for url, caption in [
        (r['url_before'], 'SAR Before (VV backscatter)'),
        (r['url_after'],  'SAR After (VV backscatter)'),
        (r['url_change'], 'SAR Backscatter Change'),
        (r['url_flood'],  'Flood Classification Map'),
    ]:
        raw = _fetch_png(url)
        if raw:
            try:
                img_buf = io.BytesIO(raw)
                doc.add_picture(img_buf, width=Inches(3.0))
                doc.paragraphs[-1].runs[0].font.size = Pt(9)
                doc.add_paragraph(caption).runs[0].font.size = Pt(9) if doc.paragraphs[-1].runs else None
            except Exception:
                doc.add_paragraph(f'[{caption} — image unavailable]')

    doc.add_paragraph('')

    # ---- Impact brief ----
    doc.add_heading('Layer 3 — AI Impact Brief', level=1)
    if model_used:
        doc.add_paragraph(f'Generated by: {model_used}').runs[0].font.italic = True

    # Strip markdown and add structured paragraphs
    for line in brief_text.splitlines():
        line = line.strip()
        if not line:
            doc.add_paragraph('')
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
        elif line.startswith('**') and line.endswith('**'):
            p = doc.add_paragraph()
            run = p.add_run(line.strip('*'))
            run.bold = True
        elif line.startswith('- '):
            doc.add_paragraph(line[2:], style='List Bullet')
        elif line.startswith('DATA QUALITY:'):
            p = doc.add_paragraph(line)
            for run in p.runs:
                run.font.bold = True
        else:
            doc.add_paragraph(line)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Markdown download builder
# ---------------------------------------------------------------------------

def _build_markdown(results: dict, brief_text: str, model_used: str) -> str:
    r = results
    lines = [
        f"# Flood Intelligence Brief",
        f"",
        f"**Event:** {r['event_key']}",
        f"**Context:** {r['event']['context']}",
        f"",
        f"---",
        f"",
        f"## Layer 2 — Flood Extent Statistics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Permanent water baseline | {r['area_permanent']:,.0f} km² |",
        f"| SAR-only new flood | {r['area_sar_flood']:,.0f} km² |",
        f"| Confirmed flood (SAR+NDWI) | {r['area_confirmed']:,.0f} km² |",
        f"| Total new flood area | {r['area_total_flood']:,.0f} km² |",
        f"| Confidence level | {r['confidence']} |",
        f"",
        f"**Confidence note:** {r['confidence_note']}",
        f"",
        f"---",
        f"",
        f"## Data Quality",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| SAR scenes — before period | {r['before_count']} |",
        f"| SAR scenes — after period | {r['after_count']} |",
        f"| Cloud-free optical scenes | {r['s2_count']} |",
        f"| Average cloud cover | {r['avg_cloud']} |",
        f"",
        f"---",
        f"",
        f"## Layer 3 — AI Impact Brief",
        f"",
        f"*Generated by: {model_used or 'built-in fallback'}*",
        f"",
        brief_text,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main render function — called from app.py
# ---------------------------------------------------------------------------

def render():
    """Render the Flood Intelligence module. Called by app.py."""

    st.markdown("## 🌊 Flood Intelligence")
    st.markdown(
        "SAR-based flood detection and impact mapping. "
        "Sentinel-1 radar penetrates cloud cover — making it the only viable sensor "
        "during a flood event, when cloud cover is heaviest."
    )

    gee_ok = _init_gee()

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Flood Intelligence")
        st.divider()

        event_key = st.selectbox(
            "Select flood event",
            list(FLOOD_EVENTS.keys()),
            help="Each event has validated Sentinel-1 coverage and known impact context."
        )

        event = FLOOD_EVENTS[event_key]
        st.markdown(f"**Before period:** {event['before_start']} to {event['before_end']}")
        st.markdown(f"**After period:** {event['after_start']} to {event['after_end']}")

        st.divider()
        st.markdown("**SAR threshold**")
        sar_threshold = st.slider(
            "Backscatter change (dB)",
            min_value=-8.0, max_value=-1.0, value=float(event["sar_threshold"]),
            step=0.5,
            help="Pixels with backscatter change below this threshold are classified as flood. More negative = stricter filter."
        )
        st.caption(f"−3 dB = backscatter halved. Typical flood signal: −3 to −6 dB.")

        run_btn = st.button("Run flood analysis", type="primary", use_container_width=True)

    # ------------------------------------------------------------------
    # Context card — always visible
    # ------------------------------------------------------------------
    with st.expander("Event context", expanded=True):
        st.markdown(f"**{event_key}**")
        st.markdown(event["context"])

    st.divider()

    # ------------------------------------------------------------------
    # Cache key — rerun if event or threshold changes
    # ------------------------------------------------------------------
    cache_key = f"flood_results_{event_key}_{sar_threshold}"

    # Two-step run pattern (matches corridor_risk.py)
    if run_btn:
        st.session_state["flood_pending"] = cache_key
        st.rerun()

    if st.session_state.get("flood_pending") == cache_key:
        del st.session_state["flood_pending"]

        if not gee_ok:
            st.error("GEE not available. Cannot run flood analysis without Earth Engine credentials.")
            st.stop()

        with st.spinner("Running flood detection — fetching SAR composites, computing change, classifying flood extent..."):
            try:
                # Temporarily override threshold from slider
                import ee
                event_copy = dict(event)
                event_copy["sar_threshold"] = sar_threshold
                FLOOD_EVENTS[event_key]["sar_threshold"] = sar_threshold

                results = _run_flood_analysis(event_key)
                st.session_state[cache_key] = results
            except Exception as ex:
                st.error(f"Flood analysis failed: {ex}")
                st.stop()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    if cache_key not in st.session_state:
        st.info("Select an event and click **Run flood analysis** to begin.")

        # Educational panel — always shown before first run
        with st.expander("Why SAR for flood detection?", expanded=True):
            st.markdown("""
**The cloud problem.**

Every major flood event is accompanied by heavy cloud cover — the same weather system
that causes flooding also blocks optical satellites from seeing the ground.
Sentinel-2, Landsat, and all cameras in orbit that detect reflected sunlight
cannot see through cloud. When a flood is at its worst, they produce nothing useful.

**Why SAR is different.**

Sentinel-1 is a Synthetic Aperture Radar. It transmits its own microwave energy
at C-band frequency (5.4 GHz, ~5.6 cm wavelength) and measures what comes back.
Cloud water droplets are 10–100 micrometres in diameter — more than 500 times
smaller than the radar wavelength. The radar pulse passes through cloud without
scattering. Cloud cover is invisible to Sentinel-1.

**The flood signal.**

Dry land returns a strong radar signal. Rough surfaces (vegetation, bare soil,
buildings) scatter the radar pulse back toward the sensor from many angles.
Flooded land returns a very weak signal. Smooth water reflects the radar pulse
away from the sensor like a mirror — almost nothing comes back.
The change in backscatter between a before scene and an after scene is the
flood detection signal. A drop of −3 dB means the returned power halved.

**Three supporting layers.**

SAR alone detects all smooth water — rivers, lakes, canals, and flood water.
Three additional layers are applied to isolate new inundation:
1. **Slope mask** — flat terrain only. Steep slopes drain too fast to pond.
2. **Permanent water mask** — remove existing rivers and water bodies.
3. **NDWI from optical** — secondary confirmation where cloud-free scenes exist.
            """)
        return

    results    = st.session_state[cache_key]
    r          = results

    # ------------------------------------------------------------------
    # Layer 1: Data display
    # ------------------------------------------------------------------
    st.markdown("### Layer 1 — Signal Processing")
    st.markdown(
        "Four signals combined: SAR backscatter change, NDWI optical water index, "
        "SRTM slope mask, JRC permanent water mask."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SAR scenes (before)", r["before_count"])
    with col2:
        st.metric("SAR scenes (after)", r["after_count"])
    with col3:
        st.metric("Optical scenes (<20% cloud)", r["s2_count"])
    with col4:
        st.metric("Avg cloud cover", r["avg_cloud"])

    # SAR thumbnails
    sar_fig_bytes = _build_sar_thumbnails_fig(results)
    st.image(sar_fig_bytes, use_column_width=True)

    # Flood map thumbnail
    if r["url_flood"]:
        flood_raw = _fetch_png(r["url_flood"])
        if flood_raw:
            col_map, col_legend = st.columns([2, 1])
            with col_map:
                st.image(flood_raw, caption="Flood classification map", use_column_width=True)
            with col_legend:
                st.markdown("""
**Map legend**

🟦 Permanent water (baseline)
🟧 SAR-only flood (new)
🟥 Confirmed flood (SAR + NDWI)
⬜ Not flooded
                """)

    st.divider()

    # ------------------------------------------------------------------
    # Layer 2: Algorithm output
    # ------------------------------------------------------------------
    st.markdown("### Layer 2 — Flood Extent Classification")
    st.markdown("Combined four-signal analysis. New flood area only — permanent water excluded.")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Permanent water", f"{r['area_permanent']:,.0f} km²")
    with col_b:
        st.metric("SAR-only flood", f"{r['area_sar_flood']:,.0f} km²")
    with col_c:
        st.metric("Confirmed flood", f"{r['area_confirmed']:,.0f} km²")
    with col_d:
        st.metric("Total new flood", f"{r['area_total_flood']:,.0f} km²",
                  delta=r["confidence"], delta_color="off")

    # Confidence block
    conf_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(r["confidence"], "⚪")
    st.info(f"{conf_color} **Confidence: {r['confidence']}** — {r['confidence_note']}")

    # Area chart
    chart_bytes = _build_extent_chart(results)
    st.image(chart_bytes, use_column_width=False, width=500)

    st.divider()

    # ------------------------------------------------------------------
    # Layer 3: AI impact brief
    # ------------------------------------------------------------------
    st.markdown("### Layer 3 — AI Impact Brief")
    st.markdown(
        "Five-element structured brief for emergency coordinators and field teams. "
        "Every claim is grounded in the Layer 2 measurements above."
    )

    # Generate brief if not already cached
    brief_key  = f"flood_brief_{cache_key}"
    model_key  = f"flood_model_{cache_key}"

    if brief_key not in st.session_state:
        prompt = _build_impact_prompt(results)
        text, model_used = ai_chain.complete(
            prompt,
            groq_key=config.GROQ_API_KEY,
            gemini_key=config.GEMINI_API_KEY
        )
        if text:
            st.session_state[brief_key] = text
            st.session_state[model_key] = model_used
        else:
            st.session_state[brief_key] = _fallback_brief(results)
            st.session_state[model_key] = None

    brief_text = st.session_state[brief_key]
    model_used = st.session_state.get(model_key)

    with st.expander("Impact brief — expand to read", expanded=True):
        st.markdown(brief_text)
        if model_used:
            st.caption(f"AI response from {model_used}")
        else:
            st.caption("Built-in analysis — no AI key present")

    st.divider()

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------
    st.markdown("### Downloads")

    col_md, col_doc = st.columns(2)

    with col_md:
        md_content = _build_markdown(results, brief_text, model_used or "built-in fallback")
        st.download_button(
            label="Download brief (.md)",
            data=md_content.encode('utf-8'),
            file_name=f"flood_intelligence_{event_key[:20].replace(' ', '_').replace(',', '')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_doc:
        word_bytes = _build_word_doc(results, brief_text, model_used or "built-in fallback")
        st.download_button(
            label="Download brief (.docx)",
            data=word_bytes,
            file_name=f"flood_intelligence_{event_key[:20].replace(' ', '_').replace(',', '')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # Educational expander
    # ------------------------------------------------------------------
    with st.expander("What did each algorithm step do?", expanded=False):
        st.markdown(f"""
**SAR backscatter change — threshold: {r['sar_threshold']:.0f} dB**

Pixels where VV backscatter dropped by {abs(r['sar_threshold']):.0f} dB or more between the before and
after composites are flagged as flood candidates. A {abs(r['sar_threshold']):.0f} dB drop represents a
{"halving" if r['sar_threshold'] == -3 else "major reduction"} of returned radar power — a strong
signal of surface change from rough dry land to smooth water. The SAR before composite
used {r['before_count']} scenes (median). The after composite used {r['after_count']} scenes (median).
Using a median composite reduces speckle noise and removes transient anomalies.

**Slope mask — SRTM DEM, threshold: 5 degrees**

Flood water ponds on flat terrain. Slopes above 5 degrees drain too quickly for surface
ponding to persist. The SRTM DEM at 30m resolution was used to compute slope across
the scene. Pixels on terrain steeper than 5 degrees were excluded from flood classification.
This also removes SAR geometry artefacts: steep slopes produce layover and shadow in SAR
imagery that can create false backscatter drops unrelated to water.

**Permanent water mask — JRC Global Surface Water, threshold: 75% occurrence**

The JRC dataset records how often each pixel was observed as water between 1984 and 2021
using Landsat imagery. Pixels flagged as water in more than 75% of valid observations are
classified as permanent water — rivers, lakes, canals, reservoirs. These {r['area_permanent']:,.0f} km²
of baseline water were excluded from the new flood count. Without this mask, the Indus River
and the irrigation canal network would dominate the flood extent statistics.

**NDWI optical confirmation — threshold: 0.2**

Where cloud-free Sentinel-2 scenes were available ({r['s2_count']} scene{"s" if r['s2_count'] != 1 else ""}),
NDWI (Normalised Difference Water Index = (Green − NIR) / (Green + NIR)) was computed.
Pixels with NDWI > 0.2 confirm standing water from an independent sensor.
{"No cloud-free optical scenes were available during the flood period — SAR was the sole detection instrument, as expected for a monsoon-season event." if r['s2_count'] == 0 else f"Where SAR and NDWI agreed, pixels were classified as Confirmed Flood ({r['area_confirmed']:,.0f} km²). Where only SAR detected the signal, pixels were classified as SAR-Only Flood ({r['area_sar_flood']:,.0f} km²)."}
        """)
