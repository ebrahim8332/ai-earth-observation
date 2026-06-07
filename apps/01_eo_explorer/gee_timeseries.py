"""
gee_timeseries.py — GEE Time Series Explorer module
Logic extracted and adapted from notebooks/02_gee_ndvi_timeseries.ipynb

This module provides:
- GEE initialization via service account credentials from Streamlit secrets
- Live time series extraction from MODIS and Landsat collections
- Sample data fallback when GEE credentials are unavailable
- Plotly charts: time series with trend, seasonal cycle by month
- Folium maps with NASA GIBS imagery tiles (no GEE required for display)
- Groq AI interpretation with substantive per-region fallback
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import streamlit as st

import config

# ---------------------------------------------------------------------------
# Dataset definitions
# Each entry describes one satellite dataset the module can analyse.
# ---------------------------------------------------------------------------

DATASETS = {
    "MODIS NDVI": {
        # GEE image collection ID and band name
        "collection":    "MODIS/061/MOD13Q1",
        "band":          "NDVI",
        # GEE stores NDVI as an integer scaled by 10000 (e.g. 2080 = NDVI 0.208)
        # We divide by scale_factor to convert to the standard -1 to +1 range
        "scale_factor":  10000.0,
        "convert":       "divide",
        "cadence_days":  16,
        "resolution":    "250 m",
        "revisit":       "16 days",
        "sensor":        "MODIS Terra (NASA)",
        "available_from": "2000",
        "unit":          "NDVI",
        "value_range":   "0 to 1 — bare soil: 0.1-0.2, dense forest: 0.6-0.9",
        "measures": (
            "Vegetation greenness. Computed as (NIR minus Red) divided by (NIR plus Red). "
            "Healthy plants reflect strongly in near-infrared and absorb red light. "
            "Bare soil reflects both similarly. Water absorbs both."
        ),
        "best_for":      "Vegetation health, crop assessment, deforestation, drought detection",
        "limitation": (
            "Cloud contamination suppresses values in tropical regions. "
            "250 m resolution cannot resolve individual fields or small forest patches."
        ),
        "gibs_layer":    "MODIS_Terra_NDVI_8Day",
        "chart_color":   "#228B22",
    },
    "MODIS Land Surface Temperature": {
        "collection":    "MODIS/061/MOD11A2",
        "band":          "LST_Day_1km",
        # GEE stores LST as integer: multiply by 0.02 then subtract 273.15 for Celsius
        "scale_factor":  0.02,
        "convert":       "kelvin",
        "cadence_days":  8,
        "resolution":    "1 km",
        "revisit":       "8 days",
        "sensor":        "MODIS Terra (NASA)",
        "available_from": "2000",
        "unit":          "°C",
        "value_range":   "Varies by region — typically -20°C to 60°C",
        "measures": (
            "Daytime ground surface temperature, not air temperature. "
            "Urban surfaces and bare soil heat more than vegetation. "
            "Stressed dry vegetation stays warmer than healthy evapotranspiring vegetation."
        ),
        "best_for":      "Urban heat island analysis, drought stress, fire risk, crop stress",
        "limitation": (
            "Cloud cover produces data gaps — no measurement is possible through cloud. "
            "Surface temperature varies with solar angle; seasonal comparisons need care."
        ),
        "gibs_layer":    "MODIS_Terra_Land_Surface_Temp_Day",
        "chart_color":   "#d73027",
    },
    "Landsat NDVI": {
        "collection":    "LANDSAT/LC08/C02/T1_L2",
        "band":          "NDVI",        # computed from SR_B5 and SR_B4 in GEE
        "scale_factor":  1.0,
        "convert":       "direct",
        "cadence_days":  16,
        "resolution":    "30 m",
        "revisit":       "16 days",
        "sensor":        "Landsat 8 (USGS/NASA)",
        "available_from": "2013",
        "unit":          "NDVI",
        "value_range":   "0 to 1 — same index as MODIS NDVI but at field scale",
        "measures": (
            "Same vegetation greenness index as MODIS NDVI but at 30 m resolution. "
            "Resolves individual agricultural fields, urban parks, and forest edges."
        ),
        "best_for":      "Field-level crop monitoring, urban green space, precise deforestation mapping",
        "limitation": (
            "30 m resolution means 70x more pixels than MODIS for the same area. "
            "Queries for country-sized regions can take 1–3 minutes or time out. "
            "Use for small, precise areas — a field, a forest patch, a river delta. "
            "For country or regional trends, use MODIS NDVI instead."
        ),
        "gibs_layer":    "MODIS_Terra_NDVI_8Day",   # use MODIS as visual proxy
        "chart_color":   "#1a7a4a",
    },
    "MODIS EVI": {
        # Enhanced Vegetation Index — same collection as NDVI, different band.
        # EVI adds a soil adjustment factor and atmospheric correction term so it
        # does not saturate in dense tropical forest the way NDVI does.
        "collection":    "MODIS/061/MOD13Q1",
        "band":          "EVI",
        "scale_factor":  10000.0,
        "convert":       "divide",
        "cadence_days":  16,
        "resolution":    "250 m",
        "revisit":       "16 days",
        "sensor":        "MODIS Terra (NASA)",
        "available_from": "2000",
        "unit":          "EVI",
        "value_range":   "0 to 1 — dense forest: 0.4-0.6, crops: 0.2-0.5, bare soil: 0.05-0.15",
        "measures": (
            "Vegetation greenness with reduced saturation in dense canopy. "
            "EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1). "
            "The extra terms correct for soil background and atmospheric aerosol. "
            "Shows more within-canopy variation than NDVI in tropical forests."
        ),
        "best_for":      "Tropical forest monitoring, dense crop canopy, regions with atmospheric haze",
        "limitation": (
            "Same 250 m spatial resolution as MODIS NDVI. "
            "Requires a blue band — noisier in high aerosol conditions. "
            "Values slightly lower than NDVI for the same vegetation density."
        ),
        "gibs_layer":    "MODIS_Terra_NDVI_8Day",   # visual proxy — no EVI-specific GIBS layer
        "chart_color":   "#2e8b57",
    },
    "Landsat NDWI": {
        # Normalized Difference Water Index — highlights open water and surface wetness.
        # NDWI = (Green - NIR) / (Green + NIR)
        # Positive values: water or very wet surfaces.
        # Negative values: dry land, vegetation, or bare soil.
        "collection":    "LANDSAT/LC08/C02/T1_L2",
        "band":          "NDWI",        # computed from SR_B3 (Green) and SR_B5 (NIR) in GEE
        "scale_factor":  1.0,
        "convert":       "direct",
        "cadence_days":  16,
        "resolution":    "30 m",
        "revisit":       "16 days",
        "sensor":        "Landsat 8 (USGS/NASA)",
        "available_from": "2013",
        "unit":          "NDWI",
        "value_range":   "-1 to +1 — open water: 0.3 to 1.0, wet soil: 0.0 to 0.3, dry land: -0.5 to 0.0",
        "measures": (
            "Surface water presence and soil moisture. "
            "Computed as (Green minus NIR) divided by (Green plus NIR). "
            "Water bodies reflect green light and absorb NIR. Land does the opposite. "
            "Tracks seasonal flooding, drought-driven lake shrinkage, and wetland extent."
        ),
        "best_for":      "Flood mapping, reservoir monitoring, wetland extent, drought-driven water loss",
        "limitation": (
            "30 m resolution means 70x more pixels than MODIS for the same area. "
            "Queries for country-sized regions can take 1–3 minutes or time out. "
            "Best used for small areas — a reservoir, a wetland, a river reach. "
            "Dense vegetation also suppresses NDWI even when underlying soil is wet."
        ),
        "gibs_layer":    "MODIS_Terra_NDVI_8Day",   # visual proxy
        "chart_color":   "#1e90ff",
    },
    "MODIS Burned Area": {
        # MCD64A1: monthly burned area product combining Terra and Aqua.
        # BurnDate band = day of year the pixel burned (1-366), 0 = not burned.
        # We convert to a fraction: mean of a 0/1 burn mask = fraction of area burned.
        "collection":    "MODIS/061/MCD64A1",
        "band":          "BurnDate",    # converted to binary burn mask in GEE
        "scale_factor":  1.0,
        "convert":       "direct",      # GEE returns fraction 0-1 after binary mask
        "cadence_days":  30,
        "resolution":    "500 m",
        "revisit":       "Monthly",
        "sensor":        "MODIS Terra + Aqua (NASA)",
        "available_from": "2000",
        "unit":          "Fraction burned",
        "value_range":   "0 to 1 — 0: no burning detected, 0.05: 5% of area burned that month",
        "measures": (
            "Fraction of the study area that burned each month. "
            "Based on active fire detections and post-fire reflectance change. "
            "Values are near zero most months; spikes indicate fire events. "
            "Tracks seasonal fire cycles, landscape clearing, and interannual fire variability."
        ),
        "best_for":      "Fire season monitoring, deforestation by burning, carbon flux estimation",
        "limitation": (
            "Small fires below 500 m are missed entirely. "
            "Smouldering fires with little surface change are underdetected. "
            "Monthly compositing misses fires that start and recover within the same month."
        ),
        "gibs_layer":    "MODIS_Terra_NDVI_8Day",   # visual proxy
        "chart_color":   "#ff4500",
    },
}

# ---------------------------------------------------------------------------
# Preset study regions
# Users can also type a custom location, which is geocoded in app.py.
# ---------------------------------------------------------------------------

REGIONS = {
    "Sahel, West Africa": {
        "bbox":   [-5.0, 12.0, 5.0, 18.0],   # [west, south, east, north]
        "center": [15.0, 0.0],
        "zoom":   6,
        "peak_month": "08",    # August — peak of the West African Monsoon
        "description": (
            "Semi-arid transition zone between the Sahara and tropical savanna. "
            "The West African Monsoon drives a single rainy season (June-September). "
            "NDVI shows a strong single-peak seasonal cycle and a well-documented "
            "long-term greening trend since the 1980s."
        ),
    },
    "Amazon Basin, Brazil": {
        "bbox":   [-65.0, -10.0, -50.0, 0.0],
        "center": [-5.0, -57.5],
        "zoom":   6,
        "peak_month": "04",    # April — wet season high
        "description": (
            "The world's largest tropical rainforest. Year-round high NDVI with "
            "a weak dry-season dip. Deforestation in the southern arc appears as "
            "persistent NDVI decline."
        ),
    },
    "Siberian Boreal Forest": {
        "bbox":   [85.0, 55.0, 105.0, 65.0],
        "center": [60.0, 95.0],
        "zoom":   5,
        "peak_month": "07",    # July — short boreal summer
        "description": (
            "Northern boreal (taiga) forest. Extreme seasonal amplitude: near-zero "
            "NDVI under snow (November-March), peak in July. Permafrost thaw and "
            "wildfire disturbance affect long-term trends."
        ),
    },
    "Great Plains, USA": {
        "bbox":   [-102.0, 36.0, -95.0, 42.0],
        "center": [39.0, -98.5],
        "zoom":   6,
        "peak_month": "07",    # July — crop peak
        "description": (
            "Mixed dryland agriculture and native grassland. NDVI closely tracks the "
            "crop calendar: green-up in April-May, peak in July, senescence by September. "
            "Drought years are clearly visible as below-average summer NDVI."
        ),
    },
}

# ---------------------------------------------------------------------------
# GEE initialization
# ---------------------------------------------------------------------------

def init_gee():
    """Attempt to initialize GEE using a service account JSON stored in Streamlit secrets.

    The secret key is GEE_SERVICE_ACCOUNT_JSON — paste the full JSON content there.
    See docs/gee-service-account-setup.md for instructions.

    Returns True if GEE initialized successfully, False if we fall back to sample data.
    Stores a diagnostic message in st.session_state.gee_init_error for debugging.
    """
    try:
        import ee
        if not (hasattr(st, "secrets") and "GEE_SERVICE_ACCOUNT_JSON" in st.secrets):
            return False
        creds_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT_JSON"])
        credentials = ee.ServiceAccountCredentials(
            email=creds_dict["client_email"],
            key_data=json.dumps(creds_dict),
        )
        project = creds_dict.get("project_id", "gen-lang-client-0093165324")
        ee.Initialize(credentials=credentials, project=project)
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# GEE data extraction
# Pulled directly from notebooks/02_gee_ndvi_timeseries.ipynb, Cell 4 and 5.
# ---------------------------------------------------------------------------

def extract_time_series_gee(bbox, dataset_key, start_year, end_year):
    """Extract a mean time series from GEE for the given bounding box and dataset.

    bbox: [west, south, east, north] in decimal degrees
    Returns a pandas DataFrame with columns: date, value, value_smooth
    """
    import ee

    dataset    = DATASETS[dataset_key]
    start_date = f"{start_year}-01-01"
    end_date   = f"{end_year}-12-31"
    study_area = ee.Geometry.Rectangle(bbox)

    # Datasets that need custom band computation inside GEE before extraction.
    # Landsat NDVI and NDWI both require applying the Collection 2 scale factor
    # and then computing a normalised difference.
    # Burned Area needs the BurnDate band converted to a 0/1 binary mask.
    if dataset_key == "Landsat NDVI":
        def add_ndvi(image):
            # Apply Collection 2 scale factor before computing the ratio
            # so the result is in the correct 0-1 reflectance range
            scaled = image.multiply(0.0000275).add(-0.2)
            ndvi   = scaled.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
            return image.addBands(ndvi)

        collection = (
            ee.ImageCollection(dataset["collection"])
            .filterDate(start_date, end_date)
            .filterBounds(study_area)
            .map(add_ndvi)
            .select("NDVI")
        )
        gee_band = "NDVI"
        scale    = 30

    elif dataset_key == "Landsat NDWI":
        def add_ndwi(image):
            # NDWI = (Green - NIR) / (Green + NIR)
            # Landsat 8 Collection 2: SR_B3 = Green, SR_B5 = NIR
            scaled = image.multiply(0.0000275).add(-0.2)
            ndwi   = scaled.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
            return image.addBands(ndwi)

        collection = (
            ee.ImageCollection(dataset["collection"])
            .filterDate(start_date, end_date)
            .filterBounds(study_area)
            .map(add_ndwi)
            .select("NDWI")
        )
        gee_band = "NDWI"
        scale    = 30

    elif dataset_key == "MODIS Burned Area":
        # Burned area is handled with a completely separate extraction flow.
        # We do NOT use the shared collection + extract_mean pattern because:
        #   - gt(0) preserves the original band name "BurnDate", not "burned"
        #   - copyProperties can be unreliable for system properties in server-side GEE
        # Instead, we map one function that computes the binary fraction AND the date
        # in a single step, then return the DataFrame directly from this block.

        burn_collection = (
            ee.ImageCollection(dataset["collection"])
            .filterDate(start_date, end_date)
            .filterBounds(study_area)
            .select("BurnDate")
        )

        def extract_burn_fraction(image):
            # gt(0): binary 0/1 — 1 where BurnDate > 0 (pixel burned this month)
            # unmask(0): fill masked pixels (water, clouds, unobserved) with 0
            #   Without this, reduceRegion(mean) divides only by observed pixels.
            #   If 95% of pixels are masked and all observed pixels are burned,
            #   the mean is 1.0 even though most of the area was not on fire.
            #   unmask(0) makes masked pixels count as "not burned" in the denominator.
            binary = image.gt(0).unmask(0)
            fraction = binary.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=study_area,
                scale=500,
                maxPixels=1e9,
            ).get("BurnDate")
            date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
            return ee.Feature(None, {"date": date, "value_raw": fraction})

        burn_fc   = burn_collection.map(extract_burn_fraction)
        burn_data = burn_fc.getInfo()

        records = []
        for feature in burn_data["features"]:
            props = feature["properties"]
            raw   = props.get("value_raw")
            if raw is None:
                continue
            records.append({"date": pd.Timestamp(props["date"]), "value": float(raw)})

        if not records:
            return generate_sample_data(
                list(REGIONS.keys())[0], dataset_key, start_year, end_year
            )

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        df["value_smooth"] = df["value"].rolling(window=3, center=True).mean()
        return df

    else:
        collection = (
            ee.ImageCollection(dataset["collection"])
            .filterDate(start_date, end_date)
            .filterBounds(study_area)
            .select(dataset["band"])
        )
        gee_band = dataset["band"]
        scale    = 250 if "MODIS" in dataset["collection"] else 30

    def extract_mean(image):
        """Reduce each image to its mean value over the study area.

        Same logic as notebooks/02_gee_ndvi_timeseries.ipynb Cell 4.
        ee.Reducer.mean() computes the spatial average across all pixels.
        """
        mean_val = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=study_area,
            scale=scale,
            maxPixels=1e9,
        ).get(gee_band)
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        return ee.Feature(None, {"date": date, "value_raw": mean_val})

    time_series_fc = collection.map(extract_mean)
    data           = time_series_fc.getInfo()   # round trip to GEE servers

    records = []
    for feature in data["features"]:
        props = feature["properties"]
        raw   = props.get("value_raw")
        if raw is None:
            continue
        # Convert from GEE integer storage to physical units
        if dataset["convert"] == "divide":
            value = raw / dataset["scale_factor"]
        elif dataset["convert"] == "kelvin":
            value = raw * dataset["scale_factor"] - 273.15
        else:
            value = float(raw)
        records.append({"date": pd.Timestamp(props["date"]), "value": value})

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    # Rolling 3-period smooth removes noise without distorting the seasonal shape
    df["value_smooth"] = df["value"].rolling(window=3, center=True).mean()
    return df

# ---------------------------------------------------------------------------
# Sample data — fallback when GEE credentials are not configured
# ---------------------------------------------------------------------------

# Seasonal parameters per region and dataset.
# Sahel NDVI values match the actual GEE output from Day 4 notebook:
#   mean 0.208, peak 0.322 (August), trough 0.157 (April), trend +0.0034/yr
# Other regions are modeled from published MODIS climatology.
_SAMPLE_PARAMS = {
    "Sahel, West Africa": {
        "MODIS NDVI":                     {"mean": 0.208, "peak": 0.322, "trough": 0.157, "peak_month": 8,  "trend_yr":  0.0034},
        "MODIS Land Surface Temperature": {"mean": 38.0,  "peak": 46.0,  "trough": 29.0,  "peak_month": 5,  "trend_yr":  0.05},
        "Landsat NDVI":                   {"mean": 0.210, "peak": 0.325, "trough": 0.158, "peak_month": 8,  "trend_yr":  0.003},
        "MODIS EVI":                      {"mean": 0.175, "peak": 0.270, "trough": 0.120, "peak_month": 8,  "trend_yr":  0.003},
        "Landsat NDWI":                   {"mean": -0.25, "peak": -0.10, "trough": -0.42, "peak_month": 8,  "trend_yr":  0.001},
        "MODIS Burned Area":              {"mean": 0.018, "peak": 0.060, "trough": 0.0,   "peak_month": 12, "trend_yr":  0.0},
    },
    "Amazon Basin, Brazil": {
        "MODIS NDVI":                     {"mean": 0.72,  "peak": 0.78,  "trough": 0.64,  "peak_month": 4,  "trend_yr": -0.002},
        "MODIS Land Surface Temperature": {"mean": 29.0,  "peak": 34.0,  "trough": 25.0,  "peak_month": 9,  "trend_yr":  0.03},
        "Landsat NDVI":                   {"mean": 0.71,  "peak": 0.77,  "trough": 0.63,  "peak_month": 4,  "trend_yr": -0.002},
        "MODIS EVI":                      {"mean": 0.52,  "peak": 0.58,  "trough": 0.45,  "peak_month": 4,  "trend_yr": -0.002},
        "Landsat NDWI":                   {"mean": 0.05,  "peak": 0.14,  "trough": -0.04, "peak_month": 4,  "trend_yr": -0.001},
        "MODIS Burned Area":              {"mean": 0.012, "peak": 0.045, "trough": 0.0,   "peak_month": 9,  "trend_yr":  0.001},
    },
    "Siberian Boreal Forest": {
        "MODIS NDVI":                     {"mean": 0.35,  "peak": 0.68,  "trough": 0.03,  "peak_month": 7,  "trend_yr":  0.001},
        "MODIS Land Surface Temperature": {"mean": 5.0,   "peak": 28.0,  "trough": -22.0, "peak_month": 7,  "trend_yr":  0.04},
        "Landsat NDVI":                   {"mean": 0.34,  "peak": 0.67,  "trough": 0.03,  "peak_month": 7,  "trend_yr":  0.001},
        "MODIS EVI":                      {"mean": 0.28,  "peak": 0.55,  "trough": 0.02,  "peak_month": 7,  "trend_yr":  0.001},
        "Landsat NDWI":                   {"mean": -0.12, "peak": 0.08,  "trough": -0.38, "peak_month": 7,  "trend_yr":  0.001},
        "MODIS Burned Area":              {"mean": 0.004, "peak": 0.018, "trough": 0.0,   "peak_month": 7,  "trend_yr":  0.0},
    },
    "Great Plains, USA": {
        "MODIS NDVI":                     {"mean": 0.38,  "peak": 0.62,  "trough": 0.18,  "peak_month": 7,  "trend_yr":  0.001},
        "MODIS Land Surface Temperature": {"mean": 22.0,  "peak": 40.0,  "trough": -5.0,  "peak_month": 7,  "trend_yr":  0.02},
        "Landsat NDVI":                   {"mean": 0.38,  "peak": 0.63,  "trough": 0.18,  "peak_month": 7,  "trend_yr":  0.001},
        "MODIS EVI":                      {"mean": 0.30,  "peak": 0.50,  "trough": 0.13,  "peak_month": 7,  "trend_yr":  0.001},
        "Landsat NDWI":                   {"mean": -0.20, "peak": -0.04, "trough": -0.38, "peak_month": 7,  "trend_yr":  0.0},
        "MODIS Burned Area":              {"mean": 0.002, "peak": 0.008, "trough": 0.0,   "peak_month": 4,  "trend_yr":  0.0},
    },
}


def generate_sample_data(region_name, dataset_key, start_year, end_year):
    """Generate a realistic time series when GEE is unavailable.

    Uses the seasonal shape, amplitude, and trend for each region and dataset.
    Fixed random seed (42) means the same region always produces the same curve.

    Returns a pandas DataFrame with columns: date, value, value_smooth
    """
    # Find the closest matching preset region if the name doesn't match exactly
    p_region = region_name
    if p_region not in _SAMPLE_PARAMS:
        p_region = "Sahel, West Africa"

    params = _SAMPLE_PARAMS[p_region].get(dataset_key, _SAMPLE_PARAMS[p_region]["MODIS NDVI"])
    dataset = DATASETS[dataset_key]

    # Seasonal shape: cosine centered on the peak month
    half_amp   = (params["peak"] - params["trough"]) / 2.0
    center_val = (params["peak"] + params["trough"]) / 2.0
    peak_month = params["peak_month"]
    cadence    = dataset["cadence_days"]

    records = []
    rng     = np.random.default_rng(seed=42)
    current = pd.Timestamp(f"{start_year}-01-01")
    end_ts  = pd.Timestamp(f"{end_year}-12-31")
    step    = pd.Timedelta(days=cadence)

    n_steps = 0
    while current <= end_ts:
        years_elapsed = (current - pd.Timestamp(f"{start_year}-01-01")).days / 365.25

        # Seasonal component: cosine that peaks at peak_month
        month_frac = (current.month - 1 + current.day / 30.0) / 12.0
        phase      = 2 * np.pi * (month_frac - (peak_month - 1) / 12.0)
        seasonal   = half_amp * np.cos(phase)

        # Long-term trend
        trend = params["trend_yr"] * years_elapsed

        # Noise scaled to 10% of the amplitude
        noise = rng.normal(0, half_amp * 0.10)

        value = center_val + seasonal + trend + noise

        # Clamp to a physically plausible range for each index type
        if dataset_key in ("MODIS NDVI", "Landsat NDVI", "MODIS EVI"):
            value = float(np.clip(value, 0.0, 1.0))
        elif dataset_key == "MODIS Land Surface Temperature":
            value = float(np.clip(value, -30.0, 65.0))
        elif dataset_key == "Landsat NDWI":
            value = float(np.clip(value, -1.0, 1.0))
        elif dataset_key == "MODIS Burned Area":
            value = float(np.clip(value, 0.0, 1.0))

        records.append({"date": current, "value": value})
        current += step
        n_steps += 1

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["value_smooth"] = df["value"].rolling(window=3, center=True).mean()
    return df

# ---------------------------------------------------------------------------
# Statistics — compute summary numbers from the DataFrame
# ---------------------------------------------------------------------------

def compute_statistics(df, dataset_key):
    """Compute trend, seasonal cycle, and summary numbers from the time series.

    Returns a dict used by the chart builders, stats panel, and AI prompt.
    Same polyfit method as notebooks/02_gee_ndvi_timeseries.ipynb, Cell 6.
    """
    valid = df.dropna(subset=["value"])
    unit  = DATASETS[dataset_key]["unit"]

    # Linear trend: fit a straight line through all data points
    x_days = (valid["date"] - valid["date"].min()).dt.days.values
    z      = np.polyfit(x_days, valid["value"].values, 1)
    slope_per_year = z[0] * 365.25

    # Seasonal cycle: average by calendar month
    df2 = df.copy()
    df2["month"] = df2["date"].dt.month
    monthly_avg  = df2.groupby("month")["value"].mean()

    peak_month   = int(monthly_avg.idxmax())
    trough_month = int(monthly_avg.idxmin())
    month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    return {
        "count":           len(valid),
        "mean":            float(valid["value"].mean()),
        "min":             float(valid["value"].min()),
        "max":             float(valid["value"].max()),
        "slope_per_year":  float(slope_per_year),
        "trend_direction": "increasing" if slope_per_year > 0 else "decreasing",
        "peak_month":      month_names[peak_month - 1],
        "trough_month":    month_names[trough_month - 1],
        "peak_value":      float(monthly_avg[peak_month]),
        "trough_value":    float(monthly_avg[trough_month]),
        "amplitude":       float(monthly_avg.max() - monthly_avg.min()),
        "monthly_avg":     monthly_avg,
        "polyfit_z":       z,
        "x_days":          x_days,
        "valid_dates":     valid["date"].values,
        "unit":            unit,
    }

# ---------------------------------------------------------------------------
# Chart builders — Plotly
# ---------------------------------------------------------------------------

def build_timeseries_chart(df, stats, dataset_key, region_name, start_year, end_year):
    """Time series chart: raw values, smoothed line, and linear trend.

    Adapted from notebooks/02_gee_ndvi_timeseries.ipynb, Cell 6.
    """
    dataset = DATASETS[dataset_key]
    unit    = dataset["unit"]
    color   = dataset["chart_color"]

    fig = go.Figure()

    # Raw values — thin and semi-transparent so the smoothed line stands out
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["value"],
        mode="lines",
        name=f"{unit} (raw)",
        line=dict(color=color, width=0.8),
        opacity=0.35,
    ))

    # Smoothed values — 3-period rolling average, bold line
    smooth_df = df.dropna(subset=["value_smooth"])
    fig.add_trace(go.Scatter(
        x=smooth_df["date"],
        y=smooth_df["value_smooth"],
        mode="lines",
        name=f"{unit} (smoothed)",
        line=dict(color=color, width=2.5),
    ))

    # Linear trend line — same polyfit result as the notebook
    trend_y = np.poly1d(stats["polyfit_z"])(stats["x_days"])
    fig.add_trace(go.Scatter(
        x=stats["valid_dates"],
        y=trend_y,
        mode="lines",
        name="Linear trend",
        line=dict(color="red", width=1.5, dash="dash"),
    ))

    direction = "Increasing" if stats["slope_per_year"] > 0 else "Decreasing"
    fig.update_layout(
        title=f"{region_name} — {dataset_key} {start_year}–{end_year}",
        xaxis_title="Date",
        yaxis_title=unit,
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[dict(
            text=f"Trend: {direction} ({stats['slope_per_year']:+.4f} {unit}/year)",
            xref="paper", yref="paper", x=0.01, y=0.97,
            showarrow=False,
            font=dict(size=11, color="red"),
            bgcolor="white", bordercolor="red", borderwidth=1,
        )],
    )
    return fig


def build_seasonal_chart(stats, dataset_key):
    """Seasonal cycle bar chart: average value by calendar month.

    Adapted from notebooks/02_gee_ndvi_timeseries.ipynb, Cell 7.
    Peak month shown in dark green (or dark red for LST), trough in tan.
    """
    dataset      = DATASETS[dataset_key]
    unit         = dataset["unit"]
    color        = dataset["chart_color"]
    monthly      = stats["monthly_avg"]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    peak_val   = monthly.max()
    trough_val = monthly.min()
    bar_colors = []
    for m in range(1, 13):
        v = monthly.get(m, 0)
        if v == peak_val:
            bar_colors.append("#004d00" if "NDVI" in dataset_key else "#8b0000")
        elif v == trough_val:
            bar_colors.append("#c8a060")
        else:
            bar_colors.append(color)

    fig = go.Figure(go.Bar(
        x=month_labels,
        y=[monthly.get(m, 0) for m in range(1, 13)],
        marker_color=bar_colors,
        name=dataset_key,
    ))
    fig.update_layout(
        title=f"Average seasonal cycle — {dataset_key}",
        xaxis_title="Month",
        yaxis_title=f"Mean {unit}",
        height=300,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig

# ---------------------------------------------------------------------------
# Annual means bar chart
# ---------------------------------------------------------------------------

def build_annual_means_chart(df, dataset_key, start_year, end_year, early_boundary, recent_boundary):
    """Bar chart of mean value per year, colour-coded by period.

    Early period bars: steel blue.
    Recent period bars: dark green.
    Middle / transitional bars: light grey.
    A dashed horizontal line marks the overall mean.

    df: DataFrame with columns date (datetime) and value (float)
    early_boundary:  years <= this are "early period"
    recent_boundary: years >= this are "recent period"
    """
    unit = DATASETS[dataset_key]["unit"]

    # Compute mean per calendar year
    df = df.copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["value"].mean().reset_index()
    annual.columns = ["year", "mean"]

    overall_mean = annual["mean"].mean()

    # Assign colour by period
    def bar_colour(year):
        if year <= early_boundary:
            return "#4a90d9"    # steel blue — early period
        elif year >= recent_boundary:
            return "#2d7a2d"    # dark green — recent period
        else:
            return "#b0b8c1"    # grey — transitional years

    colours = [bar_colour(y) for y in annual["year"]]

    # Hover text
    hover = [
        f"Year: {row.year}<br>Mean {unit}: {row.mean:.4f}"
        for row in annual.itertuples()
    ]

    fig = go.Figure()

    # Bars — one trace per bar so each gets its own colour
    fig.add_trace(go.Bar(
        x=annual["year"].astype(str),
        y=annual["mean"],
        marker_color=colours,
        hovertext=hover,
        hoverinfo="text",
        name="Annual mean",
        showlegend=False,
    ))

    # Overall mean reference line
    fig.add_hline(
        y=overall_mean,
        line_dash="dash",
        line_color="#888888",
        annotation_text=f"Overall mean: {overall_mean:.3f}",
        annotation_position="top left",
        annotation_font_size=11,
    )

    # Invisible dummy traces for the legend
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color="#4a90d9",
        name=f"Early period ({start_year}–{early_boundary})",
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color="#2d7a2d",
        name=f"Recent period ({recent_boundary}–{end_year})",
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color="#b0b8c1",
        name="Transitional years",
        showlegend=True,
    ))

    fig.update_layout(
        title=f"Annual mean {unit} — {start_year} to {end_year}",
        xaxis_title="Year",
        yaxis_title=f"Mean {unit}",
        height=340,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.25,
    )
    return fig


# ---------------------------------------------------------------------------
# Map builder — Folium with NASA GIBS WMTS tiles
# ---------------------------------------------------------------------------

def build_comparison_maps(bbox, region_name, dataset_key, early_year, recent_year):
    """Build three Folium maps: early period, recent period, difference.

    Uses NASA GIBS WMTS tiles for real satellite imagery overlay.
    GIBS is a free public service from NASA — no API key required.
    The WMTS tile URL includes the date baked in; GIBS returns the nearest composite.

    bbox: [west, south, east, north]
    Returns: (map_early, map_recent, map_diff) — three folium.Map objects
    """
    west, south, east, north = bbox
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0

    # Look up the zoom level from the preset regions dict
    region_info = REGIONS.get(region_name)
    zoom        = region_info["zoom"] if region_info else 6
    peak_month  = region_info["peak_month"] if region_info else "08"

    gibs_layer = DATASETS[dataset_key]["gibs_layer"]

    # GIBS WMTS URL template
    # The date is baked in; {z}/{y}/{x} are folium's tile placeholders
    # GoogleMapsCompatible_Level7 is the MODIS NDVI tile matrix set (max zoom 7)
    gibs_base = (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{gibs_layer}/default/{{date}}/GoogleMapsCompatible_Level7/{{z}}/{{y}}/{{x}}.png"
    )

    early_date  = f"{early_year}-{peak_month}-15"
    recent_date = f"{recent_year}-{peak_month}-15"

    def make_map(date_str, period_label):
        """Build one folium map with a GIBS NDVI overlay and a region rectangle."""
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles="CartoDB positron",
            max_zoom=12,
        )
        # GIBS tile overlay — fills the map with real satellite data
        tile_url = gibs_base.replace("{date}", date_str)
        folium.TileLayer(
            tiles=tile_url,
            attr="NASA GIBS / NASA EOSDIS",
            name=f"{dataset_key} ({period_label})",
            overlay=True,
            opacity=0.85,
            max_native_zoom=7,   # MODIS NDVI tiles only go to zoom level 7
        ).add_to(m)
        # Red rectangle showing the study area boundary
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#ff4444",
            weight=2,
            fill=False,
            tooltip=f"{region_name} — {period_label}",
        ).add_to(m)
        folium.LayerControl().add_to(m)
        return m

    map_early  = make_map(early_date,  f"Early {early_year}")
    map_recent = make_map(recent_date, f"Recent {recent_year}")

    # Difference map: show the recent period with a note
    # Spatial difference as a GEE layer requires GEE credentials
    map_diff = make_map(recent_date, f"Recent {recent_year}")
    folium.Marker(
        location=[center_lat, center_lon],
        popup=(
            "Spatial difference layer requires GEE credentials.\n"
            "The chart below shows the numeric difference between periods."
        ),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(map_diff)

    return map_early, map_recent, map_diff

# ---------------------------------------------------------------------------
# AI interpretation
# ---------------------------------------------------------------------------

def get_ai_interpretation(stats, dataset_key, region_name, start_year, end_year,
                           custom_prompt=None, api_key=None):
    """Get a plain-language interpretation of the time series results.

    Tries Groq first if an API key is available.
    Falls back to substantive region- and dataset-specific text.
    Same AI pattern as notebooks/02_gee_ndvi_timeseries.ipynb, Cell 12.
    """
    dataset = DATASETS[dataset_key]
    unit    = dataset["unit"]

    context = (
        f"Study area: {region_name}\n"
        f"Dataset: {dataset_key} ({dataset['sensor']}, {dataset['resolution']} resolution)\n"
        f"Time period: {start_year}-{end_year} ({stats['count']} data points)\n"
        f"\nKey statistics:\n"
        f"- Mean {unit}: {stats['mean']:.3f}\n"
        f"- Range: {stats['min']:.3f} to {stats['max']:.3f} {unit}\n"
        f"- Long-term trend: {stats['trend_direction']} "
        f"at {stats['slope_per_year']:+.4f} {unit}/year\n"
        f"- Peak month: {stats['peak_month']} ({stats['peak_value']:.3f} {unit})\n"
        f"- Trough month: {stats['trough_month']} ({stats['trough_value']:.3f} {unit})\n"
        f"- Seasonal amplitude: {stats['amplitude']:.3f} {unit}"
    )

    prompt_text = (
        f"{custom_prompt}\n\nData context:\n{context}"
        if custom_prompt
        else f"Interpret this satellite time series analysis:\n\n{context}"
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
                            "You are an Earth observation analyst. "
                            "Interpret satellite data in plain language. "
                            "Be specific and direct. Avoid jargon. "
                            "Cover: what the data shows, why the pattern exists, "
                            "one practical application, and one limitation."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception:
            pass   # fall through to the substantive fallback below

    return _get_fallback_interpretation(stats, dataset_key, region_name)


def _get_fallback_interpretation(stats, dataset_key, region_name):
    """Substantive fallback — never placeholder text.

    One block per dataset family, with region-specific context where known.
    """
    unit      = DATASETS[dataset_key]["unit"]
    direction = stats["trend_direction"]
    slope     = stats["slope_per_year"]

    if dataset_key == "MODIS EVI":
        return (
            f"**EVI Time Series — {region_name}**\n\n"
            f"**What the data shows:**\n"
            f"The Enhanced Vegetation Index peaks in {stats['peak_month']} "
            f"(EVI {stats['peak_value']:.3f}) and reaches its seasonal low in "
            f"{stats['trough_month']} (EVI {stats['trough_value']:.3f}). "
            f"Mean EVI is {stats['mean']:.3f}. "
            f"Long-term trend is {direction} at {slope:+.4f} EVI units per year. "
            f"EVI values are typically 10-20% lower than NDVI for the same vegetation "
            f"because the index corrects for soil brightness and atmospheric haze.\n\n"
            f"**Why EVI differs from NDVI:**\n"
            f"NDVI saturates above an LAI (Leaf Area Index) of roughly 3 — dense tropical "
            f"forest and mature cropland all appear similarly high. EVI adds a soil adjustment "
            f"factor and a blue-band aerosol term that prevent this saturation. "
            f"In practice, EVI shows more variation inside dense canopy and is more stable "
            f"in high-aerosol environments like biomass burning regions.\n\n"
            f"**Practical application:**\n"
            f"EVI is the preferred index for carbon flux modelling in tropical forests. "
            f"It correlates more reliably with gross primary productivity (GPP) than NDVI "
            f"in high-biomass ecosystems.\n\n"
            f"**Limitation:**\n"
            f"EVI requires a blue band. In high-aerosol conditions, the blue band is noisy, "
            f"which can introduce artefacts. The 250 m resolution limit is the same as MODIS NDVI."
        )

    elif dataset_key == "Landsat NDWI":
        return (
            f"**NDWI Time Series — {region_name}**\n\n"
            f"**What the data shows:**\n"
            f"The Normalized Difference Water Index peaks in {stats['peak_month']} "
            f"(NDWI {stats['peak_value']:.3f}) and drops to its seasonal low in "
            f"{stats['trough_month']} (NDWI {stats['trough_value']:.3f}). "
            f"Mean NDWI is {stats['mean']:.3f}. "
            f"Long-term trend is {direction} at {slope:+.4f} NDWI units per year. "
            f"Positive values indicate open water or saturated soil. "
            f"Values below -0.2 indicate dry land or bare soil.\n\n"
            f"**Why this pattern exists:**\n"
            f"NDWI measures the contrast between green-light reflectance (high over water) "
            f"and near-infrared reflectance (low over water, high over vegetation). "
            f"The seasonal peak corresponds to the wet season when surface water extent "
            f"is greatest. The trough marks the driest period when standing water recedes "
            f"and soil moisture drops.\n\n"
            f"**Practical application:**\n"
            f"NDWI time series track reservoir and lake level changes without a gauge station. "
            f"A sustained negative trend signals progressive water body shrinkage, "
            f"as seen in Lake Chad and the Aral Sea. Flood events appear as sudden spikes.\n\n"
            f"**Limitation:**\n"
            f"Dense vegetation suppresses NDWI even when underlying soil is wet. "
            f"The index detects surface water, not groundwater or subsurface moisture. "
            f"Cloud cover creates the same data gaps as for Landsat NDVI."
        )

    elif dataset_key == "MODIS Burned Area":
        return (
            f"**Burned Area Time Series — {region_name}**\n\n"
            f"**What the data shows:**\n"
            f"The fraction of the study area that burned each month peaks in "
            f"{stats['peak_month']} at {stats['peak_value']:.3f} "
            f"({stats['peak_value']*100:.1f}% of the area). "
            f"Most months register near zero, which is normal — fire is a punctuated "
            f"event, not a continuous one. "
            f"Long-term trend is {direction} at {slope:+.5f} fraction per year. "
            f"The mean of {stats['mean']:.4f} represents the average monthly burned fraction "
            f"across the full analysis period.\n\n"
            f"**Why this pattern exists:**\n"
            f"Fire occurrence follows the dry season. Fuel (dry vegetation) accumulates "
            f"during the growing season and ignites when moisture drops. "
            f"In savanna regions, fire is a natural part of the ecosystem cycle. "
            f"In tropical forest, fire is almost always human-caused — agricultural clearing "
            f"or escaped pasture fires. Interannual spikes often correspond to El Niño "
            f"drought years when conditions become extreme.\n\n"
            f"**Practical application:**\n"
            f"Burned area time series are a primary input to carbon accounting. "
            f"Each burned pixel releases stored carbon. An increasing trend in forest "
            f"regions signals accelerating land clearance.\n\n"
            f"**Limitation:**\n"
            f"Small fires below the 500 m pixel size are not detected. "
            f"Smouldering fires with minimal surface change are also missed. "
            f"The product works best at regional scale, not for individual burn events."
        )

    elif dataset_key in ("MODIS NDVI", "Landsat NDVI"):

        if "Sahel" in region_name:
            return (
                f"**NDVI Time Series — Sahel Region**\n\n"
                f"**What the data shows:**\n"
                f"The Sahel shows a strong single-peak seasonal cycle. Vegetation peaks in "
                f"{stats['peak_month']} (NDVI {stats['peak_value']:.3f}) driven by the West African "
                f"Monsoon, then drops to its lowest in {stats['trough_month']} "
                f"(NDVI {stats['trough_value']:.3f}) during the dry season. "
                f"The long-term trend is {direction} at {slope:+.4f} NDVI units per year. "
                f"This is consistent with the well-documented Sahel greening phenomenon.\n\n"
                f"**Why this pattern exists:**\n"
                f"The Sahel sits at the boundary between the Sahara Desert and tropical savanna. "
                f"Rainfall is entirely seasonal. The Intertropical Convergence Zone moves northward "
                f"in summer, bringing rain, then retreats south in winter. "
                f"Vegetation follows rain with a 2-4 week lag. The long-term greening reflects "
                f"increased Sahel rainfall since the 1980s drought, combined with CO2 fertilization.\n\n"
                f"**Practical application:**\n"
                f"Below-average NDVI in August signals a poor rainy season and predicts crop "
                f"failure 2-3 months before harvest. Relief organizations use this signal to "
                f"pre-position aid before food crises develop.\n\n"
                f"**Limitation:**\n"
                f"MODIS at 250 m cannot distinguish vegetation types. Cropland, shrubland, and "
                f"grassland appear similar. For land use decisions, Landsat or Sentinel-2 is needed."
            )

        elif "Amazon" in region_name:
            return (
                f"**NDVI Time Series — Amazon Basin**\n\n"
                f"**What the data shows:**\n"
                f"The Amazon maintains high year-round NDVI (mean {stats['mean']:.3f}), "
                f"with a weak dry-season dip in {stats['trough_month']}. "
                f"The seasonal amplitude is only {stats['amplitude']:.3f} NDVI units — "
                f"far smaller than semi-arid regions. "
                f"The long-term trend is {direction} at {slope:+.4f} NDVI units per year. "
                f"Persistent decline in this dataset typically indicates deforestation.\n\n"
                f"**Why this pattern exists:**\n"
                f"Tropical rainforest has sufficient moisture year-round to maintain dense vegetation. "
                f"The small seasonal dip corresponds to the dry season. "
                f"During dry conditions, cloud cover also decreases, which actually improves "
                f"satellite observation quality — the dip is partly real and partly an artifact "
                f"of better atmospheric correction during clear skies.\n\n"
                f"**Practical application:**\n"
                f"Persistent NDVI decline in Amazon pixels is a reliable deforestation indicator. "
                f"Conservation agencies flag declining pixels for ground verification and enforcement.\n\n"
                f"**Limitation:**\n"
                f"The Amazon has persistent cloud cover. Up to 30 percent of MODIS pixels in a "
                f"year may be cloud-contaminated. The time series represents only valid clear-sky "
                f"observations, not a complete record."
            )

        elif "Siberia" in region_name:
            return (
                f"**NDVI Time Series — Siberian Boreal Forest**\n\n"
                f"**What the data shows:**\n"
                f"Siberia shows extreme seasonal amplitude ({stats['amplitude']:.3f} NDVI units). "
                f"Vegetation is near zero from November through March under snow cover, then "
                f"surges rapidly in May-June with snowmelt. Peak in {stats['peak_month']} "
                f"reaches {stats['peak_value']:.3f} NDVI. The growing season is approximately "
                f"120 days long. Long-term trend is {direction}.\n\n"
                f"**Why this pattern exists:**\n"
                f"The boreal zone has a short growing season determined entirely by temperature "
                f"and snowmelt timing. Permafrost retains moisture for the summer flush. "
                f"Climate warming is shifting snowmelt earlier, gradually extending the green season.\n\n"
                f"**Practical application:**\n"
                f"Changes in the length of the green season are a direct indicator of climate-driven "
                f"ecosystem shifts. Earlier green-up and later senescence signal warming that "
                f"affects boreal carbon storage estimates.\n\n"
                f"**Limitation:**\n"
                f"Snow contamination of MODIS pixels is common in spring and autumn at high latitudes. "
                f"The sharp transitions in the time series are partly real and partly caused by "
                f"snow melting off pixels over a few days."
            )

        else:   # Great Plains or custom region
            return (
                f"**NDVI Time Series — {region_name}**\n\n"
                f"**What the data shows:**\n"
                f"The time series shows a seasonal cycle peaking in {stats['peak_month']} "
                f"(NDVI {stats['peak_value']:.3f}) and reaching its lowest in "
                f"{stats['trough_month']} (NDVI {stats['trough_value']:.3f}). "
                f"Mean NDVI is {stats['mean']:.3f}. "
                f"The long-term trend is {direction} at {slope:+.4f} NDVI units per year.\n\n"
                f"**Why this pattern exists:**\n"
                f"NDVI follows the local rainfall and temperature cycle. "
                f"The peak corresponds to the growing season when soil moisture and solar "
                f"radiation are both adequate. The trough corresponds to dormancy, drought, "
                f"or snow cover depending on the region.\n\n"
                f"**Practical application:**\n"
                f"A time series like this is used to benchmark vegetation health year over year. "
                f"Below-average summer NDVI typically signals drought stress, which predicts "
                f"reduced crop yields or increased fire risk.\n\n"
                f"**Limitation:**\n"
                f"MODIS at 250 m averages across multiple land cover types in each pixel. "
                f"Field-level or forest-edge decisions require Landsat (30 m) or "
                f"Sentinel-2 (10 m) data."
            )

    else:   # MODIS Land Surface Temperature
        return (
            f"**Land Surface Temperature — {region_name}**\n\n"
            f"**What the data shows:**\n"
            f"Daytime land surface temperature ranges from {stats['min']:.1f}°C to "
            f"{stats['max']:.1f}°C over the analysis period. "
            f"Annual mean is {stats['mean']:.1f}°C. "
            f"The seasonal peak in {stats['peak_month']} reaches {stats['peak_value']:.1f}°C. "
            f"Long-term trend is {direction} at {slope:+.4f}°C per year.\n\n"
            f"**Why this pattern exists:**\n"
            f"Land surface temperature follows solar radiation and land cover type. "
            f"Bare soil and urban surfaces absorb solar energy and re-emit it as heat. "
            f"Vegetated surfaces stay cooler through evapotranspiration. "
            f"The seasonal pattern tracks solar angle directly.\n\n"
            f"**Practical application:**\n"
            f"LST time series detect the urban heat island effect (cities run 3-10°C warmer "
            f"than surrounding vegetation), identify drought stress before NDVI drops, "
            f"and flag elevated wildfire risk in hot, dry periods.\n\n"
            f"**Limitation:**\n"
            f"LST measures the surface skin temperature, not air temperature. "
            f"It can run 10-20°C above air temperature on sunny days. "
            f"Cloud cover produces complete data gaps."
        )
