# EOIL Learning Guide

One entry per day. What was built, what was learned, key concepts to remember.

---

## Day 1: EO Explorer v1.0

**What was built:** The foundational Streamlit app. Interactive Folium map with four sample
layers across six EO themes. AI assistant using Groq with substantive fallback responses.
Six modular Python files: app.py, config.py, data_catalog.py, sample_layers.py,
map_builder.py, ai_assistant.py.

**Key concepts:**

- **Streamlit** is a Python library that turns a script into a web app. No HTML required.
  Every time you interact with the app, the script reruns from top to bottom.
- **Folium** builds interactive maps in Python. It wraps Leaflet.js (a JavaScript mapping
  library) so you do not have to write JavaScript.
- **GeoJSON** is a text format for geographic shapes. A polygon in GeoJSON is a list of
  lat/lon coordinate pairs. Folium reads GeoJSON directly to draw shapes on the map.
- **Modular structure** means one file per responsibility. app.py only does layout.
  data_catalog.py only holds metadata. This makes the code easier to maintain and extend.
- **Graceful degradation** means the app must work with no API key. Fallback responses
  must be real content, not placeholder text.

**What to remember:** The 11-model AI fallback chain. Gemini uses the `google-genai` SDK
(new), not `google-generativeai` (deprecated). Import is `from google import genai`.

---

## Day 2: STAC and Cloud-Native Data

**What was built:** Jupyter notebook `notebooks/01_stac_query_demo.ipynb`. Queried
Planetary Computer for Sentinel-2 imagery over the Nile Delta using the STAC protocol.
Streamed a Cloud-Optimized GeoTIFF (COG) without downloading the full file. Rendered
a true-color RGB composite and an NDVI chip.

**Key concepts:**

- **STAC** (SpatioTemporal Asset Catalog) is a standard for indexing geospatial data.
  Think of it as a structured API for satellite imagery. Every image has metadata:
  date, bounding box, cloud cover, band names. You query STAC to find what you need
  before downloading anything.
- **COG** (Cloud-Optimized GeoTIFF) is a satellite image file stored so you can stream
  just the tile you need, not the whole scene. A full Sentinel-2 scene is about 800 MB.
  With COG you can fetch a 50 KB window over your area of interest.
- **NDVI** (Normalized Difference Vegetation Index) measures plant health using the
  ratio of near-infrared and red light. Formula: (NIR - Red) / (NIR + Red).
  Values near 1.0 are dense vegetation. Values near 0 are bare soil or water.
- **Planetary Computer** is Microsoft's free geospatial data platform. It hosts
  Sentinel-2, Landsat, Copernicus DEM, and others. Access via `pystac-client`.
- **rasterio** reads raster files (satellite images). **rioxarray** wraps rasterio
  with xarray to add dimension labels (time, band, x, y). **xarray** is like pandas
  but for multi-dimensional arrays.

**What to remember:** Sign requests with `planetary_computer.sign()` before streaming.
Unsigned requests to Planetary Computer return 403 errors. The signing adds a short-lived
token to the URL.

---

## Day 3: EO Explorer v1.1 — Spectral Explorer

**What was built:** Upgraded EO Explorer to v1.1. Added a second tab: Spectral Explorer.
User enters any location by name, selects a date range and bands, and the app fetches
a live Sentinel-2 chip from Planetary Computer, calculates spectral indices (NDVI, NDWI,
NDSI), and renders a false-color composite. Added three new modules: spectral_explorer.py,
satellite_catalog.py, geocoder.py.

**Key concepts:**

- **Spectral index** is a mathematical ratio of two or more bands that highlights a
  specific surface feature. NDVI for vegetation, NDWI for water, NDSI for snow.
  Each index is designed around how that feature reflects or absorbs specific wavelengths.
- **False color composite** assigns bands to RGB display channels in a non-standard way.
  For example, assigning NIR to Red, Red to Green, Green to Blue makes healthy vegetation
  appear bright red. It reveals features invisible in true color.
- **Geocoder** converts a place name ("Cairo") to lat/lon coordinates. We used the
  Nominatim service from OpenStreetMap, which is free and requires no API key.
- **MGRS tiles** are the grid system Sentinel-2 uses to divide the globe. Each tile is
  100 km x 100 km. If a bounding box crosses a tile boundary, the query returns a split
  image from two different tiles. Keep bounding boxes within one zone to avoid this.
- **Gemini vision** can interpret an image by receiving it as base64-encoded bytes.
  We used it to write plain-language descriptions of NDVI chips.

**What to remember:** The deploy config.toml must have `headless = true` for Streamlit
Cloud. `headless = false` is correct for local development. Two separate config files,
one per environment.

---

## Day 4: GEE NDVI Time Series

**What was built:** Jupyter notebook `notebooks/02_gee_ndvi_timeseries.ipynb`. 10-year
NDVI time series for the Sahel region of West Africa using MODIS MOD13Q1 via Google Earth
Engine. 253 data points at 16-day intervals from 2014-2024. Three charts: time series with
trend line, seasonal cycle bar chart, and three interactive geemap maps (early period,
recent period, NDVI difference). Groq AI interpretation with substantive fallback.

**Key concepts:**

- **Google Earth Engine (GEE)** is Google's cloud platform for planetary-scale geospatial
  analysis. You write Python code locally but all computation runs on Google's servers.
  You never download the full dataset — you ask GEE to compute results and return only
  what you need.
- **ee.ImageCollection** is a stack of satellite images filtered by date and location.
  Think of it as a folder of images you can filter, sort, and process with one command.
- **ee.Reducer.mean()** aggregates all pixels in a region to a single average value.
  This is how you convert a spatial image into a single number for a time series.
- **`.map()` in GEE** applies a function to every image in a collection. It runs on
  GEE's servers in parallel — much faster than a Python for loop.
- **Median composite** takes the middle pixel value across multiple images over time.
  This removes clouds, shadows, and sensor noise. A median of 70 images is far cleaner
  than any single image.
- **Scale factor** — MODIS stores NDVI as integers multiplied by 10000 to save storage.
  NDVI of 0.35 is stored as 3500. Always divide by 10000 when working with MODIS NDVI.
- **MODIS vs Sentinel-2 for time series** — MODIS goes back to 2000 and updates every
  16 days. Sentinel-2 only starts in 2017. For 10-year trend analysis, use MODIS.
  For detailed spatial analysis of a single scene, use Sentinel-2.

**What the data showed:**
- Mean NDVI over 10 years: 0.208 (semi-arid, expected)
- Trend: +0.0034 NDVI units per year (gradual greening confirmed)
- Peak month: August. Lowest month: April. Seasonal amplitude: 0.165
- This confirms the Sahel greening phenomenon documented in the scientific literature.

**What to remember:** `.getInfo()` fetches data from GEE to local Python. Every call is
a round trip to Google's servers. Use it sparingly — one call to get the full time series
is better than calling it inside a loop for each image. That would be 253 separate
network requests instead of one.
