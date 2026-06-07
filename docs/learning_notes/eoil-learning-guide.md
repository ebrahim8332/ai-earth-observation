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

---

## Day 5: Sentinel-1 SAR Basics

**What was built:** Jupyter notebook `notebooks/03_sentinel1_sar_basics.ipynb`. Sentinel-1
GRD analysis over the Port of Rotterdam using GEE. VV band map, VH band map, false color
composite (VV/VH/ratio), two-date comparison, backscatter change map, Groq interpretation.
Confirmed water at -24 dB, ships pushing VV max to +29.9 dB.

**Key concepts:**

- **SAR (Synthetic Aperture Radar)** transmits its own microwave pulses and records the
  return. It works through clouds and at night. Optical sensors need sunlight and clear skies.
  SAR does not. This is the defining advantage for flood mapping, ship detection, and
  monitoring in cloudy tropical regions.
- **Backscatter** is the intensity of the radar signal that returns to the sensor. Measured
  in dB (decibels), a logarithmic scale. Every 3 dB increase = double the energy returned.
- **Specular reflection** is what calm water does to radar. It acts like a mirror and
  bounces the pulse away from the sensor. Result: water appears very dark (-20 to -25 dB).
- **Corner reflector** is what ships and buildings do. Right-angle metal surfaces bounce
  energy directly back. Result: ships appear as very bright point targets (up to +30 dB).
- **VV polarization** (vertical transmit, vertical receive) responds to surface roughness
  and vertical structures. Strong for urban areas, ports, and open water contrast.
- **VH polarization** (vertical transmit, horizontal receive) responds to volume scattering.
  Strong for vegetation canopy, forest, and complex 3D structures.
- **False color composite** stacks VV, VH, and VV/VH ratio into RGB channels. Each channel
  highlights a different surface type. One image reveals water, vegetation, and urban areas
  simultaneously.
- **GRD (Ground Range Detected)** means the raw radar data has been processed into
  ground-projected intensity values. It is the standard ready-to-use Sentinel-1 product.
  The alternative (SLC — Single Look Complex) retains phase information for interferometry.

**What to remember:** Both SAR dates over Rotterdam were acquired in winter when cloud
cover exceeds 80%. An optical sensor would show nothing but white cloud on those days.
This is not a hypothetical advantage — it is why SAR is the primary sensor for maritime
monitoring, disaster response, and tropical agriculture.

---

## Day 6: Time Series Explorer Portal Module

**What was built:** EO Explorer upgraded to v1.3. Added Time Series Explorer as a sidebar
module. User picks any region and date range and the app plots a vegetation or temperature
trend over time using Google Earth Engine. Datasets: MODIS NDVI, EVI, LST, NDWI, Burned
Area, and Landsat NDVI/NDWI. Annual comparison bar chart added. GEE credentials wired in.

**Key concepts:**

- **EVI (Enhanced Vegetation Index)** is a refined version of NDVI. It reduces atmospheric
  distortion and soil background noise. Better than NDVI for dense canopy regions (rainforest,
  high-biomass cropland) where NDVI saturates above ~0.8.
- **LST (Land Surface Temperature)** is the temperature of the ground surface as seen from
  space. Not air temperature. Urban heat islands show as bright spots in LST. Desert sand
  can reach 60–70°C surface temperature while air temperature is 40°C.
- **NDWI (Normalized Difference Water Index)** uses Green and NIR bands to detect open water
  and wet soil. Positive values indicate water. Negative values indicate vegetation or dry land.
- **Burned Area (MCD64A1)** is a MODIS product that detects fire scars globally at 500m
  resolution, updated monthly. Burn scars absorb more light post-fire, which drops reflectance
  in a detectable pattern.

**What to remember:** Landsat time series takes longer than MODIS because Landsat requires
computing NDVI from raw bands (B5 and B4) for every image in the collection. MODIS has
pre-computed NDVI values ready to read directly. Use MODIS for quick trend analysis.
Use Landsat when you need higher spatial resolution or pre-2000 data is not needed.

---

## Day 7: SAR Explorer + Welcome Panel

**What was built:** Added two more sidebar modules to the portal, bumped to v1.4. SAR Explorer
lets users pick any coastal or port region, select two dates, and renders four GEE-backed
Folium layers: VV backscatter, VH backscatter, false color composite, and a change map showing
what shifted between the two dates. Welcome Panel replaced the old sample-data map as the
default landing page.

**Key concepts:**

- **GEE tile URL pattern** is how the Change Map, SAR Explorer, and Change Detection modules
  all render satellite data on a Folium map without downloading pixels. You call
  `image.getMapId(vis_params)` on GEE's server. GEE returns a URL template. Folium adds that
  URL as a TileLayer. The user's browser fetches individual map tiles on demand as they pan
  and zoom. No pixel arrays ever reach the Python process.
- **Two-step run pattern** prevents Streamlit from re-running analysis on every widget
  interaction. The Run button stores parameters in `st.session_state` and calls `st.rerun()`.
  On the next render the app detects the stored params and executes the analysis. Without this
  pattern, analysis would re-run every time the user touches any widget.
- **Bbox size warning** flags when the user's region is too large for meaningful SAR analysis.
  SAR change maps lose interpretability above roughly 500 km width because small changes
  (individual ships, structure shifts) become invisible at the required zoom level.

**What to remember:** The portal uses a single `app.py` with a sidebar radio button. All
routing is one `if/elif` block. Adding a new module means: create the module file, add one
radio option, add one routing block. Everything else stays the same.

---

## Day 8: EVI, NDWI, Burned Area, Week 1 Primer

**What was built:** Added EVI, NDWI, and Burned Area to the Time Series Explorer dataset
dropdown. Wrote the Week 1 primer document in `docs/learning_notes/week1-primer.md` covering
the three data platforms, five key concepts, and practical notes for team onboarding.

**Key concepts:**

- **Dataset dropdown pattern** — adding a new time series dataset means one new entry in
  the `DATASETS` dict in `gee_timeseries.py`. Each entry specifies the GEE collection,
  the band name, a scale factor, and display labels. The rest of the module reads from
  that dict at runtime. New datasets cost about five lines of code.
- **Team primer purpose** — the primer translates technical outputs into business language
  a non-specialist can act on. It answers: what data is available, what can be measured,
  what decisions it supports. Writing it forces clarity about what was actually built.

**What to remember:** Writing documentation immediately after building is faster and more
accurate than writing it later. The mechanics are fresh. The decisions are still visible.
Deferred documentation always loses detail.

---

## Day 9: Change Detection Portal Module

**What was built:** Added Change Detection as the fifth sidebar module. EO Explorer v1.5.
User picks a location and two dates. App fetches NDVI from GEE for each date (Sentinel-2
with MODIS fallback), computes the pixel-by-pixel difference, and renders an interactive
Folium map with three toggleable layers. Also created `ai_chain.py` — a 11-model AI fallback
chain shared across all modules. Notebook `04_change_detection.ipynb` committed.

**Key concepts:**

- **NDVI change detection** is the simplest form of land cover change analysis. You compute
  NDVI for Date 1, NDVI for Date 2, subtract them. Positive values mean the area got greener.
  Negative values mean it lost vegetation. A threshold (we use ±0.1) separates signal from
  noise. Pixels below the threshold are treated as stable and shown in white.
- **Change threshold** is the minimum NDVI difference you call a real change. We use 0.1.
  Below that, the difference is likely measurement noise, atmospheric variation, or sensor
  inconsistency between dates rather than an actual land cover change.
- **Extreme threshold** is a second, higher threshold (we use 0.3) that flags severe or
  rapid change. Useful for detecting deforestation, crop failure, or fire recovery — events
  where NDVI shifts dramatically, not just seasonally.
- **GEE grouped reducer** is the efficient way to compute area statistics across multiple
  categories in one server call. Instead of five separate `pixelArea()` calls (one per
  class), you build a classified image (each pixel gets a class code 1–5), stack it with
  a pixel area band, and run one `Reducer.sum().group()` call. GEE returns all class areas
  in one round-trip. Key rule: the data band must come first, the class band second.
  `groupField=1` points at band index 1 (the class band).
- **Session locking** — the AI chain tries models in order and locks to the first one that
  succeeds. The lock is stored in `st.session_state`. Every subsequent AI call in that
  session starts at the locked position. If the locked model fails mid-session (rate limit),
  the chain continues from that position and re-locks to the next working model.
- **ArcGIS geocoding** is more reliable than Nominatim on shared cloud IPs. Nominatim
  enforces a strict rate limit of 1 request per second and blocks high-traffic IPs.
  ArcGIS World Geocoding is free, requires no API key, and handles concurrent requests.
  ArcGIS returns an `extent` object (xmin/ymin/xmax/ymax) which is a proper bounding box,
  not just a point. Use it as the primary. Keep Nominatim as backup.

**What the module shows (Zanzibar test):**
- February 2023 (short rainy season): dense green vegetation across the island
- September 2023 (dry season): lighter green, vegetation drying
- Change map: red patches in Zanzibar Town and central areas (urban heat, reduced canopy),
  green patches in northern and eastern areas (localised rainfall or irrigated land)
- The pattern is ecologically correct for these seasons and this latitude

**Errors encountered and fixed:**
- `Group input must come after weighted inputs` — GEE grouped reducer requires area band
  first, class band second. Fixed by swapping band order and changing groupField from 0 to 1.
- `Need 2 bands for Reducer.combine, has 3` — passing a 3-band image to a combined reducer
  that expects exactly 2 bands (one per reducer). Fixed by splitting into two separate
  `reduceRegion` calls.
- Geocoding failure on all locations — Nominatim rate-limiting on Streamlit Cloud shared IPs.
  Fixed by switching to ArcGIS as primary geocoder.
- AI response truncated mid-sentence — `max_output_tokens=1024` too low for 3-4 paragraph
  responses. Fixed by increasing to 2048 in both Gemini and Groq calls.

**What to remember:** GEE computation cost scales with region area and pixel count. The
`scale` parameter in `reduceRegion` controls pixel size. At 500m scale, GEE processes 4x
more pixels than at 1000m scale, with less than 2% difference in the area totals. For
summary statistics (km² of gain/loss), 1000m is accurate enough and runs roughly twice
as fast. Reserve 500m or finer for precision applications where pixel-level accuracy matters.
