# PROJECT.md
# AI-Native Earth Observation Innovation Lab
# Current status and next steps
# Update this file at the end of every session

---

## Current Status

**Phase:** Arc-based program (revised after Day 12)
**Last completed:** Arc 2 complete — notebook run clean end-to-end, portal module live.
**Next session:** Arc 3 — Flood Detection and Impact Mapping.

---

## Program Design

This program follows a learning arc structure. Each arc covers one analytical domain.
No fixed day count. Each arc takes as long as the learning requires.

Every arc follows the same sequence:
1. Notebook first. Understand the data, run the algorithm, see raw outputs cell by cell.
2. Portal module second. Turn the notebook logic into an interactive sidebar module.
3. Three layers always present in every module: data processing, AI algorithm, generative AI.

---

## Three-Layer Architecture

**Layer 1: Signal processing and data preparation.**
Raw data cleaned, calibrated, and structured. Engineering, not AI.

**Layer 2: AI algorithm layer.**
The analytical intelligence. Finds the pattern. Clustering, classification,
anomaly detection, segmentation, neural networks, foundation models.
This is where real AI work happens. Generative AI never substitutes for this layer.

**Layer 3: Generative AI layer.**
Groq or Gemini interprets what Layer 2 found. Contextualises and recommends.
Always receives structured inputs from Layer 2. Never invents the analysis.

All three layers are visually distinct in every module UI.

---

## Portal Modules — Live

All modules deployed at https://eoil-explorer.streamlit.app

| Module | Added | Status |
|--------|-------|--------|
| Welcome Panel | Day 7 | Live |
| Spectral Explorer | Day 3 | Live — enhancements pending |
| Time Series Explorer | Day 6 | Live |
| SAR Explorer | Day 7 | Live |
| Change Detection | Day 9 | Live |
| AI Imagery Interpreter | Day 10 | Live |
| Emissions Explorer | Day 12 | Live |
| Land Cover Intelligence | Arc 1 | Live |
| Corridor Risk Intelligence | Arc 2 | Live |

---

## Pre-Arc 2 Enhancements — Existing Modules

These improvements are drawn from the backlog review. They complete the existing
modules before new arcs begin. All are additions to live modules, not new modules.

### Enhancement 1: Spectral Explorer — Additional Indices
**Module:** spectral_explorer.py
**What:** Add five spectral indices as new band combination presets.
- NDMI: Normalized Difference Moisture Index — (B08 - B11) / (B08 + B11). Vegetation water stress.
- NBR: Normalized Burn Ratio — (B08 - B12) / (B08 + B12). Burn severity and fire scars.
- SAVI: Soil-Adjusted Vegetation Index — ((B08 - B04) / (B08 + B04 + 0.5)) * 1.5. Vegetation in sparse cover.
- EVI: Enhanced Vegetation Index — 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1). Better than NDVI in dense canopy.
- BSI: Bare Soil Index — ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02)). Bare and exposed soil.
**Why:** Expands analytical range of an already-live module. Each index costs one line of computation.
**Status:** Pending

### Enhancement 2: Export Layer — Three Modules
**Modules:** land_cover.py, gee_change.py, gee_timeseries.py
**What:** Add download buttons to three modules.
- Land Cover Intelligence: Download classification stats as CSV. Download classified map as PNG.
- Change Detection: Download change statistics (area increased, decreased, stable) as CSV.
- Time Series Explorer: Download index time series (date, value) as CSV.
**Why:** Makes portal outputs usable outside the browser. Demonstrates a complete workflow.
**Status:** Complete. Deployed.

### Enhancement 3: Confidence and Data Quality Note — All analytical modules
**Modules:** land_cover.py, gee_change.py, gee_timeseries.py, methane_explorer.py
**What:** Add a small structured data quality block near each Layer 3 output showing:
cloud cover %, number of valid scenes used, spatial resolution, and a plain-English
confidence statement.
**Why:** Every output should communicate its own limitations. Honest and professional.
**Status:** Complete. Deployed.

---

## Arc Plan

### Arc 2 — Corridor and Vegetation Risk

**Goal:** Multi-algorithm vegetation encroachment analysis for utility corridors.
Move beyond a snapshot to a growth trend and prioritised action output.

**Data:** GEE multi-date Sentinel-2 NDVI

**Layer 2 algorithms:**
- K-means clustering for snapshot land cover classification
- Threshold classification for comparison (rule-based vs learned)
- Linear regression per pixel across 6 time steps for growth trend
- Isolation Forest to flag anomalous growth zones

**Layer 3:** Groq synthesises all algorithm outputs into a prioritised risk table:
critical, warning, monitor. Includes inspection priority scoring per segment
(weighted: change magnitude, proximity to asset, persistence, confidence).

**Notebook:** Vegetation encroachment — multi-algorithm corridor analysis
**Portal module:** Corridor Risk Intelligence

**Key concepts:**
- Unsupervised clustering vs rule-based thresholding
- Pixel-level trend analysis over time
- Isolation Forest for anomaly detection without labeled data
- Inspection priority scoring: converting satellite findings into ranked action
- How three algorithms on the same data produce a richer risk picture than one

**Status:** Complete. Notebook and portal module live. Deployed to https://eoil-explorer.streamlit.app

**Corridor alignment note:**
The NC corridor is the only validated bbox. The other four (UK, Australia, South Africa, Tanzania)
are illustrative — geographically plausible for the named region but not traced to verified
transmission line coordinates. For production use, derive bboxes from real line geometry:
OpenStreetMap (power=line via Overpass Turbo), Global Energy Monitor, or utility GIS portals.
The UI and code comments now reflect this distinction.

---

### Arc 3 — Flood Detection and Impact Mapping

**Goal:** Detect flood extent using Sentinel-1 SAR and optical imagery.
Demonstrate that SAR is practically useful, not just educational.

**Data:** Sentinel-1 SAR before/after via GEE, Sentinel-2 water indices, SRTM DEM

**Layer 2 algorithms:**
- SAR backscatter change detection (before vs after)
- NDWI thresholding on Sentinel-2 where cloud-free
- Slope masking from DEM to exclude non-flood areas
- Permanent water masking from Global Surface Water

**Layer 3:** Groq generates impact summary: estimated flood extent, affected area,
infrastructure exposure, recommended response actions.

**Notebook:** Flood detection — SAR and optical fusion
**Portal module:** Flood Intelligence

**Key concepts:**
- Why SAR detects floods through clouds when optical sensors cannot
- Water masking: separating flood water from permanent water bodies
- DEM slope masking: why flat areas flood, steep areas do not
- Fusing two sensor types for better results than either alone

**Status:** Planned

---

### Arc 4 — Burn Severity and Wildfire Analysis

**Goal:** Detect wildfire burn scars and estimate severity using NBR and dNBR.

**Data:** Sentinel-2 and Landsat via Planetary Computer or GEE, FIRMS fire alerts via GEE

**Layer 2 algorithms:**
- NBR: Normalized Burn Ratio from pre-fire scene
- dNBR: Differenced NBR between pre- and post-fire scenes
- Severity classification: unburned, low, moderate, high, very high severity
- FIRMS active fire point overlay for context

**Layer 3:** Groq explains severity classes, estimates recovery timeline,
flags areas requiring field inspection.

**Notebook:** Burn severity — NBR and dNBR on Sentinel-2
**Portal module:** Burn Severity Intelligence
**Note:** NBR added to Spectral Explorer as a preset in the Pre-Arc 2 enhancements.

**Key concepts:**
- What NBR measures and why shortwave infrared responds to fire damage
- Differenced index: why change between two dates matters more than absolute value
- Severity classes and what they mean for vegetation recovery
- FIRMS fire data: what it is, how it is produced, its limitations

**Status:** Planned

---

### Arc 5 — Urban Growth and Load Intelligence

**Goal:** Detect new development and urban expansion as a proxy for utility load growth.

**Data:** Sentinel-2 NDBI change via GEE or Planetary Computer, VIIRS night lights via GEE,
Dynamic World land cover via GEE

**Layer 2 algorithms:**
- NDBI change detection between two dates
- VIIRS night-light trend analysis per location
- Dynamic World land-cover transition classification (from → to class)
- Isolation Forest to flag abnormal growth zones

**Layer 3:** Groq translates geospatial findings into utility planning language:
new-development hotspots, growth rate, candidate areas for infrastructure planning,
load growth implications.

**Notebook:** Urban growth — NDBI change, night lights, Dynamic World
**Portal module:** Urban Growth Intelligence

**Key concepts:**
- NDBI change vs single-date NDBI: why temporal comparison is more reliable
- VIIRS night lights as a proxy for economic activity and load
- Dynamic World: near-real-time land cover from Google
- Translating remote sensing output into infrastructure planning language

**Status:** Planned

---

### Arc 6 — LiDAR Clearance Intelligence

**Goal:** Work with real 3D point cloud data. Derive canopy height, ground model,
and clearance distances. Use AI for precision classification at the point level.

**Data:** USGS 3DEP LiDAR via OpenTopography (real point cloud data, US coverage)

**Layer 2 algorithms:**
- DBSCAN clustering for individual tree crown delineation
- Random Forest on point features (height, intensity, return number) for
  ground vs vegetation vs structure classification
- Height threshold for clearance distance calculation

**Layer 3:** Groq interprets clearance violations by severity. Recommends
inspection priority. Explains what LiDAR sees that satellites cannot.

**Notebook:** LiDAR intelligence — point cloud processing and clearance analysis
**Portal module:** LiDAR Clearance Intelligence

**Key concepts:**
- What LiDAR measures and how point clouds differ from raster imagery
- DBSCAN: density-based clustering, arbitrary cluster shapes, noise handling
- Feature engineering on point clouds: what features matter for classification
- The resolution gap: satellite vs LiDAR vs drone

**Status:** Planned

---

### Arc 7 — Foundation Models

**Goal:** Understand what geospatial foundation models are, how they work,
and when they outperform task-specific models.

**Data:** Sentinel-2 HLS tiles (Harmonised Landsat Sentinel)

**Layer 2 algorithms:**
- Prithvi (IBM/NASA geospatial foundation model) for land cover and change detection
- SAM for fine segmentation on the same scenes
- Compare Prithvi output to K-means from Arc 1 on matching scenes

**Layer 3:** Groq explains what the foundation model found. Compares to classical
methods. Explains when to use a foundation model vs a task-specific classifier.

**Notebook:** Geospatial foundation models — Prithvi and SAM applied to Sentinel-2
**Portal module:** Foundation Model Explorer

**Key concepts:**
- What a foundation model is: pre-trained on massive data, fine-tunable for tasks
- How Prithvi differs from a model trained on one task
- Zero-shot vs fine-tuned performance
- Compute requirements and when foundation models are worth the cost

**Status:** Planned

---

### Arc 8 — Time Series Anomaly Intelligence

**Goal:** Compare three anomaly detection approaches on the same time series data.
Understand when each method is appropriate and why they flag different events.

**Data:** MODIS or Landsat long time series (10+ years) via GEE

**Layer 2 algorithms:**
- Isolation Forest: tree-based, no assumptions about distribution
- Autoencoder on time series windows: learns seasonal pattern, flags deviations
- LSTM: learns temporal dependencies, flags points that break the learned sequence

**Layer 3:** Groq explains why each method flagged different points.
Interprets the likely cause of each anomaly: drought, fire, land use change,
sensor error. Recommends follow-up action per flagged event.

**Notebook:** Time series anomaly detection — Isolation Forest vs Autoencoder vs LSTM
**Portal module:** Time Series Anomaly Intelligence

**Key concepts:**
- Why three methods on the same series produce different results
- What Isolation Forest assumes and where it fails
- What an LSTM learns that a statistical method cannot
- Distinguishing real anomalies from sensor artifacts

**Status:** Planned

---

### Arc 9 — Hyperspectral Intelligence

**Goal:** Work with real hyperspectral data. Understand spectral signatures.
Use dimensionality reduction and clustering to identify materials and anomalies.

**Data:** EMIT hyperspectral scenes via NASA Earthdata (free account required)

**Layer 2 algorithms:**
- PCA for dimensionality reduction across 285 bands to 10 components
- K-means on PCA components for material clustering
- Autoencoder for spectral unmixing and anomaly detection

**Layer 3:** Groq interprets material classifications. Flags unusual spectral
signatures. Explains what each material cluster likely represents and why
hyperspectral data can identify things multispectral cannot.

**Notebook:** Hyperspectral intelligence — EMIT data, PCA, clustering, unmixing
**Portal module:** Hyperspectral Material Intelligence

**Key concepts:**
- The difference between multispectral (10-15 bands) and hyperspectral (200+ bands)
- Why dimensionality reduction is necessary before applying ML to hyperspectral data
- Spectral unmixing: a pixel is rarely one pure material
- What EMIT measures and why it was deployed on the ISS

**Status:** Planned

---

### Arc 10 — Atmospheric Anomaly Intelligence

**Goal:** Extend the existing Emissions Explorer with real anomaly detection
and trend analysis. Move from a visualisation module to an analytical one.

**Data:** TROPOMI via GEE (already integrated in Emissions Explorer)

**Layer 2 algorithms:**
- Isolation Forest on concentration time series per location
- K-means on spatial emission patterns to identify emission clusters
- Linear trend per pixel to flag rising vs falling concentrations

**Layer 3:** Groq interprets anomalous concentration events. Suggests likely
source types. Flags trend direction and rate of change per region.

**Notebook:** Atmospheric anomaly detection — TROPOMI time series analysis
**Portal module:** Upgrade existing Emissions Explorer to add anomaly and trend layers

**Key concepts:**
- Applying anomaly detection to atmospheric concentration data
- Spatial clustering of emission sources
- How TROPOMI detects methane and why the signal requires careful interpretation

**Status:** Planned

---

### Arc 11 — Capstone: Utility Intelligence Brief

**Goal:** Synthesise outputs from all arcs into one decision-ready brief.
Demonstrate the full three-layer architecture operating end to end.
Produce an AI-generated executive report suitable for a utility client.

**Inputs:** Real outputs from Arc 2 (vegetation risk), Arc 3 (flood),
Arc 5 (urban growth), Arc 8 (time series anomalies), Arc 10 (atmospheric).

**Layer 2:** Aggregates risk scores across layers. Weights by severity.
Produces a structured risk summary as a data object, not a narrative.

**Layer 3:** Groq receives the structured risk object and writes a one-page
executive brief. Includes: findings, confidence level, recommended actions,
limitations. Every claim traceable to a Layer 2 output.
Includes CSV and GeoJSON export of all flagged zones.

**Portal module:** Utility Intelligence Brief

**Key concepts:**
- AI as a synthesis layer: ML finds, AI explains, humans decide
- Why the brief is only as strong as the inputs feeding it
- What a decision-ready output looks like vs a data display
- Uncertainty and confidence communication in executive language

**Status:** Planned

---

## Completed Notebooks

| Notebook | Arc | Status |
|----------|-----|--------|
| 02_gee_ndvi_timeseries.ipynb | Day 4 | Complete |
| 03_sentinel1_sar_basics.ipynb | Day 5 | Complete |
| 04_change_detection.ipynb | Day 9 | Complete |
| 06_land_cover_classification.ipynb | Arc 1 | Complete |

---

## Requirements.txt Status

| Group | Status |
|-------|--------|
| Core (streamlit, folium, groq, etc.) | Active |
| scikit-learn | Active |
| Raster (rasterio, rioxarray, xarray, odc-stac) | Commented out |
| GEE (earthengine-api, geemap) | Commented out |
| SAM (segment-anything) | Commented out — activate for Arc 7 |
| LiDAR (laspy, open3d) | Commented out — activate for Arc 6 |
| Hyperspectral (netCDF4, h5py, spectral) | Commented out — activate for Arc 9 |
| Deep learning (torch) | Not yet added — activate for Arc 8 |

---

## Blockers and Notes

- GEE credentials: service account JSON in apps/01_eo_explorer/.streamlit/secrets.toml (local, gitignored)
- Groq API key: console.groq.com
- Gemini API key: aistudio.google.com
- NASA Earthdata account: required for EMIT data in Arc 9. Register free at urs.earthdata.nasa.gov
- ASF account: required for raw Sentinel-1 SLC data. Register free at search.asf.alaska.edu
- OpenTopography: no account required for most USGS 3DEP datasets
- Planetary Computer: intermittent timeouts on free tier. Retry logic added to land_cover.py,
  spectral_explorer.py, and notebook Cell 3.

---

## Backlog

- SAR Explorer: "Port of Rotterdam" geocodes inland. Map picker solves this for manual use.
- Time Series Explorer: tooltip for index definitions still pending.
- Spectral Explorer: B12 band not currently fetched — needed for NBR. Add to fetch list in Pre-Arc 2 work.
- Corridor Risk: four illustrative corridors (UK, Australia, South Africa, Tanzania) need bboxes
  traced to verified transmission line geometry. Sources: OpenStreetMap Overpass Turbo (power=line),
  Global Energy Monitor, individual utility GIS portals. NC corridor is already validated.
- Corridor Risk: free-form region search planned (geocoder text input + map picker). Deferred to future session.

---

## Session Log

- [x] Day 1-8: Portal foundation, spectral analysis, time series, SAR, change detection, week 1 primer
- [x] Day 9: Change Detection portal module
- [x] Day 10: AI Imagery Interpreter portal module
- [x] Day 11: Shared map picker — map_picker.py, all five modules wired
- [x] Day 12: Emissions Explorer — TROPOMI CH4/NO2/CO/SO2 via GEE
- [x] Arc 1: Land Cover Intelligence — notebook + portal module complete (v1.9)
- [x] Backlog review: 25 features assessed, plan revised, arcs 2-11 defined
- [x] Pre-Arc 2: Module enhancements — spectral indices, export layer, confidence notes
- [x] Arc 2: Corridor and Vegetation Risk — notebook + portal module complete, live
- [ ] Arc 3: Flood Detection and Impact Mapping
- [ ] Arc 4: Burn Severity and Wildfire Analysis
- [ ] Arc 5: Urban Growth and Load Intelligence
- [ ] Arc 6: LiDAR Clearance Intelligence
- [ ] Arc 7: Foundation Models
- [ ] Arc 8: Time Series Anomaly Intelligence
- [ ] Arc 9: Hyperspectral Intelligence
- [ ] Arc 10: Atmospheric Anomaly Intelligence
- [ ] Arc 11: Capstone — Utility Intelligence Brief

---

## How to Update This File

At the end of every session:
1. Update Current Status: last completed, next session
2. Update the arc status field for the arc just completed
3. Check off the session log entry
4. Update Requirements.txt Status if any group was activated
5. Add any new blockers or notes
6. Commit PROJECT.md with every session commit
