# PROJECT.md
# AI-Native Earth Observation Innovation Lab
# Current status and next steps
# Update this file at the end of every session

---

## Current Status

**Phase:** Arc-based program (revised after Day 12)
**Last completed:** Arc 1 — Land Cover Intelligence. K-means clustering and Random Forest on Sentinel-2 via Planetary Computer. Notebook 06_land_cover_classification.ipynb complete. Portal module land_cover.py added (v1.9).
**Next session:** Arc 2 — Corridor and Vegetation Risk

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
| Spectral Explorer | Day 3 | Live |
| Time Series Explorer | Day 6 | Live |
| SAR Explorer | Day 7 | Live |
| Change Detection | Day 9 | Live |
| AI Imagery Interpreter | Day 10 | Live |
| Emissions Explorer | Day 12 | Live |
| Land Cover Intelligence | Arc 1 | Live |

---

## Arc Plan

### Arc 1 — Optical Intelligence and Land Cover

**Goal:** Deep understanding of multispectral classification.
Compare unsupervised and supervised approaches on the same scene.
Introduce SAM as a foundation model for segmentation.

**Data:** Sentinel-2 via Planetary Computer or GEE

**Layer 2 algorithms:**
- K-means clustering on NDVI and spectral bands
- Random Forest classifier trained on labeled pixels
- SAM (Segment Anything Model) for object segmentation
- Compare all three outputs on the same scene

**Layer 3:** Groq interprets classification map. Compares methods. Flags discrepancies.

**Notebook:** Land cover classification — K-means vs Random Forest vs SAM
**Portal module:** Land Cover Intelligence

**Key concepts:**
- Supervised vs unsupervised classification
- What a foundation model is and how it differs from task-specific models
- Why three methods on the same data teach more than one

**Status:** Complete — notebook 06_land_cover_classification.ipynb, portal module land_cover.py, app.py v1.9. SAM reserved for Arc 5.

---

### Arc 2 — Corridor and Vegetation Risk

**Goal:** Multi-algorithm vegetation encroachment analysis for utility corridors.
Move beyond a snapshot to a growth trend and prioritised action output.

**Data:** GEE multi-date Sentinel-2 NDVI

**Layer 2 algorithms:**
- K-means clustering for snapshot land cover classification
- Threshold classification for comparison (rule-based vs learned)
- Linear regression per pixel across 6 time steps for growth trend
- Isolation Forest to flag anomalous growth zones

**Layer 3:** Groq synthesises all three algorithm outputs into a prioritised
risk table: critical, warning, monitor. Recommends inspection sequencing.

**Notebook:** Vegetation encroachment — multi-algorithm corridor analysis
**Portal module:** Corridor Risk Intelligence

**Key concepts:**
- Unsupervised clustering vs rule-based thresholding
- Pixel-level trend analysis over time
- Isolation Forest for anomaly detection without labeled data
- How three algorithms on the same data produce a richer risk picture than one

**Status:** Next to build

---

### Arc 3 — Radar and Ground Movement

**Goal:** Understand SAR coherence and use it to detect ground movement.
Introduce autoencoders as a neural network approach to anomaly detection.

**Data:** Sentinel-1 GRD and SLC via ASF Data Search or GEE

**Layer 2 algorithms:**
- PCA on multi-date SAR coherence stack to find change dimensions
- Autoencoder trained on baseline coherence to flag reconstruction errors
- Statistical threshold comparison against autoencoder output

**Layer 3:** Groq interprets coherence loss patterns. Identifies infrastructure
risk zones. Explains what InSAR coherence measures and why it detects subsidence.

**Notebook:** SAR coherence and ground movement — PCA and autoencoder approach
**Portal module:** Ground Movement and Subsidence

**Key concepts:**
- What SAR coherence is and how it measures ground stability
- PCA as a dimensionality reduction tool on multi-date stacks
- What an autoencoder is: compress, reconstruct, flag what reconstructs poorly
- Why neural network anomaly detection differs from statistical thresholding

**Status:** Planned

---

### Arc 4 — LiDAR Intelligence

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

### Arc 5 — Foundation Models

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

### Arc 6 — Time Series and Anomaly Detection

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

### Arc 7 — Hyperspectral Intelligence

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

### Arc 8 — Atmospheric Intelligence

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

### Arc 9 — Capstone: Utility Intelligence Brief

**Goal:** Synthesise outputs from all arcs into one decision-ready brief.
Demonstrate the full three-layer architecture operating end to end.

**Inputs:** Real outputs from Arc 2 (vegetation risk), Arc 3 (ground movement),
Arc 4 (LiDAR clearance), Arc 6 (time series anomalies), Arc 8 (atmospheric).

**Layer 2:** Aggregates risk scores across layers. Weights by severity.
Produces a structured risk summary as a data object, not a narrative.

**Layer 3:** Groq receives the structured risk object and writes a one-page
executive brief. Every claim in the brief is traceable to a Layer 2 output.

**Portal module:** Utility Intelligence Brief

**Key concepts:**
- AI as a synthesis layer: ML finds, AI explains, humans decide
- Why the brief is only as strong as the inputs feeding it
- What a decision-ready output looks like vs a data display

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
| Raster (rasterio, rioxarray, xarray, odc-stac) | Commented out |
| GEE (earthengine-api, geemap) | Commented out |
| SAM (segment-anything) | Commented out — activate for Arc 1 and Arc 5 |
| LiDAR (laspy, open3d) | Commented out — activate for Arc 4 |
| Hyperspectral (netCDF4, h5py, spectral) | Commented out — activate for Arc 7 |
| Deep learning (torch, scikit-learn) | Not yet added — activate for Arc 3 and Arc 6 |

---

## Blockers and Notes

- GEE credentials: service account JSON in apps/01_eo_explorer/.streamlit/secrets.toml (local, gitignored)
- Groq API key: console.groq.com
- Gemini API key: aistudio.google.com
- NASA Earthdata account: required for EMIT data in Arc 7. Register free at urs.earthdata.nasa.gov
- ASF account: required for raw Sentinel-1 SLC data in Arc 3. Register free at search.asf.alaska.edu
- OpenTopography: no account required for most USGS 3DEP datasets

---

## Backlog — Carry Forward

- SAR Explorer: "Port of Rotterdam" geocodes inland. Map picker solves this for manual use.
- Time Series Explorer: EVI, NDWI, Burned Area added Day 8. Tooltip for index definitions still pending.

---

## Session Log

- [x] Day 1-8: Portal foundation, spectral analysis, time series, SAR, change detection, week 1 primer
- [x] Day 9: Change Detection portal module
- [x] Day 10: AI Imagery Interpreter portal module
- [x] Day 11: Shared map picker — map_picker.py, all five modules wired
- [x] Day 12: Emissions Explorer — TROPOMI CH4/NO2/CO/SO2 via GEE
- [x] Arc 1: Land Cover Intelligence — notebook + portal module complete (v1.9)
- [ ] Arc 2: Corridor and Vegetation Risk — next session

---

## How to Update This File

At the end of every session:
1. Update Current Status: last completed, next session
2. Update the arc status field for the arc just completed
3. Check off the session log entry
4. Update Requirements.txt Status if any group was activated
5. Add any new blockers or notes
6. Commit PROJECT.md with every session commit
