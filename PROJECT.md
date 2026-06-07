# PROJECT.md
# AI-Native Earth Observation Innovation Lab
# Current status and next steps
# Update this file at the end of every session

---

## Current Status

**Program day:** 8 complete
**Phase:** Week 1 complete — Week 2 starting next session
**Last completed:** Week 1 primer doc; EVI, NDWI, Burned Area added to Time Series Explorer; Landsat performance warning added; all bugs fixed and deployed
**Next session:** Day 9 — Change Detection Dashboard (apps/02_change_detection/)

---

## Today's Task: Day 1 - EO Explorer v1.0

**Goal:** Build and deploy the foundational application.
**Location:** apps/01_eo_explorer/
**Prompt file:** day1-prompt-03-eo-explorer.md

**Definition of done:**
- [ ] App runs locally: streamlit run apps/01_eo_explorer/app.py
- [ ] Deployed to Streamlit Cloud without errors
- [ ] Interactive Folium map loads with four sample layers
- [ ] All six themes selectable, explanation panel updates
- [ ] AI assistant works in fallback mode with no API key
- [ ] AI assistant works when GROQ_API_KEY is present
- [ ] All six module files have inline comments on every function
- [ ] apps/01_eo_explorer/README.md exists
- [ ] Committed with message: [day-01] add EO Explorer v1.0 foundational app
- [ ] Streamlit Cloud URL added to README.md

---

## Deployed App URLs

Update this section as each app goes live.

All modules are deployed as one portal app at https://eoil-explorer.streamlit.app
Sidebar navigation routes between modules. One URL for everything.

| Module | Added Day | Status |
|--------|-----------|--------|
| Welcome Panel | 7 | Live |
| Spectral Explorer | 3 | Live |
| Time Series Explorer | 6 | Live |
| SAR Explorer | 7 | Live |
| EO Explorer v1.2 (drone) | future | Planned |
| Change Detection | 9 | Planned |
| AI Imagery Interpreter | 10 | Planned |
| EO Conversational Assistant | 13 | Planned |
| Environmental Intelligence | 15 | Planned |
| Atmospheric Intelligence | 17 | Planned |
| Multi-Layer Geospatial | 20 | Planned |
| EO Decision Support Platform | 24 | _add after deployment_ |
| EO Vendor Evaluator | 26 | _add after deployment_ |
| EOIL Curriculum Index | 28 | _add after deployment_ |

---

## Completed Notebooks

Update this section as each notebook is committed.

| Notebook | Day | Status |
|----------|-----|--------|
| 01_stac_query_demo.ipynb | 2 | pending |
| 02_gee_ndvi_timeseries.ipynb | 4 | complete |
| 03_sentinel1_sar_basics.ipynb | 5 | complete |
| 04_land_cover_classification.ipynb | 8 | pending |
| 05_sam_segmentation_demo.ipynb | 11 | pending |
| 06_prithvi_foundation_model.ipynb | 12 | pending |
| 07_sar_infrastructure_monitoring.ipynb | 16 | pending |
| 08_lidar_intelligence.ipynb | 18 | pending |
| 09_emit_hyperspectral.ipynb | 19 | pending |

---

## Requirements.txt Status

Track which dependency groups are active.

| Group | Days | Status |
|-------|------|--------|
| Core (streamlit, folium, groq, etc.) | Day 1+ | active |
| Raster (rasterio, rioxarray, xarray, odc-stac) | Day 3 | commented out |
| GEE (earthengine-api, geemap) | Day 4 | commented out |
| SAM (segment-anything) | Day 11 | commented out |
| LiDAR (laspy, open3d) | Day 18 | commented out |
| Hyperspectral (netCDF4, h5py, spectral) | Day 19 | commented out |

---

## Blockers and Notes

- GEE account: must be activated before Day 4. Request at earthengine.google.com
- Groq API key: get at console.groq.com
- Gemini API key: get at aistudio.google.com
- Mapbox token: get at mapbox.com
- GEE credentials: service account JSON stored in apps/01_eo_explorer/.streamlit/secrets.toml (local only, gitignored). For Streamlit Cloud deployment, add GEE_SERVICE_ACCOUNT_JSON to the app's secrets in the Streamlit Cloud dashboard.

---

## Backlog — SAR Explorer improvements

- **Map picker for location** (Option D): Replace text geocoding with a small Folium map the user clicks to set the centre point. Build the bbox around the click. Eliminates all geocoding ambiguity for port and industrial feature queries. Text geocoding stays as a fallback for users who want to type a city name first, then fine-tune on the map.
- Known bad geocodes: "Port of Rotterdam" resolves to an inland agricultural area. "Maasvlakte Rotterdam" or coordinates work correctly.

---

## Backlog — Time Series Explorer improvements (carry into Day 7+)

These were identified during Day 6 testing. Do not forget.

**Additional indices to add to the Dataset dropdown:**
- EVI (Enhanced Vegetation Index) — already in MODIS/061/MOD13Q1, same collection as NDVI. Just add a new entry to DATASETS dict in gee_timeseries.py. Easiest add.
- NDWI (Normalized Difference Water Index) — requires Landsat band math (NIR minus SWIR divided by NIR plus SWIR). Measures water content and wetness.
- Burned Area — MODIS MCD64A1 collection, pre-computed. Shows fire extent over time.
- Plan: add EVI first (10 min), then NDWI and Burned Area in a later session.

**Other Time Series improvements noted:**
- The GIBS tile overlay on the old comparison maps was not rendering visibly. Replaced with Annual Comparison bar chart (done Day 6). No further action needed.
- Consider adding a "What does this index measure?" tooltip or expander next to the dataset selector so users understand what they are looking at before running analysis.

---

## Day-by-Day Progress Log

Update the checkbox and add a one-line note after each day is complete.

### Week 1: Orientation and Data Access

- [ ] Day 1: EO Explorer v1.0
- [ ] Day 2: STAC and cloud-native data notebook
- [ ] Day 3: EO Explorer v1.1 with spectral index module
- [x] Day 4: GEE NDVI time series notebook — Sahel 10-year NDVI, 253 data points, trend +0.0034/year
- [x] Day 5: Sentinel-1 SAR basics notebook — Rotterdam port, VV/VH/false color, backscatter change map
- [x] Day 6: GEE Time Series Explorer portal module — MODIS NDVI, LST; Annual Comparison chart; GEE credentials
- [x] Day 7: SAR Explorer module — Sentinel-1 VV/VH/false color/change map; Welcome panel; bbox size warning
- [x] Day 8: Week 1 primer doc; EVI, NDWI, Burned Area added to Time Series Explorer; Landsat performance warning

### Week 2: Core Analysis and AI Integration

- [ ] Day 8: Land cover classification notebook
- [ ] Day 9: Change Detection Dashboard app
- [ ] Day 10: AI Imagery Interpreter app
- [ ] Day 11: SAM segmentation notebook
- [ ] Day 12: Prithvi foundation model notebook (Colab)
- [ ] Day 13: EO Conversational Assistant app
- [ ] Day 14: Week 2 review and AI-applied-to-EO document

### Week 3: Specialized Domains

- [ ] Day 15: Environmental Intelligence Dashboard app
- [ ] Day 16: SAR infrastructure monitoring notebook
- [ ] Day 17: Atmospheric Intelligence Dashboard app
- [ ] Day 18: LiDAR intelligence notebook
- [ ] Day 19: EMIT hyperspectral notebook
- [ ] Day 20: Multi-Layer Geospatial Dashboard app
- [ ] Day 21: Week 3 review and four domain briefs

### Week 4: Capstone Integration

- [ ] Day 22: Decision Support Platform Panel 1
- [ ] Day 23: Decision Support Platform Panel 2
- [ ] Day 24: Decision Support Platform Panel 3 + deploy
- [ ] Day 25: EO Vendor Evaluator data layer
- [ ] Day 26: EO Vendor Evaluator app + deploy
- [ ] Day 27: Curriculum Index repository cleanup
- [ ] Day 28: EOIL Curriculum Index app + deploy
- [ ] Day 29: Conference presentation outline
- [ ] Day 30: Gap assessment and cycle 2 planning

---

## How to Update This File

At the end of every session, update:
1. Current Status section: change the day number and last completed item
2. Today's Task section: replace with the next day's task and definition of done
3. Deployed App URLs: add the Streamlit Cloud URL after each deployment
4. Completed Notebooks: mark status as complete after each commit
5. Requirements.txt Status: mark groups as active when uncommented
6. Progress Log: check the completed day, add a one-line note

Commit PROJECT.md with every session commit.


