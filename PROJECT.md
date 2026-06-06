# PROJECT.md
# AI-Native Earth Observation Innovation Lab
# Current status and next steps
# Update this file at the end of every session

---

## Current Status

**Program day:** 1
**Phase:** Week 1 - Orientation and Data Access
**Last completed:** Repository scaffold and requirements.txt created
**Next session:** Build EO Explorer v1.0

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

| App | Day | URL |
|-----|-----|-----|
| EO Explorer v1.1 | 1-3 | https://eoil-explorer.streamlit.app |
| EO Explorer v1.2 | 6 | _add after deployment_ |
| Change Detection Dashboard | 9 | _add after deployment_ |
| AI Imagery Interpreter | 10 | _add after deployment_ |
| EO Conversational Assistant | 13 | _add after deployment_ |
| Environmental Intelligence Dashboard | 15 | _add after deployment_ |
| Atmospheric Intelligence Dashboard | 17 | _add after deployment_ |
| Multi-Layer Geospatial Dashboard | 20 | _add after deployment_ |
| EO Decision Support Platform | 24 | _add after deployment_ |
| EO Vendor Evaluator | 26 | _add after deployment_ |
| EOIL Curriculum Index | 28 | _add after deployment_ |

---

## Completed Notebooks

Update this section as each notebook is committed.

| Notebook | Day | Status |
|----------|-----|--------|
| 01_stac_query_demo.ipynb | 2 | pending |
| 02_gee_ndvi_timeseries.ipynb | 4 | pending |
| 03_sentinel1_sar_basics.ipynb | 5 | pending |
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

_Add any blockers, decisions, or notes here during the program._

- GEE account: must be activated before Day 4. Request at earthengine.google.com
- Groq API key: get at console.groq.com
- Gemini API key: get at aistudio.google.com
- Mapbox token: get at mapbox.com

---

## Day-by-Day Progress Log

Update the checkbox and add a one-line note after each day is complete.

### Week 1: Orientation and Data Access

- [ ] Day 1: EO Explorer v1.0
- [ ] Day 2: STAC and cloud-native data notebook
- [ ] Day 3: EO Explorer v1.1 with spectral index module
- [ ] Day 4: GEE NDVI time series notebook
- [ ] Day 5: Sentinel-1 SAR basics notebook
- [ ] Day 6: EO Explorer v1.2 with drone imagery module
- [ ] Day 7: Week 1 review and team primer document

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
