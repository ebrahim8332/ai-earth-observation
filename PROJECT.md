# PROJECT.md
# AI-Native Earth Observation Innovation Lab
# Current status and next steps
# Update this file at the end of every session

---

## Current Status

**Program day:** 12 complete
**Phase:** Week 2 in progress
**Last completed:** Day 12 — Emissions Explorer portal module. TROPOMI Sentinel-5P data via GEE for CH4, NO2, CO, SO2. 7-day composite mosaic. Point-sampling stats bypass. Folium map with scaled colorbar. AI interpretation via ai_chain. Colorbar labels fixed (display_scale per gas). Overlay opacity tuned to 0.55.
**Next session:** Day 13 — Vegetation Encroachment Detection

---

## Completed: Day 11 — Shared Map Picker

- [x] map_picker.py created — shared component, ~160 lines
- [x] render_map_picker(centre_bbox, picker_key, default_size_km) — single public function
- [x] CartoDB Positron basemap (no API key needed)
- [x] Auto zoom level from geocoded bbox width
- [x] Size slider: 25 to 500 km, step 25
- [x] Blue rectangle drawn on map showing exact analysis area
- [x] CircleMarker at click centre point
- [x] clear_click(picker_key) helper — called when user types a new location
- [x] Wired into all five modules as expander: "📍 Refine location — click map to set exact area"
- [x] Default sizes: Spectral 50 km, Time Series 100 km, SAR 50 km, Change Detection 100 km, Imagery 50 km
- [x] Portal bumped to v1.7, deployed and confirmed clean startup
- [x] Backlog item "Map picker for location (Option D)" cleared
- [x] Render clip fix: _clip_to_bbox() helper added to spectral_explorer.py — local pixel crop using item.bbox so rendered images respect the map picker selection. PC Render API /preview always used; clip applied locally after fetch.
- [x] _pad_to_ratio() helper added — prevents long/wide image on large radius selections by padding to 2:1 max aspect ratio
- [x] _crop_to_valid() aspect ratio guard added — prevents thin-strip thumbnails when tile only marginally overlaps search area

## Completed: Day 10 — AI Imagery Interpreter

- [x] imagery_interpreter.py created — 7-model vision chain (5 Gemini + 2 Groq Llama-4)
- [x] fetch_chip(bbox, date_start, date_end) — searches user-set date range, picks best scene
- [x] array_to_jpeg_bytes() — numpy to JPEG at quality 85
- [x] interpret_image() — returns (text, model_name) tuple
- [x] max_output_tokens=4096 for Gemini (thinking tokens reduce output budget)
- [x] Date range inputs (start + end) — no hidden search window
- [x] Area size guard: warn >25,000 km², block >100,000 km²
- [x] Wired into app.py as sixth sidebar entry (v1.6)
- [x] Confirmed working on Zanzibar, Tanzania

## Completed: Day 9 — Change Detection Module

- [x] gee_change.py created with GEE extraction and Folium map builder
- [x] Module wired into app.py sidebar as fifth entry
- [x] NDVI difference computed (Sentinel-2 with MODIS fallback)
- [x] Three-layer change map: NDVI Date 1, NDVI Date 2, NDVI diff (default)
- [x] 12 summary statistics: mean/std change, NDVI baselines, area gain/loss/stable, net change, gain/loss ratio, extreme gain/loss
- [x] ai_chain.py created — 11-model fallback chain (6 Gemini + 5 Groq), session locking
- [x] ArcGIS geocoder replaces Nominatim as primary — reliable on shared cloud IPs
- [x] Grouped GEE reducer pattern — 2 round-trips instead of 5, STATS_SCALE 1000m
- [x] AI interpretation returns (text, model_name) tuple; model shown below response
- [x] Substantive fallback if all AI models unavailable
- [x] gee_timeseries.py and gee_sar.py updated to use ai_chain
- [x] notebooks/04_change_detection.ipynb committed
- [x] Portal bumped to v1.5
- [x] Confirmed working on Zanzibar, Tanzania and Sierra Nevada, California

---

## Deployed App URLs

Update this section as each app goes live.

All modules are deployed as one portal app at https://eoil-explorer.streamlit.app
Sidebar navigation routes between modules. One URL for everything.
No standalone apps. Every deliverable is a new sidebar module in the portal.

| Module | Added Day | Status |
|--------|-----------|--------|
| Welcome Panel | 7 | Live |
| Spectral Explorer | 3 | Live |
| Time Series Explorer | 6 | Live |
| SAR Explorer | 7 | Live |
| Change Detection | 9 | Live |
| AI Imagery Interpreter | 10 | Live |
| Emissions Explorer | 12 | Live |
| Vegetation Encroachment | 13 | Planned |
| Ground Movement / Subsidence | 14 | Planned |
| LiDAR for Infrastructure | 15 | Planned |
| Drone Imagery and AI Defect Detection | 16 | Planned |
| Anomaly Detection | 17 | Planned |
| Commercial Market Map | 18 | Planned |
| Capstone: Utility Intelligence Brief | 19 | Planned |

---

## Completed Notebooks

Update this section as each notebook is committed.

| Notebook | Day | Status |
|----------|-----|--------|
| 01_stac_query_demo.ipynb | 2 | pending |
| 02_gee_ndvi_timeseries.ipynb | 4 | complete |
| 03_sentinel1_sar_basics.ipynb | 5 | complete |
| 04_change_detection.ipynb | 9 | complete |
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

- Known bad geocodes: "Port of Rotterdam" resolves to an inland agricultural area. "Maasvlakte Rotterdam" or coordinates work correctly. The map picker (Day 11) now solves this — type the city, then click the exact port area on the map.

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

### Revised Plan (agreed Day 12 session)

**Guiding principle:** Option A throughout. Conceptual demonstration, not deep implementation.
No real operational data required. Every module must be understandable to a non-specialist
and demonstrable in a 10-minute conversation. AI present in every module — extracting meaning,
flagging anomalies, or explaining what an algorithm found.

- [x] Day 9: Change Detection portal module — NDVI difference map, three-layer Folium change overlay, AI interpretation
- [x] Day 10: AI Imagery Interpreter portal module — Sentinel-2 chip to Gemini vision, structured interpretation
- [x] Day 11: Shared map picker (map_picker.py) — all five modules; render clip fix in spectral_explorer.py
- [x] Day 12: Emissions Explorer — TROPOMI CH4/NO2/CO/SO2 via GEE, 7-day composite, colorbar, AI interpretation
- [ ] Day 13: Vegetation Encroachment Detection — NDVI threshold concept, right-of-way monitoring, electric T&D relevance
- [ ] Day 14: Ground Movement and Subsidence — InSAR/Sentinel-1 coherence concept, pipeline and underground infrastructure
- [ ] Day 15: LiDAR for Infrastructure — canopy height models, clearance distance, 3D point cloud basics, vegetation management
- [ ] Day 16: Drone Imagery and AI Defect Detection — high-resolution inspection, computer vision defect classification, asset management
- [ ] Day 17: Anomaly Detection — unsupervised ML on time series, flagging deviations without labeled data
- [ ] Day 18: Commercial Market Map — Planet, Capella, ICEYE, Maxar, GHGSat, Carbon Mapper reference document
- [ ] Day 19: Capstone — Utility Intelligence Brief, one integrated portal view for a utility audience

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


