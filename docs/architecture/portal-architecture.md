# EOIL Portal — Architecture Guide
# AI-Native Earth Observation Innovation Lab
# Version: v1.8 (Day 12)
# Last updated: June 2026

---

## Executive Summary

The EOIL Portal is a single-URL web application that demonstrates how satellite,
aerial, and atmospheric data can be combined with AI to produce analytical insights.
It is built for education and demonstration — not for production operations.

The portal lives at https://eoil-explorer.streamlit.app. All seven analytical modules
sit inside one application. The user navigates between them via a sidebar. There is
no separate login, no database, and no stored user data.

This document explains how the portal is structured, what each file does, what data
sources are called, what accounts and API keys are required, and what each module
takes as input and produces as output.

---

## AI Architecture — Two-Layer Rule (applies from Day 12 onward)

Every module from Day 12 onward implements two distinct AI layers. Both are required.

**Layer 1 — ML / Algorithm layer**
Classical machine learning or deep learning finds the pattern in the data.
Examples: K-means clustering, Isolation Forest, PCA, DBSCAN, CNN classification.
This layer computes. It does not describe. It finds structure the human eye cannot
see at scale.

**Layer 2 — Generative AI layer**
Groq or Gemini explains what the ML layer found. It interprets, contextualises,
and recommends action. It never substitutes for real analysis. It always receives
structured inputs from Layer 1.

**UI rule:** The two layers are visually distinct in every module. Users can see
what the algorithm found and what the AI said about it separately.

**Why both layers.**
Generative AI alone describes but does not find. ML alone finds but does not explain.
Together they produce the correct architecture: find the pattern, explain what it means.
This reflects how real production EO workflows operate.

---

## The Big Picture — How a User Action Flows Through the System

Every module follows the same basic flow:

```
User types a location
        ↓
Geocoder converts the place name to a bounding box (geographic coordinates)
        ↓
Map picker (optional) — user clicks the map to refine the exact area
        ↓
Data source is queried (Planetary Computer or Google Earth Engine)
        ↓
Data is fetched and processed (imagery rendered, index computed, etc.)
        ↓
Result is displayed (image, chart, or map)
        ↓
AI interprets the result in plain language
        ↓
User reads the output
```

The application does not store anything between sessions. Every run starts fresh.
Session state (clicked locations, rendered images, AI responses) lives only in the
browser tab and disappears when the page is closed.

---

## Accounts and API Keys Required

This table shows every external service the portal uses, whether it is free,
and what is needed to access it.

| Service | What it provides | Free? | What you need |
|---|---|---|---|
| Planetary Computer | Sentinel-2 and Landsat satellite imagery | Free, no account needed | Nothing — open API |
| Google Earth Engine | NDVI, SAR, change detection, time series | Free for research and education | GEE account + service account JSON |
| ArcGIS Geocoding | Converts place names to coordinates | Free (rate limited) | Nothing — open API |
| Groq | Text AI — interprets analysis results | Free tier available | GROQ_API_KEY from console.groq.com |
| Gemini | Text AI + vision AI — first choice in chain | Free tier available | GEMINI_API_KEY from aistudio.google.com |
| Streamlit Cloud | Hosts the deployed portal | Free tier | GitHub account + Streamlit Cloud account |
| GitHub | Stores the deploy copy of the code | Free | GitHub account |

**What happens with no API keys**

The portal works without any API keys. Planetary Computer and ArcGIS geocoding
require no keys. GEE requires a service account. Without Groq or Gemini keys,
every module shows a substantive fallback interpretation instead of an AI-generated
one. The fallback text is pre-written and genuinely useful — it is not a placeholder.

**GEE service account — the only complex setup**

Google Earth Engine requires a service account (a technical user credential, not a
personal login). The credential is a JSON file stored at:

```
apps/01_eo_explorer/.streamlit/secrets.toml
```

This file is excluded from GitHub. On Streamlit Cloud, the same credential is
entered manually in the app's Secrets dashboard. The service account email is:

```
eoil-gee-service@gen-lang-client-0093165324.iam.gserviceaccount.com
```

Without this, the Time Series Explorer, SAR Explorer, and Change Detection modules
will not run. The Spectral Explorer and AI Imagery Interpreter are not affected —
they use Planetary Computer, which needs no account.

---

## The Two-Repo Deploy System

The code lives in two separate folders. This separation keeps the development
environment clean and the deployed version lean.

| Folder | Purpose | On GitHub? |
|---|---|---|
| Desktop\code\ai-earth-observation\ | Development source — full project including notebooks, docs, experiments | No — never pushed |
| Desktop\code\ai-earth-observation-feedback\ | Deploy copy — only the files Streamlit Cloud needs to run | Yes — pushed to GitHub |

**Critical rule:** Streamlit Cloud is configured to run the file at
`apps/01_eo_explorer/app.py` inside the feedback repo. It does NOT run a file
at the repo root. Every file must be copied to `apps/01_eo_explorer/` — never
to the repo root.

**Deploy procedure every time a change is made:**

```powershell
# Copy the changed files
Copy-Item "...ai-earth-observation\apps\01_eo_explorer\[filename]" `
          "...ai-earth-observation-feedback\apps\01_eo_explorer\[filename]"

# Push from the feedback folder
cd "...ai-earth-observation-feedback"
git add .
git commit -m "[day-XX] description"
git push
```

Streamlit Cloud detects the push and redeploys automatically within about one minute.

---

## File Map — Every File and What It Does

All files live inside `apps/01_eo_explorer/`.

### Core application

| File | What it does | Called by |
|---|---|---|
| `app.py` | Portal entry point. Sidebar navigation, module routing, page layout. No business logic — it wires everything together. | Streamlit Cloud (entry point) |
| `config.py` | Loads API keys from environment variables. Every other module imports from here. Provides `has_groq()`, `has_gemini()`, `has_any_key()` helper functions. | All modules |

### AI

| File | What it does | Called by |
|---|---|---|
| `ai_chain.py` | 10-model text AI fallback chain. Tries Gemini models first (5 models), then Groq models (5 models). Locks to the first model that succeeds and reuses it for the session. Returns `(text, model_name)` tuple. | `gee_timeseries.py`, `gee_sar.py`, `gee_change.py`, `ai_assistant.py` |
| `ai_assistant.py` | Older AI wrapper. Provides `ask()` and `auto_explain()` functions used by the Spectral Explorer. Will be unified with ai_chain.py in a future session. | `spectral_explorer.py`, `app.py` |
| `imagery_interpreter.py` | 7-model vision AI chain (5 Gemini + 2 Groq Llama-4). Sends a satellite image as JPEG bytes to a vision model and returns a plain-language interpretation. Used exclusively by the AI Imagery Interpreter module. | `app.py` |

### Data fetching and processing

| File | What it does | Called by |
|---|---|---|
| `spectral_explorer.py` | Searches Planetary Computer for Sentinel-2 and Landsat scenes. Renders any band combination as an RGB image using the PC Rendering API. Computes NDVI and NDWI renders. Clips renders to the user's selected bbox. | `app.py`, `imagery_interpreter.py` |
| `gee_timeseries.py` | Queries Google Earth Engine for NDVI, EVI, LST, NDWI, and Burned Area time series over a user-defined region. Returns a Plotly chart and an annual comparison chart. | `app.py` |
| `gee_sar.py` | Queries Google Earth Engine for Sentinel-1 SAR imagery (VV, VH, false color, change map). Returns a Folium map with toggleable layers. | `app.py` |
| `gee_change.py` | Queries Google Earth Engine for NDVI difference between two dates. Returns a three-layer Folium change map and 12 summary statistics. | `app.py` |
| `methane_explorer.py` | Fetches TROPOMI Sentinel-5P atmospheric gas data from GEE for CH4, NO2, CO, SO2. 7-day composite mosaic. Point-sampling stats. Folium map with scaled colorbar. AI interpretation via ai_chain. | `app.py` |
| `geocoder.py` | Converts a place name to a bounding box using the ArcGIS geocoding API. Returns `[min_lon, min_lat, max_lon, max_lat]`. Caches results in session state to avoid repeat calls. | `app.py` |

### Supporting files

| File | What it does | Called by |
|---|---|---|
| `map_picker.py` | Shared click-to-set-location widget. Renders a Folium map centred on the geocoded location. User clicks to place an analysis bbox. Returns the clicked bbox or None. | `app.py` (all five analytical modules) |
| `satellite_catalog.py` | Static metadata for all supported satellites. Band names, wavelengths, descriptions, preset band combinations, rescale values. No API calls — pure data. | `spectral_explorer.py`, `app.py` |
| `map_builder.py` | Builds base Folium map objects. Used in early portal versions — partially superseded by per-module map building in gee_sar.py and gee_change.py. | `app.py` |
| `data_catalog.py` | Static metadata for datasets and themes. Used in the Welcome panel. | `app.py` |
| `sample_layers.py` | Hardcoded sample GeoJSON for fallback map layers. Used when live data is unavailable. | `app.py` |

---

## Module Reference — Inputs, Processing, Outputs

### 1. Welcome Panel

**What question it answers:** What is this portal and what can I do here?

**User provides:** Nothing. Passive landing page.

**Data sources:** None.

**Processing:** Static content display. Module descriptions, technology stack, how-to-start guide.

**Output:** Informational panels. No analysis.

**AI:** Not used.

**Accounts needed:** None.

---

### 2. Spectral Explorer

**What question it answers:** What does this location look like from space, and what do different band combinations reveal?

**User provides:**
- Location (place name — e.g. "Rotterdam" or "Zanzibar")
- Satellite (Sentinel-2 L2A or Landsat 8/9)
- Date range
- Max cloud cover percentage
- Band combination (preset or custom R/G/B selection)
- Optional: map picker click to set exact analysis area and size

**Data sources:**
- ArcGIS geocoding API (place name → coordinates) — free, no account
- Planetary Computer STAC API (scene search) — free, no account
- Planetary Computer Rendering API (image render) — free, no account

**Processing:**
1. Geocoder resolves place name to bounding box
2. Map picker (optional) refines bbox via map click
3. STAC search finds scenes matching location, date range, cloud cover
4. Best scene selected by coverage scoring (renders a small test image per candidate)
5. Full image rendered via PC Rendering API for the selected band combination
6. Image clipped locally to the user's bbox using item geographic extent
7. NDVI and NDWI optional renders available

**Output:**
- Rendered satellite image (true color, false color, or custom combination)
- Scene metadata (date, cloud cover, satellite, band descriptions)
- NDVI/NDWI renders on request
- Contact sheet: all preset combinations rendered as thumbnails with channel labels
- Download button for the rendered image

**AI:** Optional. "AI: Explain this view" button calls the text AI chain to describe what the band combination reveals for the location. Requires Groq or Gemini key.

**Accounts needed:** None required. Groq or Gemini key for AI explanations (optional).

---

### 3. Time Series Explorer

**What question it answers:** How has this location changed over months or years — vegetation, temperature, water, or fire?

**User provides:**
- Location (place name)
- Dataset (NDVI, EVI, LST, NDWI, Burned Area)
- Date range (start and end year/month)
- Optional: map picker to set analysis area

**Data sources:**
- ArcGIS geocoding — free, no account
- Google Earth Engine — free for research, requires GEE service account
  - MODIS MOD13Q1 (NDVI, EVI) — 250m resolution, 16-day composites
  - MODIS MOD11A2 (LST) — land surface temperature
  - Landsat 8/9 (NDWI) — 30m resolution
  - MODIS MCD64A1 (Burned Area) — monthly fire extent

**Processing:**
1. GEE queries the selected dataset over the location and date range
2. Time series values extracted as monthly or 16-day data points
3. Plotly line chart built from the time series
4. Annual comparison bar chart built from yearly averages
5. AI interprets the trend

**Output:**
- Interactive time series chart (zoom, hover for values)
- Annual comparison bar chart
- AI interpretation of the trend (what is causing it, what it means)
- Model name shown below AI response

**AI:** Always attempted. Uses ai_chain.py (10-model fallback). If all models fail, shows a pre-written substantive fallback explaining how to read the index.

**Accounts needed:** GEE service account (required). Groq or Gemini key for AI (optional but recommended).

---

### 4. Change Detection

**What question it answers:** What changed between two dates — where did vegetation increase or decrease?

**User provides:**
- Location (place name)
- Date 1 and Date 2 (two separate dates to compare)
- Optional: map picker to set analysis area

**Data sources:**
- ArcGIS geocoding — free, no account
- Google Earth Engine — requires GEE service account
  - Sentinel-2 SR (primary) — 10m resolution optical
  - MODIS MOD13Q1 (fallback if Sentinel-2 unavailable) — 250m

**Processing:**
1. GEE fetches NDVI for both dates over the location
2. NDVI difference computed: Date 2 minus Date 1
3. Positive difference = vegetation gain (green). Negative = loss (red).
4. Three-layer Folium map built: NDVI Date 1, NDVI Date 2, NDVI difference
5. 12 summary statistics computed: mean change, area gained/lost, gain/loss ratio, etc.
6. AI interprets the change pattern

**Output:**
- Interactive three-layer Folium map (toggle between views)
- 12 summary statistics in a metrics panel
- AI interpretation of what caused the change and what it means
- Model name shown below AI response

**AI:** Always attempted. Uses ai_chain.py. Substantive fallback if all models fail.

**Accounts needed:** GEE service account (required). Groq or Gemini key for AI (optional but recommended).

---

### 5. AI Imagery Interpreter

**What question it answers:** What does this location look like right now, and what does a vision AI see in the image?

**User provides:**
- Location (place name)
- Date range (start and end date)
- Max cloud cover percentage
- Optional: map picker to set exact area and size (25–500 km)

**Data sources:**
- ArcGIS geocoding — free, no account
- Planetary Computer STAC API (scene search) — free, no account
- Planetary Computer (image fetch) — free, no account

**Processing:**
1. Geocoder resolves location
2. Map picker (optional) refines bbox
3. Area size guard: warns if area exceeds 25,000 km², blocks if over 100,000 km²
4. STAC search finds best Sentinel-2 true-color scene (lowest cloud, best coverage)
5. Scene rendered as true-color JPEG (B04 Red, B03 Green, B02 Blue) — clipped to bbox
6. JPEG bytes sent to vision AI chain (7 models)
7. Vision AI describes land cover, notable features, seasonal state, and decision applications

**Output:**
- True-color satellite image displayed in the portal
- Scene metadata (date, cloud cover, satellite, band descriptions)
- AI interpretation (3–4 paragraphs)
- Model name shown below interpretation

**AI:** Central to this module. Uses imagery_interpreter.py vision chain (5 Gemini + 2 Groq Llama-4). Requires Gemini or Groq key. If no vision model is available, shows a reading guide for interpreting true-color imagery manually.

**Accounts needed:** None for image fetch. Gemini key strongly recommended (Gemini models perform significantly better on image interpretation than Groq Llama-4).

---

### 6. Emissions Explorer

**What question it answers:** What atmospheric gases are elevated over this region, and what does that mean for infrastructure operators?

**User provides:**
- Location (place name)
- Gas (CH4, NO2, CO, SO2)
- Target date (module uses a 7-day window centred on this date)

**Data sources:**
- ArcGIS geocoding — free, no account
- Google Earth Engine — requires GEE service account
  - COPERNICUS/S5P/OFFL/L3_CH4 (Methane)
  - COPERNICUS/S5P/OFFL/L3_NO2 (Nitrogen Dioxide)
  - COPERNICUS/S5P/OFFL/L3_CO (Carbon Monoxide)
  - COPERNICUS/S5P/OFFL/L3_SO2 (Sulfur Dioxide)

**Processing:**
1. GEE fetches all TROPOMI orbital passes within a 7-day window (approx. 98 passes)
2. Composite mean image computed across all passes for solid coverage
3. Regional mean concentration sampled via 3×3 point grid (bypasses projection issues)
4. Folium map built with GEE tile layer at 0.55 opacity over CartoDB Positron base
5. Colorbar values scaled to readable units per gas (see display_scale in GAS_CONFIG)
6. AI interprets the concentration level, likely sources, regulatory context, and utility action

**Output:**
- Regional average concentration (scaled display unit)
- 7-day composite date window and orbital pass count
- Interactive Folium map with gas concentration overlay
- AI interpretation (4 sections: Pattern, Source Attribution, Regulatory Context, Utility Action)
- Model name shown below AI response

**AI:** Always attempted. Uses ai_chain.py. Substantive fallback if all models fail.

**Accounts needed:** GEE service account (required). Groq or Gemini key for AI (optional but recommended).

**Two-layer note:** This module is primarily a Layer 2 (Generative AI interpretation) module.
The ML clustering layer is introduced from Day 13 onward.

---

### 7. SAR Explorer

**What question it answers:** What does radar see at this location — and where has backscatter changed?

**User provides:**
- Location (place name)
- Date range
- Optional: map picker to set analysis area

**Data sources:**
- ArcGIS geocoding — free, no account
- Google Earth Engine — requires GEE service account
  - Sentinel-1 GRD (Ground Range Detected) — C-band radar, free, global

**Processing:**
1. GEE fetches Sentinel-1 imagery for the location and date range
2. Four layers computed:
   - VV polarization (good for urban areas and open water)
   - VH polarization (good for vegetation structure)
   - False color composite (RGB = VV, VH, VV/VH ratio)
   - Change map (difference between earliest and latest scene in range)
3. Folium map built with all four layers as toggleable overlays
4. AI interprets what the radar is showing

**Output:**
- Interactive four-layer Folium map
- AI interpretation of the radar signature and change pattern
- Model name shown below AI response
- Warning shown if selected area exceeds recommended size for SAR analysis

**AI:** Always attempted. Uses ai_chain.py. Substantive fallback explains how to read SAR imagery.

**Accounts needed:** GEE service account (required). Groq or Gemini key for AI (optional but recommended).

---

## The AI Chains — How the Model Fallback Works

### Text chain (used by 4 modules)

File: `ai_chain.py`

Ten models tried in order, best quality first:

```
1.  gemini-2.5-flash        (Google Gemini — best free tier quality)
2.  gemini-2.0-flash        (Google Gemini)
3.  gemini-2.0-flash-lite   (Google Gemini)
4.  gemini-2.5-flash-lite   (Google Gemini)
5.  gemini-flash-latest     (Google Gemini — alias)
6.  llama-3.3-70b           (Groq)
7.  llama-4-scout-17b       (Groq)
8.  qwen3-32b               (Groq)
9.  gpt-oss-120b            (Groq)
10. llama-3.1-8b-instant    (Groq — last resort)
```

Session locking: once a model responds successfully, the chain locks to that model
for the rest of the browser session. This avoids redundant retries on every request.
If the locked model fails later (rate limit or outage), the chain continues from
that point and re-locks to the next working model.

### Vision chain (used by AI Imagery Interpreter only)

File: `imagery_interpreter.py`

Seven multimodal-capable models only. Text-only models excluded.

```
1.  gemini-2.5-flash        (Google Gemini)
2.  gemini-2.0-flash        (Google Gemini)
3.  gemini-2.0-flash-lite   (Google Gemini)
4.  gemini-2.5-flash-lite   (Google Gemini)
5.  gemini-flash-latest     (Google Gemini)
6.  llama-4-scout-17b       (Groq — Llama 4 multimodal)
7.  llama-4-maverick        (Groq — Llama 4 multimodal)
```

Separate session lock from the text chain — the two chains do not interfere.

### Graceful degradation

Every module has a pre-written fallback response. If all AI models are unavailable
(no keys, rate limits, outages), the fallback text is shown instead. The fallback
is substantive — it explains how to read the output manually, what the index means,
what to look for. It is never a placeholder or an error message.

---

## Data Sources — What Is Free and What Is Not

### Planetary Computer (Microsoft)

- **URL:** planetarycomputer.microsoft.com
- **Cost:** Free. No account required. No API key.
- **What it provides:** Sentinel-2 L2A (10m optical), Landsat 8/9 (30m optical), Copernicus DEM, and many other datasets via STAC (SpatioTemporal Asset Catalog).
- **Limitations:** Rate limited. Large area requests or many concurrent renders may slow down or time out. Not intended for operational production use.
- **Used by:** Spectral Explorer, AI Imagery Interpreter.

### Google Earth Engine (Google)

- **URL:** earthengine.google.com
- **Cost:** Free for research, education, and non-commercial use. Commercial use requires a paid licence.
- **What it provides:** Cloud-scale geospatial analysis. MODIS, Sentinel-1, Sentinel-2, Landsat, and many other datasets. Processes data on Google's servers — you send a query, it returns results.
- **Account required:** Yes. Personal GEE account plus a service account with a JSON credential file.
- **Limitations:** Computation quotas apply. Complex analyses over large areas or long time ranges may time out. The portal uses simplified reducer patterns to stay within limits.
- **Used by:** Time Series Explorer, SAR Explorer, Change Detection.

### ArcGIS Geocoding (Esri)

- **URL:** geocode.arcgis.com
- **Cost:** Free at the usage levels this portal generates. No account required.
- **What it provides:** Converts a place name or address to geographic coordinates (bounding box).
- **Limitations:** Rate limited. Unusual place names or very specific geographic features may not resolve correctly. The map picker was built specifically to handle cases where geocoding returns an imprecise result.
- **Used by:** All five analytical modules.

### Groq

- **URL:** console.groq.com
- **Cost:** Free tier available. Rate limits apply (tokens per day, requests per minute). Paid tiers available for higher throughput.
- **What it provides:** Fast inference on open-source language models (Llama, Qwen, etc.).
- **Account required:** Yes. Sign up at console.groq.com. Generate an API key.
- **Used by:** Text chain (all analytical modules), vision chain (AI Imagery Interpreter — Llama-4 models only).

### Google Gemini

- **URL:** aistudio.google.com
- **Cost:** Free tier available (gemini-2.5-flash and others). Rate limits apply. Gemini 2.5 Pro requires a paid billing account.
- **What it provides:** Google's multimodal AI models. Handles both text and images.
- **Account required:** Yes. Google account. Generate an API key at aistudio.google.com.
- **Used by:** Text chain (all analytical modules, first priority), vision chain (AI Imagery Interpreter, all five Gemini models).

---

## The Map Picker — Shared Component

File: `map_picker.py`

The map picker solves a specific problem: geocoding returns approximate bounding
boxes that are often much larger than the area the user wants to analyse. "Port of
Rotterdam" might resolve to an inland agricultural area. "Nile River" might return
a box spanning 140,000 km².

The map picker appears as a collapsible expander below the location caption in
every analytical module. The user opens it, sees a map centred on the geocoded
location, and clicks anywhere to place the analysis area. A size slider (25 to
500 km, step 25) controls the bbox size around the clicked point.

**How it works:**

1. The geocoded bbox centres the map.
2. The user clicks a point on the map.
3. The click coordinates are stored in Streamlit session state using a module-specific
   key prefix (se_, ts_, sar_, cd_, ii_) so modules do not interfere with each other.
4. A blue rectangle is drawn showing the exact area that will be analysed.
5. The module uses the clicked bbox instead of the geocoded bbox for all data fetching.
6. When the user types a new location, `clear_click()` is called to discard the
   old click so it does not carry over.

**Public interface:**

```
render_map_picker(centre_bbox, picker_key, default_size_km)
  → returns [min_lon, min_lat, max_lon, max_lat] if clicked, or None

clear_click(picker_key)
  → call when the user types a new location
```

---

## Technology Stack Summary

| Layer | Technology | Purpose |
|---|---|---|
| App framework | Streamlit | UI, layout, session state, deployment |
| Maps | Folium + streamlit-folium | Interactive maps, layer overlays |
| Charts | Plotly | Time series, bar charts |
| Satellite data (optical) | pystac-client + planetary-computer | STAC search and COG access |
| Satellite data (radar + time series) | earthengine-api | GEE cloud processing |
| Image processing | PIL, NumPy | Image rendering, clipping, conversion |
| Geocoding | requests (ArcGIS REST API) | Place name to coordinates |
| AI text | groq, google-genai | Language model inference |
| AI vision | google-genai, groq | Multimodal image interpretation |
| Environment config | python-dotenv | API key loading |
| Hosting | Streamlit Cloud | Free deployment from GitHub |
| Version control | GitHub | Deploy repo only |

---

## What Is Not In This Portal

Understanding the boundaries is as important as understanding what is included.

**No real-time data.** Every module fetches historical data. Sentinel-2 revisits every
5 days. MODIS is 16-day composites. SAR is based on selected date ranges. Nothing
is live or streaming.

**No user accounts or stored data.** The portal is stateless. Session state lives in
the browser tab. Nothing is saved between visits.

**No production-grade analysis.** The portal demonstrates concepts. For operational
decisions — pipeline leak detection, vegetation encroachment enforcement, flood risk
assessment — you would use purpose-built commercial platforms with validated algorithms,
calibrated data, and auditable outputs.

**No proprietary data.** Every data source used is free and publicly available.
Commercial platforms (Planet, Capella, ICEYE, GHGSat) offer higher resolution,
higher revisit frequency, and more specialized sensors — at a cost.

---

*Document maintained in: docs/architecture/portal-architecture.md*
*Update this document whenever a new module is added or a data source changes.*
