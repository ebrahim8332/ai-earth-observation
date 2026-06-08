# CLAUDE.md
# AI-Native Earth Observation Innovation Lab (EOIL)
# Permanent project context for Claude Code and Codex
# Read this file at the start of every session before doing anything

---

## Who I Am

My name is Alnoor Ebrahim. I am a technology and AI advisor. I am not a
software developer. I do not write code manually. You write all code.
My job is to understand what is built, learn from it, and deploy it.

I have a background in electrical engineering and satellite data analysis.
I worked with Landsat, Sentinel, LiDAR, spectral indices, and atmospheric
data roughly five years ago. That knowledge is stale. Treat each topic as
if I need a refresher, but do not over-explain basics I can infer.

---

## What This Project Is

The AI-Native Earth Observation Innovation Lab is a 30-day personal learning
and build program. It rebuilds my knowledge of satellite data analysis and
integrates modern AI into every workflow.

The program produces:
- 1 deployed Streamlit portal with up to 13 sidebar modules (one URL for everything)
- 9 documented Jupyter notebooks
- 5 team education documents
- A complete GitHub repository usable as a team curriculum

No standalone apps. Every analytical deliverable is a new module added to the
portal at https://eoil-explorer.streamlit.app. One URL, one experience.

Every application answers at least one of four decision questions:
1. What is happening?
2. Why is it happening?
3. What changed?
4. What should we do next?

---

## Guiding Principles

**AI first.** AI is never an afterthought. Every app includes at least one
of the five AI roles: Tutor, Developer, Analyst, Agent, Advisor.

**Build to deploy.** Every session ends with a working artifact committed
to GitHub and deployed to Streamlit Cloud.

**Decision intelligence.** The goal is to support decisions, not display
data. Every app should answer a real question.

**Graceful degradation.** Every app must work with no API key present.
Fallback responses must be substantive, not placeholder text.

**Notebook first, app second.**
Every analytical feature is built as a Jupyter notebook before it becomes an interactive app.
The notebook lets you run code cell by cell, see raw outputs, and understand what is happening
under the hood. The app is the polished, interactive version of the same logic.
This dual approach applies to every module in the program.
Notebook days are for learning the mechanics. App days are for making the mechanics usable.

---

## AI Architecture — Two-Layer Rule (agreed Day 12, applies to all remaining modules)

Every module from Day 13 onward must implement both layers. Neither layer alone is sufficient.

**Layer 1 — ML / Algorithm layer**
Classical machine learning or deep learning finds the pattern in the data.
Examples: K-means clustering, Isolation Forest, PCA, DBSCAN, CNN classification.
This layer does not describe. It computes. It finds structure the human eye cannot see
at scale. It does not require labeled data where possible (prefer unsupervised methods).

**Layer 2 — Generative AI layer**
Groq or Gemini explains what the ML layer found.
It interprets, contextualises, recommends action, and writes narrative output.
It never substitutes for real analysis. It always receives structured inputs from Layer 1.

**Why both layers.**
Generative AI alone describes but does not find. ML alone finds but does not explain.
Together they produce the correct architecture: find the pattern, explain what it means.
This is how real production EO workflows operate in 2025–2026.

**UI rule.**
The two layers are visually distinct in every module.
Users can see what the algorithm found and what the AI said about it separately.
Never blend them into a single output block.

**Option A constraint.**
All modules are conceptual demonstrations. No real operational data required.
Every module must be understandable to a non-specialist.
Every module must be demonstrable in a 10-minute conversation.

**Zero budget.** Every data source, API, library, and platform must be free.
No OpenAI. No Anthropic API. No paid tiers.

---

## The Technology Stack

**AI APIs (free tier only)**
- Groq: text-based AI features. Library: groq. Key: GROQ_API_KEY
- Gemini: vision-based AI features. Library: google-generativeai. Key: GEMINI_API_KEY
- Never use OpenAI API. Never use Anthropic API. Both cost money.

**Data sources (all free)**
- Planetary Computer: Sentinel-2, Landsat, Copernicus DEM via STAC
- Copernicus Data Space: Sentinel-1, Sentinel-5P TROPOMI
- Google Earth Engine: cloud-scale processing, 10-year time series
- NASA Earthdata: EMIT hyperspectral, ICESat-2, GEDI, MODIS
- OpenTopography: USGS 3DEP lidar point clouds
- OpenAerialMap: drone and aerial orthomosaics
- CAMS: atmospheric gas reanalysis

**Core Python libraries**
- Streamlit + streamlit-folium: all app deployment
- folium + leafmap: interactive maps
- geopandas + shapely + pyproj: vector data
- pystac-client + planetary-computer: STAC data access
- rasterio + rioxarray + xarray: raster processing
- earthengine-api + geemap: Google Earth Engine
- scikit-learn: ML classification
- laspy + open3d: LiDAR processing
- groq + google-genai: AI features (new SDK, not google-generativeai)
- python-dotenv: environment variable loading

**Deployment**
- Dev source: local only, no GitHub remote
- Deploy copy: feedback folder -> https://github.com/ebrahim8332/ai-earth-observation
- Live URL: https://eoil-explorer.streamlit.app (single portal, all modules)
- Google Colab: used for heavy model inference (foundation models)
- Key: deploy .streamlit/config.toml must have headless=true

**Deploy procedure — copy files to feedback repo, then push**

Streamlit Cloud is configured with mainModule = apps/01_eo_explorer/app.py.
It runs apps/01_eo_explorer/app.py in the feedback repo, NOT the repo root.
Always copy to apps/01_eo_explorer/ — never to the repo root.

```powershell
# Copy the main app and any new module files
Copy-Item "C:\Users\alnoo\OneDrive\Desktop\code\ai-earth-observation\apps\01_eo_explorer\app.py" `
          "C:\Users\alnoo\OneDrive\Desktop\code\ai-earth-observation-feedback\apps\01_eo_explorer\app.py"

# Example: copy a new shared component (adjust filename as needed)
Copy-Item "C:\Users\alnoo\OneDrive\Desktop\code\ai-earth-observation\apps\01_eo_explorer\map_picker.py" `
          "C:\Users\alnoo\OneDrive\Desktop\code\ai-earth-observation-feedback\apps\01_eo_explorer\map_picker.py"

# Then push from the feedback folder
cd C:\Users\alnoo\OneDrive\Desktop\code\ai-earth-observation-feedback
git add .
git commit -m "[day-XX] description"
git push
```

**WARNING:** Do NOT copy to the repo root (ai-earth-observation-feedback\app.py).
That file is not run by Streamlit Cloud. The correct destination is always
ai-earth-observation-feedback\apps\01_eo_explorer\

**App architecture: single portal**
All modules live inside one Streamlit app with sidebar navigation.
One URL, one deployment, one experience for users and clients.
Each module has its own files inside apps/01_eo_explorer/.
The main app.py routes between modules via the sidebar.
New modules are added per day as new files. The core app does not change.

Current portal modules (app.py v1.7 — grows with each day):
  Module 0 — Welcome panel (default landing page)
  Module 1 — Spectral Explorer (Planetary Computer, optical imagery)
  Module 2 — Time Series Explorer (GEE, MODIS NDVI/EVI/LST/Burned Area, Landsat NDVI/NDWI)
  Module 3 — SAR Explorer (GEE, Sentinel-1 GRD, VV/VH/false color/change map)
  Module 4 — Change Detection (GEE, Sentinel-2/MODIS NDVI diff, three-layer change map)
  Module 5 — AI Imagery Interpreter (Planetary Computer, Sentinel-2 true-color, 7-model vision chain)

Shared components (apps/01_eo_explorer/):
  map_picker.py — click-to-set-location map widget used by all five modules.
                  render_map_picker(centre_bbox, picker_key, default_size_km) returns a bbox or None.
                  clear_click(picker_key) resets click state when the user types a new location.

Architecture rule: no standalone apps. Every new deliverable is a sidebar module
added to apps/01_eo_explorer/app.py. New logic files go in apps/01_eo_explorer/.

GEE credentials: stored in apps/01_eo_explorer/.streamlit/secrets.toml (local, gitignored).
On Streamlit Cloud: add GEE_SERVICE_ACCOUNT_JSON to app secrets dashboard.
Service account: eoil-gee-service@gen-lang-client-0093165324.iam.gserviceaccount.com
Project: gen-lang-client-0093165324

---

## Repository Structure

earth-observation-innovation-lab/
â”œâ”€â”€ CLAUDE.md                    â† this file, always read first
â”œâ”€â”€ PROJECT.md                   â† current status and next steps
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .gitignore
â”œâ”€â”€ .env.example
â”‚
â”œâ”€â”€ apps/
â”‚   â”œâ”€â”€ 01_eo_explorer/          â† Days 1-7: portal app — Welcome, Spectral Explorer, Time Series Explorer, SAR Explorer
â”‚   â”œâ”€â”€ 02_change_detection/     â† Day 9
â”‚   â”œâ”€â”€ 03_ai_imagery/           â† Day 10
â”‚   â”œâ”€â”€ 04_eo_assistant/         â† Day 13
â”‚   â”œâ”€â”€ 05_environmental/        â† Day 15
â”‚   â”œâ”€â”€ 06_atmospheric/          â† Day 17
â”‚   â”œâ”€â”€ 07_multimap/             â† Day 20
â”‚   â”œâ”€â”€ 08_decision_support/     â† Days 22-24
â”‚   â”œâ”€â”€ 09_vendor_evaluator/     â† Days 25-26
â”‚   â””â”€â”€ 10_curriculum_index/     â† Days 27-28
â”‚
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_stac_query_demo.ipynb
â”‚   â”œâ”€â”€ 02_gee_ndvi_timeseries.ipynb
â”‚   â”œâ”€â”€ 03_sentinel1_sar_basics.ipynb
â”‚   â”œâ”€â”€ 04_land_cover_classification.ipynb
â”‚   â”œâ”€â”€ 05_sam_segmentation_demo.ipynb
â”‚   â”œâ”€â”€ 06_prithvi_foundation_model.ipynb
â”‚   â”œâ”€â”€ 07_sar_infrastructure_monitoring.ipynb
â”‚   â”œâ”€â”€ 08_lidar_intelligence.ipynb
â”‚   â””â”€â”€ 09_emit_hyperspectral.ipynb
â”‚
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ build_specs/             â† one build spec per app
â”‚   â””â”€â”€ learning_notes/          â† one notes file per day
â”‚
â”œâ”€â”€ prompts/
â”‚   â”œâ”€â”€ claude_code_prompts.md   â† all build prompts used, archived
â”‚   â””â”€â”€ ai_explanation_prompts.mdâ† reusable AI assistant prompts
â”‚
â”œâ”€â”€ architecture/                â† one diagram per app
â””â”€â”€ datasets/
    â””â”€â”€ sample/                  â† sample GeoJSON and static data

---

## Modular File Standard

Every app in the apps/ folder must follow this structure.
No monolithic single-file apps. One responsibility per file.

app.py          Streamlit layout and wiring only. No business logic.
config.py       Environment variable loading with safe defaults.
data_catalog.py Dataset and theme metadata. Easy to extend.
map_builder.py  Builds and returns Folium map objects.
ai_assistant.py Groq/Gemini calls with graceful fallback logic.
sample_layers.py Hardcoded sample GeoJSON for sessions before live data.

Additional files are added per app as needed (e.g., data_loader.py,
change_detector.py, raster_processor.py).

---

## AI Assistant Pattern

Every app that includes an AI assistant must follow this pattern exactly.

```python
def get_ai_response(prompt, context, api_key=None):
    if api_key:
        try:
            # call Groq or Gemini
            return call_groq(prompt, context, api_key)
        except Exception:
            return get_fallback_response(prompt, context)
    else:
        return get_fallback_response(prompt, context)

def get_fallback_response(prompt, context):
    # Returns substantive, genuinely useful predefined content
    # Never returns "API key required" or placeholder text
    # Must cover: what it is, what it measures, one use case, one limitation
    pass
```

---

## Documentation Standard

Every notebook committed to GitHub must have:
1. A header cell (Markdown) with:
   - Purpose: one sentence
   - Data sources used
   - Python libraries required
   - Expected runtime
   - Key outputs
2. A comment above every function explaining what it does and why
3. A final Markdown cell with:
   - Key findings from this session
   - One or two questions for further investigation
4. A corresponding prompt in prompts/claude_code_prompts.md

Every app must have a README.md in its folder with:
- What the app does
- How to run it locally
- Environment variables required
- Link to deployed Streamlit Cloud URL (add after deployment)

---

## Commit Message Format

Use this format for every commit:

[day-XX] short description of what was built or changed

Examples:
[day-01] add EO Explorer v1.0 foundational app
[day-03] add spectral index module to EO Explorer v1.1
[day-09] add change detection dashboard

---

## Environment Variables

All keys are loaded from .env using python-dotenv.
Never hardcode API keys. Never commit .env to GitHub.
.env is in .gitignore.

Required keys:
GROQ_API_KEY        Groq text AI. Get at console.groq.com
GEMINI_API_KEY      Gemini vision AI. Get at aistudio.google.com
MAPBOX_TOKEN        Mapbox basemaps. Get at mapbox.com

---

## Dependency Management

requirements.txt is staged. Add dependencies progressively.
Do not add heavy geospatial libraries before they are needed.
Streamlit Cloud will fail to build if requirements.txt is overloaded
before those libraries are used.

Day 1-2:   core dependencies only (streamlit, folium, groq, etc.)
Day 3:     uncomment rasterio, rioxarray, xarray, odc-stac
Day 4:     uncomment earthengine-api, geemap
Day 11:    add segment-anything separately via pip
Day 18:    uncomment laspy, open3d
Day 19:    uncomment netCDF4, h5py, spectral

---

## The 30-Day Build Plan

### WEEK 1: Orientation and Data Access

Day 1   EO Explorer v1.0
        Build the foundational app. Establish repo structure.
        Files: apps/01_eo_explorer/
        Output: deployed Streamlit app

Day 2   STAC and Cloud-Native Data
        Build: notebooks/01_stac_query_demo.ipynb
        Query Planetary Computer for Sentinel-2, stream COG, render
        true-color composite without full file download.
        Output: notebook committed to GitHub

Day 3   Sentinel-2 Spectral Analysis
        Upgrade EO Explorer to v1.1. Add live spectral index module.
        Calculate NDVI and NDWI from Planetary Computer data.
        Add Gemini vision interpretation of rendered NDVI chip.
        Uncomment rasterio, rioxarray, xarray, odc-stac in requirements.txt
        Output: EO Explorer v1.1 deployed

Day 4   Google Earth Engine
        Build: notebooks/02_gee_ndvi_timeseries.ipynb
        10-year NDVI time series for a user-specified region.
        Uncomment earthengine-api, geemap in requirements.txt
        Output: notebook committed to GitHub

Day 5   SAR Data and Sentinel-1
        Build: notebooks/03_sentinel1_sar_basics.ipynb
        Download Sentinel-1 GRD, visualize VV and VH, compute
        false color composite.
        Output: notebook committed to GitHub

Day 6   Drone and Aerial Imagery
        Upgrade EO Explorer to v1.2. Add drone imagery module.
        Load orthomosaic from OpenAerialMap, overlay NDVI, polygon
        selection with area statistics export.
        Output: EO Explorer v1.2 deployed

Day 7   Portal Redesign and Welcome Module
        Replace the sample-data themed layer map in the first portal tab
        with a proper Welcome and Overview panel.
        Content: what the portal is, what each module does, how to use it.
        Move Spectral Explorer to its own sidebar module.
        Drone imagery deferred to a later day.
        Output: redesigned portal deployed

Day 8   Week 1 Review
        Audit all notebooks for missing documentation.
        Write team primer document: docs/learning_notes/week1-primer.md
        Update README.md with all deployed app URLs.
        Output: documented repository, team primer

---

### WEEK 2: Core Analysis and AI Integration

Day 8   Land Cover Classification
        Build: notebooks/04_land_cover_classification.ipynb
        Load Sentinel-2, apply K-Means clustering, train Random Forest
        classifier on labeled polygons, produce land cover map with
        accuracy statistics. Groq interprets the result.
        Output: notebook committed to GitHub

Day 9   Change Detection Dashboard
        Build: apps/02_change_detection/
        Modular file structure. User selects two dates and a region.
        Compute NDVI difference, threshold changed pixels, render
        interactive change map. Groq provides interpretation.
        Output: deployed Streamlit app

Day 10  AI Imagery Interpreter
        Build: apps/03_ai_imagery/
        User draws bounding box on map. App retrieves Sentinel-2 chip.
        Chip passed to Gemini with structured prompt. Display AI
        interpretation with confidence indicators and limitations note.
        Output: deployed Streamlit app

Day 11  SAM Segmentation
        Build: notebooks/05_sam_segmentation_demo.ipynb
        Run SAM on satellite or drone image. Generate segments.
        Groq suggests class labels based on band statistics per segment.
        Add segment-anything to requirements.txt
        Output: notebook committed to GitHub

Day 12  EO Foundation Models
        Build: notebooks/06_prithvi_foundation_model.ipynb
        Run in Google Colab (free GPU). Load Prithvi from Hugging Face.
        Inference on Sentinel-2 chip. Visualize feature embeddings.
        Groq explains what the model is responding to.
        Output: Colab notebook, committed to GitHub

Day 13  EO Conversational Assistant
        Build: apps/04_eo_assistant/
        Split-screen layout: left panel is interactive map with analysis
        results, right panel is Groq chat assistant. Structured EO
        context injected via system prompt. All four AI roles active.
        Output: deployed Streamlit app

Day 14  Week 2 Review
        Audit all Week 2 notebooks for documentation standard.
        Write team document: docs/learning_notes/ai-applied-to-eo.md
        Update all deployed app URLs in README.
        Output: documented repository, team document

---

### WEEK 3: Specialized Domains

Day 15  Environmental Intelligence Dashboard
        Build: apps/05_environmental/
        Region and date range selector. NDVI, EVI, water body extent
        over time. Trend charts. Groq interprets anomalies.
        Data: Google Earth Engine
        Output: deployed Streamlit app

Day 16  Infrastructure Intelligence with SAR
        Build: notebooks/07_sar_infrastructure_monitoring.ipynb
        Sentinel-1 coherence products from GEE for a pipeline corridor.
        Visualize coherence loss. Flag anomalous zones.
        Groq-powered risk summary.
        Output: notebook committed to GitHub

Day 17  Atmospheric Intelligence Dashboard
        Build: apps/06_atmospheric/
        Query TROPOMI methane from Copernicus Data Space.
        Render regional concentration map for selected date.
        Flag anomalies against baseline. Gemini interpretation.
        Output: deployed Streamlit app

Day 18  LiDAR Intelligence
        Build: notebooks/08_lidar_intelligence.ipynb
        Download USGS 3DEP tile. Classify ground vs. vegetation.
        Generate DTM and Canopy Height Model. Interactive 3D viz.
        Groq plain-language terrain interpretation.
        Uncomment laspy, open3d in requirements.txt
        Output: notebook committed to GitHub

Day 19  EMIT Hyperspectral
        Build: notebooks/09_emit_hyperspectral.ipynb
        Download EMIT scene from NASA Earthdata.
        Visualize selected bands. Identify methane absorption features.
        Compare spectra for different surface types.
        Uncomment netCDF4, h5py, spectral in requirements.txt
        Output: notebook committed to GitHub

Day 20  Multi-Layer Geospatial Dashboard
        Build: apps/07_multimap/
        Mapbox basemap with toggleable layers: NDVI, SAR change,
        TROPOMI methane. Date range controls. GeoJSON export.
        Groq responds to natural language layer requests.
        Output: deployed Streamlit app

Day 21  Week 3 Review
        Write four domain briefs in docs/learning_notes/:
          - environmental-intelligence.md
          - infrastructure-intelligence.md
          - atmospheric-intelligence.md
          - lidar-intelligence.md
        Each brief: available data, analysis possible, business decisions,
        commercial alternatives.
        Output: four domain briefs, repository updated

---

### WEEK 4: Capstone Integration

Day 22  Decision Support Platform: Panel 1
        Build: apps/08_decision_support/ scaffold + Panel 1
        Mapbox map with toggleable layers: optical, SAR, atmospheric,
        LiDAR terrain. Layer control with metadata tooltips.
        Output: Panel 1 running locally

Day 23  Decision Support Platform: Panel 2
        Add Panel 2 to apps/08_decision_support/
        Time-series charts for user-selected point or polygon.
        Data from Google Earth Engine.
        Output: Panels 1 and 2 running locally

Day 24  Decision Support Platform: Panel 3 + Deploy
        Add Panel 3 to apps/08_decision_support/
        Groq AI advisor with structured EO context injection.
        Gemini image chip interpretation on demand.
        Wire all three panels. Deploy to Streamlit Cloud.
        All five AI roles active simultaneously.
        Output: deployed Streamlit app: EO Decision Support Platform

Day 25  EO Vendor Evaluator: Data Layer
        Build: apps/09_vendor_evaluator/ scaffold + data layer
        Create datasets/eo-vendor-database.json with entries for:
        Sentinel-2, Landsat, Sentinel-1, TROPOMI, EMIT, Planet,
        Capella Space, ICEYE, Maxar, Airbus, GHGSat, Carbon Mapper.
        Each entry: type, resolution, revisit, cost model, AI readiness,
        best use cases, limitations.
        Output: JSON database committed, app scaffold ready

Day 26  EO Vendor Evaluator: App + Deploy
        Complete apps/09_vendor_evaluator/
        User inputs a use case. App returns relevant platforms, vendor
        scoring matrix, and Groq-powered plain-language recommendation.
        Deploy to Streamlit Cloud.
        Output: deployed Streamlit app: EO Vendor Evaluator

Day 27  Curriculum Index: Repository Cleanup
        Audit every notebook and app for documentation standard.
        Add missing header cells, function comments, and final cells.
        Update prompts/claude_code_prompts.md with all prompts used.
        Begin apps/10_curriculum_index/ scaffold.
        Output: clean, fully documented repository

Day 28  Curriculum Index: App + Deploy
        Complete apps/10_curriculum_index/
        Interactive index of all apps and notebooks.
        Click any module to see: notebook link, app URL, one-paragraph
        summary, AI roles used, key concepts covered.
        Deploy to Streamlit Cloud.
        Update master README with all final app URLs.
        Output: deployed Streamlit app: EOIL Curriculum Index

Day 29  Conference Presentation Outline
        Build: docs/eo-ai-conference-outline.md
        Structure for a 20-minute conference presentation covering:
          - Current state of EO and AI (5 min)
          - Three applied use cases from this program (12 min)
          - Framework for evaluating EO technology investments (3 min)
        Include speaker notes for each section.
        Groq drafts the speaker notes.
        Output: presentation outline with speaker notes

Day 30  Gap Assessment
        Review all twelve apps and nine notebooks.
        For each of eight domains rate comfort level 1-5:
          constellations, data infrastructure, analysis, AI models,
          environmental, infrastructure, atmospheric, LiDAR
        Write: docs/learning_notes/30-day-gap-assessment.md
        Identify top three areas for a second 30-day cycle.
        Update README with program completion status.
        Output: gap assessment document, final repository state

---

## Architecture Reference

A plain-English architecture guide is maintained at:
docs/architecture/portal-architecture.md

It covers: every file and what it does, all six modules (inputs/outputs/data sources),
API keys required, what is free vs paid, the AI chains, the map picker, the deploy
pipeline, and what the portal does not do. Read it when joining the project or
briefing someone new.

---

## How to Start Each Session

At the start of every Claude Code session:
1. Read CLAUDE.md (this file)
2. Read PROJECT.md for current status and today's task
3. Confirm which day we are on and what the deliverable is
4. Ask if there are any blockers or changes before starting
5. Build in the sequence specified in PROJECT.md
6. Commit with the correct message format at the end
7. Update PROJECT.md with completion status and next steps

---

## How I Work With You

- I do not write code. You write all code.
- I paste prompts. You execute them completely.
- After each major step, show me what was built before continuing.
- If something will not work as specified, tell me why and propose an
  alternative before building it differently.
- When a session is complete, tell me exactly what was built, what the
  commit message was, and what the next session will cover.
- Keep responses direct. No filler. No motivational language.



