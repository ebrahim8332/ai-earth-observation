# AI-Native Earth Observation Innovation Lab (EOIL)

A 30-day personal build and learning program. Rebuilds satellite data analysis skills and integrates modern AI into every workflow.

**Live portal:** https://eoil-explorer.streamlit.app
**GitHub:** https://github.com/ebrahim8332/ai-earth-observation

---

## What This Is

One deployed Streamlit portal with sidebar modules — one URL for everything built in the program.
No standalone apps. Each day adds a new module or notebook to the same portal.

All data sources are free. No paid APIs. No OpenAI. No Anthropic.

---

## Live Portal Modules

| Module | Description | Day Added |
|---|---|---|
| Welcome | Program overview and module guide | 7 |
| Spectral Explorer | Sentinel-2 chip from Planetary Computer, NDVI/NDWI/NDSI, Gemini vision interpretation | 3 |
| Time Series Explorer | 10-year NDVI/EVI/LST/NDWI/Burned Area trends from GEE, annual comparison chart | 6 |
| SAR Explorer | Sentinel-1 VV/VH/false color/change map, backscatter stats, AI interpretation | 7 |
| Change Detection | NDVI difference between two dates, three-layer change map, 12 statistics, AI interpretation | 9 |

---

## Notebooks

| Notebook | Topic | Status |
|---|---|---|
| `01_stac_query_demo.ipynb` | Planetary Computer STAC query, COG streaming, RGB + NDVI | pending |
| `02_gee_ndvi_timeseries.ipynb` | 10-year NDVI time series for the Sahel via GEE | complete |
| `03_sentinel1_sar_basics.ipynb` | Sentinel-1 SAR over Rotterdam, VV/VH/false color/change | complete |
| `04_change_detection.ipynb` | NDVI change detection mechanics, GEE tile URLs, statistics | complete |

---

## Repository Structure

```
ai-earth-observation/
├── CLAUDE.md                  project context for Claude Code
├── PROJECT.md                 current status and day-by-day progress
├── README.md                  this file
├── requirements.txt
├── .gitignore
│
├── apps/
│   └── 01_eo_explorer/        all portal modules live here
│       ├── app.py             main Streamlit app, sidebar routing
│       ├── config.py          API key loading
│       ├── geocoder.py        ArcGIS + Nominatim place-name to bbox
│       ├── ai_chain.py        11-model AI fallback chain (6 Gemini + 5 Groq)
│       ├── gee_change.py      Change Detection module
│       ├── gee_timeseries.py  Time Series Explorer module
│       ├── gee_sar.py         SAR Explorer module
│       └── spectral_explorer.py  Spectral Explorer module
│
├── notebooks/                 one notebook per analytical topic
└── docs/
    ├── learning_notes/        day-by-day learning log and team primers
    └── gee-authentication-guide.md
```

---

## Running Locally

```powershell
# From the apps/01_eo_explorer folder:
C:\Users\alnoo\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app.py
```

Runs on http://127.0.0.1:8501

---

## Environment Variables

Create a `.env` file in `apps/01_eo_explorer/` with:

```
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

GEE credentials go in `apps/01_eo_explorer/.streamlit/secrets.toml` (local) or the Streamlit Cloud secrets dashboard.

All keys are optional. Every module has substantive fallback behaviour when keys are absent.

---

## Two-Repo Deploy Pattern

- Dev source: `Desktop\code\ai-earth-observation\` — never pushed to GitHub
- Deploy copy: `Desktop\code\ai-earth-observation-feedback\` — pushed to GitHub, triggers Streamlit Cloud redeploy

After changes: copy runtime files to the feedback folder, then `git push`.

---

## Tech Stack

| Layer | Tools |
|---|---|
| App | Python, Streamlit, streamlit-folium |
| Maps | Folium, Leaflet.js |
| Optical data | Planetary Computer (STAC), Google Earth Engine |
| SAR data | Google Earth Engine (Sentinel-1 GRD) |
| AI | Groq (Llama, Qwen), Google Gemini — 11-model fallback chain |
| Geocoding | ArcGIS World Geocoding (primary), Nominatim (backup) |
