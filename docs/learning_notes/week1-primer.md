# Week 1 Primer — Earth Observation with AI
# EOIL Program — Days 1 through 7
# Audience: team members new to satellite data analysis

---

## Executive Summary

Week 1 built the foundational layer of the Earth Observation Innovation Lab.
Seven days of work produced one deployed web portal, two documented Jupyter notebooks, and fluency with three satellite data platforms.
The portal is live at https://eoil-explorer.streamlit.app and contains four modules.
All data is free. All code is documented and committed to GitHub.

---

## What Was Built

### Live Portal — eoil-explorer.streamlit.app

One URL hosts everything built in Week 1.
Sidebar navigation switches between four modules.
No login required. Works in any browser.

| Module | What it does |
|---|---|
| Welcome | Overview of the portal and how to use each module |
| Spectral Explorer | Pulls a satellite image chip for any location and computes spectral indices |
| Time Series Explorer | Plots a 10-year vegetation or temperature trend for any region |
| SAR Explorer | Renders radar imagery for any port or coastal area across two dates |

### Jupyter Notebooks

| Notebook | What it covers |
|---|---|
| `02_gee_ndvi_timeseries.ipynb` | 10-year NDVI time series for the African Sahel. 253 data points. Trend: +0.0034 per year. |
| `03_sentinel1_sar_basics.ipynb` | Sentinel-1 SAR imagery for the Port of Rotterdam. VV, VH, false color composite, and backscatter change map. |

Both notebooks are in `notebooks/` in the GitHub repository.
Each one shows the raw mechanics before the app makes them interactive.

---

## The Three Data Platforms

### Planetary Computer (Microsoft)

Used by the Spectral Explorer.
Hosts Sentinel-2 and Landsat imagery.
Access is via STAC: a standardized catalog that lets you search by location, date, and cloud cover without downloading full images.
Sentinel-2 has 13 spectral bands at 10m to 60m resolution and a 5-day revisit cycle.

### Google Earth Engine (GEE)

Used by the Time Series Explorer and SAR Explorer.
A cloud processing platform that runs analysis on Google's servers.
You write the query; GEE runs it against petabytes of archived data and returns results.
No downloading. No local compute required.
Access requires a service account and a project registered at earthengine.google.com.

### Copernicus Sentinel-1 (via GEE)

Used by the SAR Explorer.
Sentinel-1 is a radar satellite. It sees through clouds and works at night.
It measures how strongly the ground reflects a microwave signal back to the sensor.
This is called backscatter.

---

## Key Concepts Introduced in Week 1

### Spectral Indices

Satellite sensors measure reflected light across multiple wavelength bands.
Different surfaces reflect different wavelengths at different intensities.
Spectral indices combine two or more bands into a single number that highlights a specific surface type.

| Index | Formula | What it detects |
|---|---|---|
| NDVI | (NIR - Red) / (NIR + Red) | Vegetation health and density |
| NDWI | (Green - NIR) / (Green + NIR) | Open water bodies |
| NBR | (NIR - SWIR) / (NIR + SWIR) | Burned area severity |

Values range from -1 to +1.
High NDVI means dense healthy vegetation.
Low NDVI means bare soil, water, or urban surface.

### SAR Polarization

Sentinel-1 transmits microwave pulses in two polarizations: VV and VH.

**VV (vertical transmit, vertical receive):** responds strongly to surface roughness and hard structures.
Ships, buildings, and roads appear bright. Calm water appears dark.

**VH (vertical transmit, horizontal receive):** responds to volume scattering inside a target.
Dense vegetation appears brighter than in VV. Useful for separating forest from urban surfaces.

**False Color composite:** combines VV, VH, and the VV/VH ratio into a single RGB image.
Colors indicate surface type: bright pink/magenta is typically urban or industrial.
Dark blue is water. Green tones are vegetation.

**Change Map:** subtracts VV backscatter on Date 1 from VV backscatter on Date 2.
Blue areas increased in backscatter between the two dates (e.g. new structures, wetter soil).
Red areas decreased (e.g. clearing, flooding, surface change).

### Backscatter Statistics

Backscatter is measured in decibels (dB). The scale is negative: values closer to zero are brighter.

| Surface type | Typical VV range | Typical VH range |
|---|---|---|
| Calm water | -25 to -20 dB | -30 to -25 dB |
| Urban / industrial | -10 to 0 dB | -15 to -5 dB |
| Vegetation | -15 to -8 dB | -20 to -12 dB |
| Bare soil | -18 to -10 dB | -22 to -15 dB |

The backscatter comparison chart in the SAR Explorer shows VV and VH min, mean, and max for each selected date.
A shift upward between Date 1 and Date 2 means surfaces became rougher or wetter.
A shift downward means surfaces became smoother or drier.

### STAC — the Satellite Data Catalog Standard

STAC (SpatioTemporal Asset Catalog) is an open standard for indexing satellite imagery.
Instead of downloading full scenes, you query a STAC catalog for items that match your location, date, and sensor.
The catalog returns links to specific files or bands. You download only what you need.
Planetary Computer uses STAC. Most major data archives are moving to STAC.

### Cloud-Optimized GeoTIFF (COG)

COGs are satellite image files structured so you can stream a small geographic window without downloading the full file.
A Sentinel-2 full scene is about 1 GB. A COG query for a 10 km area might transfer 2 MB.
This is what makes browser-based satellite analysis practical.

---

## Limitations to Know

**SAR is not optical.** It takes practice to interpret. A bright pixel is not necessarily a building. Context matters: shape, proximity to water, and change pattern together tell the story.

**GEE tile tokens expire.** The interactive SAR map uses tile URLs that include a temporary authentication token issued by GEE. These tokens are valid for a few hours. If the map goes blank, run the query again to get a fresh token.

**Geocoding quality varies.** Typing a country name returns a very large bounding box. The SAR Explorer is designed for ports, coastlines, river deltas, and industrial sites — areas 20 km to 150 km across. Large regions return dark or incomplete imagery.

**Cloud cover affects optical data.** Sentinel-2 and Landsat are blocked by clouds. SAR (Sentinel-1) is not. For persistent cloud cover regions, SAR is the only reliable source.

**10-day compositing window.** The SAR Explorer searches for imagery within 10 days of the selected date. Sparse coverage regions may return no results. The North Sea and major ports have dense revisit schedules (2 to 4 days).

---

## What Comes Next

Week 2 adds AI interpretation to every module and introduces change detection as a dedicated analysis layer.

Day 8: Additional spectral indices in the Time Series Explorer (EVI, NDWI, Burned Area). Week 1 documentation review.
Day 9: Change Detection Dashboard — compute NDVI difference between two dates, threshold changed pixels, render change map.
Day 10: AI Imagery Interpreter — pass a satellite chip to Gemini vision model, receive structured plain-language interpretation.
Day 13: EO Conversational Assistant — chat interface with EO context injected into the system prompt.

---

## How to Use the Portal

Go to https://eoil-explorer.streamlit.app

**Spectral Explorer:**
1. Type a location in the search box (city, landmark, coordinates)
2. Select a satellite and date
3. The app retrieves the nearest available image and renders it
4. Scroll down to see spectral index maps and AI interpretation

**Time Series Explorer:**
1. Type a location
2. Select a dataset (NDVI, LST, or others added in Day 8)
3. The app queries 10 years of GEE data and plots the trend
4. The chart shows annual range and the overall trend line

**SAR Explorer:**
1. Type a port, coastline, or industrial area (not a country)
2. Select two dates at least 30 days apart for meaningful change
3. The app retrieves Sentinel-1 GRD imagery for each date
4. Use the layer control (top right of map) to toggle between VV, VH, False Color, and Change Map
5. Scroll down to see backscatter statistics and AI interpretation

---

*Document prepared: Day 8, EOIL 30-day program.*
*Next update: Day 14 — Week 2 review.*
