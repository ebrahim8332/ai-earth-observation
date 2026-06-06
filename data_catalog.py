"""
data_catalog.py — Content library for EO themes and datasets.
This file stores structured metadata as plain Python dictionaries.
It has no imports and no logic. Adding a new theme or dataset means
adding a new entry to the dictionaries below.
"""

# THEMES: each key is the display name shown in the sidebar dropdown.
# Each theme includes:
#   description    — plain-English explanation of what this theme covers
#   datasets       — list of dataset names available under this theme
#                    (must match keys in DATASETS below)
#   business_cases — list of real-world decision questions this theme answers
THEMES: dict = {
    "EO Basics": {
        "description": (
            "Earth Observation (EO) is the collection of information about the planet's "
            "physical, chemical, and biological systems using sensors on satellites, aircraft, "
            "and ground stations. EO data is used to monitor change, detect anomalies, "
            "support planning, and answer questions that ground surveys cannot answer at scale. "
            "This theme introduces the foundational concepts, satellite programs, and sensor types "
            "that underpin all other themes in this platform."
        ),
        "datasets": [
            "Sentinel-2 Multispectral Imagery",
            "Landsat 8/9 Imagery",
        ],
        "business_cases": [
            "What satellite programs are freely available for my analysis?",
            "How do I select the right sensor for my use case?",
            "What spatial and temporal resolution do I actually need?",
        ],
    },
    "Vegetation and Agriculture": {
        "description": (
            "Vegetation monitoring uses reflected light in the near-infrared and red spectral bands "
            "to measure plant health, biomass, and stress. The most widely used index is NDVI "
            "(Normalized Difference Vegetation Index). Agricultural applications include crop health "
            "monitoring, irrigation planning, yield forecasting, and early detection of disease or "
            "drought stress. Sentinel-2 and Landsat both provide the bands needed for these analyses "
            "at no cost."
        ),
        "datasets": [
            "Sentinel-2 Multispectral Imagery",
            "Landsat 8/9 Imagery",
            "Vegetation Health Layer",
        ],
        "business_cases": [
            "Which fields are under water or drought stress right now?",
            "How has crop cover changed over the past growing season?",
            "Where are the highest-risk zones for yield loss this quarter?",
        ],
    },
    "Water and Environment": {
        "description": (
            "Water body monitoring uses EO to track surface water extent, turbidity, algal blooms, "
            "and shoreline change. Sentinel-2's short-wave infrared bands distinguish water from land "
            "clearly. Environmental monitoring extends to wetland health, flood mapping, and detecting "
            "illegal discharge. These datasets support compliance monitoring, insurance risk modeling, "
            "and infrastructure siting decisions near water."
        ),
        "datasets": [
            "Sentinel-2 Multispectral Imagery",
            "Landsat 8/9 Imagery",
        ],
        "business_cases": [
            "Has the reservoir level changed materially since last quarter?",
            "Are there signs of algal bloom or turbidity in the monitored water body?",
            "What areas are at flood risk given current soil saturation levels?",
        ],
    },
    "Atmosphere and Methane": {
        "description": (
            "Atmospheric EO measures trace gases in the lower atmosphere using satellite spectrometers. "
            "Sentinel-5P TROPOMI is the primary free source. It detects methane (CH4), nitrogen dioxide "
            "(NO2), carbon monoxide (CO), and ozone at roughly 5 km resolution globally. "
            "Methane monitoring is particularly relevant for oil and gas operators, regulators, and "
            "investors assessing Scope 1 emissions and regulatory exposure. TROPOMI data has a daily "
            "global revisit, making it the most accessible emissions intelligence tool available at no cost."
        ),
        "datasets": [
            "Sentinel-5P TROPOMI Atmospheric Data",
        ],
        "business_cases": [
            "Are there anomalous methane concentrations above any of our operating sites?",
            "How do our reported emissions compare to satellite-detected concentrations?",
            "Which regions carry the highest regulatory methane exposure?",
        ],
    },
    "Infrastructure Monitoring": {
        "description": (
            "Infrastructure monitoring uses EO to detect physical changes around energy, transport, "
            "and utility assets. Applications include pipeline corridor monitoring, detecting land "
            "subsidence near facilities, monitoring construction activity near rights-of-way, and "
            "assessing damage after weather events. Synthetic Aperture Radar (SAR) from Sentinel-1 "
            "is particularly useful because it operates through cloud cover and at night. "
            "GEDI LiDAR adds canopy height data relevant for corridor clearance and encroachment detection."
        ),
        "datasets": [
            "Infrastructure Sample Layer",
            "GEDI Spaceborne LiDAR",
            "Sentinel-2 Multispectral Imagery",
        ],
        "business_cases": [
            "Has there been construction or land disturbance within 500 meters of our pipeline?",
            "Are there signs of soil movement or subsidence near our compressor stations?",
            "What is the current vegetation encroachment status along our transmission corridor?",
        ],
    },
    "Climate and Risk": {
        "description": (
            "Climate and risk analysis uses EO to quantify physical climate risks at the asset level. "
            "This includes flood frequency mapping, wildfire risk scoring, heat stress assessment, "
            "and coastal erosion monitoring. EO data from Landsat, Sentinel-2, and MODIS provides "
            "decades of historical imagery for trend analysis. Combining EO with climate projection "
            "models produces asset-level risk scores that support insurance underwriting, portfolio "
            "stress testing, and regulatory disclosure under TCFD frameworks."
        ),
        "datasets": [
            "Landsat 8/9 Imagery",
            "Sentinel-2 Multispectral Imagery",
            "Sentinel-5P TROPOMI Atmospheric Data",
        ],
        "business_cases": [
            "Which of our assets are in the top quartile for physical climate risk?",
            "How has the flood extent around this facility changed over the past decade?",
            "What is the wildfire risk score for our California assets this season?",
        ],
    },
}


# DATASETS: each key matches a dataset name used in the THEMES lists above.
# Each dataset includes:
#   measures     — what the data captures
#   sensors      — the instruments that produce it
#   resolution   — spatial resolution (meters or km)
#   revisit      — how often new data is collected
#   use_cases    — the primary applications
#   limitations  — what it cannot do or where it performs poorly
DATASETS: dict = {
    "Sentinel-2 Multispectral Imagery": {
        "measures": (
            "Reflected sunlight across 13 spectral bands from visible to short-wave infrared. "
            "Used to compute vegetation indices (NDVI, EVI), water indices (NDWI), and bare soil indicators."
        ),
        "sensors": "MultiSpectral Instrument (MSI) on Sentinel-2A and 2B satellites (ESA).",
        "resolution": "10 meters (visible and NIR bands), 20 meters (red-edge and SWIR), 60 meters (atmospheric bands).",
        "revisit": "5 days at the equator with both satellites combined. More frequent at higher latitudes.",
        "use_cases": (
            "Crop health and yield forecasting, deforestation detection, flood mapping, "
            "urban growth monitoring, coastal water quality, wildfire burn scar assessment."
        ),
        "limitations": (
            "Optical sensor — blocked by cloud cover and smoke. Not useful for areas with persistent cloud. "
            "No nighttime imaging. No vertical structure data (canopy height, building height). "
            "Atmospheric correction required for accurate reflectance values."
        ),
    },
    "Landsat 8/9 Imagery": {
        "measures": (
            "Reflected and emitted radiation across 11 bands including thermal infrared. "
            "Enables land surface temperature mapping in addition to spectral indices."
        ),
        "sensors": "Operational Land Imager (OLI-2) and Thermal Infrared Sensor (TIRS-2) on Landsat 9 (USGS/NASA).",
        "resolution": "30 meters (multispectral), 15 meters (panchromatic), 100 meters (thermal).",
        "revisit": "16 days per satellite. Landsat 8 and 9 together give an 8-day combined revisit.",
        "use_cases": (
            "Urban heat island mapping, long-term land cover change (archives back to 1972), "
            "surface temperature monitoring near industrial sites, water body extent over decades."
        ),
        "limitations": (
            "Coarser resolution than Sentinel-2 for agriculture use. 16-day single-satellite revisit "
            "misses fast-moving events. Thermal band at 100 meters is too coarse for individual building analysis. "
            "Same cloud blockage limitations as all optical sensors."
        ),
    },
    "Sentinel-5P TROPOMI Atmospheric Data": {
        "measures": (
            "Column concentrations of methane (CH4), nitrogen dioxide (NO2), ozone (O3), "
            "sulfur dioxide (SO2), formaldehyde (HCHO), and carbon monoxide (CO) in the atmosphere."
        ),
        "sensors": "TROPOMI spectrometer on Sentinel-5 Precursor satellite (ESA).",
        "resolution": "5.5 km x 3.5 km per pixel. Not suitable for facility-level attribution without modeling.",
        "revisit": "Daily global coverage. Full-orbit swath of 2,600 km.",
        "use_cases": (
            "Regional methane hotspot detection, oil and gas basin emissions monitoring, "
            "NO2 proxy for economic activity, regulatory emissions verification, climate reporting."
        ),
        "limitations": (
            "5 km resolution cannot pinpoint individual emission sources — it detects plumes and regional signals. "
            "Retrieval quality degrades under cloud cover, over bright surfaces (desert, snow), and at low sun angles. "
            "Attribution to a specific facility requires dispersion modeling, not just the raw TROPOMI value."
        ),
    },
    "GEDI Spaceborne LiDAR": {
        "measures": (
            "Vertical structure of vegetation: canopy height, canopy cover, leaf area index, "
            "and above-ground biomass estimates. Fires laser pulses from the International Space Station."
        ),
        "sensors": "Global Ecosystem Dynamics Investigation (GEDI) LiDAR on the ISS (NASA).",
        "resolution": "25-meter footprint per shot. Shots spaced 600 meters apart along-track — not wall-to-wall coverage.",
        "revisit": "Non-systematic. The ISS orbit precesses, giving irregular revisit intervals.",
        "use_cases": (
            "Carbon stock estimation, forest biomass mapping, vegetation height along infrastructure corridors, "
            "canopy encroachment detection, biodiversity proxy for regulatory reporting."
        ),
        "limitations": (
            "Sample-based, not continuous coverage. Cannot generate a continuous canopy height map without "
            "interpolation. Operates only in non-storm conditions. Urban and low-vegetation retrievals are noisy. "
            "Data only available between 51.6 degrees north and south latitude."
        ),
    },
    "Infrastructure Sample Layer": {
        "measures": (
            "Hardcoded sample pipeline corridor geometry. Represents a gas transmission right-of-way. "
            "Not real operational data — used to demonstrate infrastructure overlay functionality."
        ),
        "sensors": "No sensor — this is a manually created vector layer for demonstration purposes.",
        "resolution": "Vector geometry. Display scale depends on zoom level.",
        "revisit": "Static. Updated manually when new sample data is loaded.",
        "use_cases": (
            "Demonstrating pipeline corridor overlays, testing spatial queries against right-of-way geometry, "
            "learning how to visualize linear infrastructure on a geospatial map."
        ),
        "limitations": (
            "Not real data. Coordinates are illustrative only. "
            "Do not use for any operational, engineering, or compliance purpose."
        ),
    },
    "Vegetation Health Layer": {
        "measures": (
            "Sample NDVI-based vegetation health classification for a demonstration area. "
            "Represents healthy, stressed, and bare ground zones as polygon features."
        ),
        "sensors": "Derived from Sentinel-2 band combinations (B8 and B4). Sample only.",
        "resolution": "Vector polygons representing NDVI zones. Original raster source: 10 meters.",
        "revisit": "Static sample layer. Live version would update on each Sentinel-2 acquisition.",
        "use_cases": (
            "Demonstrating vegetation health overlays, learning NDVI interpretation, "
            "illustrating how satellite data drives field prioritization."
        ),
        "limitations": (
            "Sample data only. Not derived from a real acquisition for the displayed area. "
            "For real analysis, use the live Sentinel-2 pipeline introduced on Day 3."
        ),
    },
}
