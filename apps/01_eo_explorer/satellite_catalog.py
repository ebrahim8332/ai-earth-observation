"""
satellite_catalog.py — Band definitions, named presets, and STAC collection IDs.

One entry per satellite. Each entry defines:
  - collection:     the STAC collection ID used to search Planetary Computer
  - cloud_field:    the metadata field name for cloud cover (differs by satellite)
  - rescale:        the pixel value range to map to 0-255 for display
  - bands:          dict of band_id -> {name, wavelength, resolution, description}
  - presets:        named band combinations that auto-fill the R/G/B dropdowns
  - sar:            True if this is a radar satellite (displayed differently)
"""

SATELLITES: dict = {

    "Sentinel-2 L2A": {
        "collection":   "sentinel-2-l2a",
        "cloud_field":  "eo:cloud_cover",
        "rescale":      "0,3000",
        "sar":          False,
        "color_formula": "Gamma RGB 3.5",
        "bands": {
            "B02": {"name": "Blue",             "wavelength": "490 nm",  "resolution": "10 m",  "description": "Visible blue light. Water, aerosols."},
            "B03": {"name": "Green",            "wavelength": "560 nm",  "resolution": "10 m",  "description": "Visible green light. Vegetation peak reflectance."},
            "B04": {"name": "Red",              "wavelength": "665 nm",  "resolution": "10 m",  "description": "Visible red light. Chlorophyll absorption."},
            "B05": {"name": "Red Edge 1",       "wavelength": "705 nm",  "resolution": "20 m",  "description": "Transition zone between red and NIR. Vegetation stress."},
            "B06": {"name": "Red Edge 2",       "wavelength": "740 nm",  "resolution": "20 m",  "description": "Red edge. Canopy chlorophyll content."},
            "B07": {"name": "Red Edge 3",       "wavelength": "783 nm",  "resolution": "20 m",  "description": "Red edge. LAI and chlorophyll estimation."},
            "B08": {"name": "NIR",              "wavelength": "842 nm",  "resolution": "10 m",  "description": "Near infrared. Vegetation biomass. Used in NDVI."},
            "B8A": {"name": "NIR Narrow",       "wavelength": "865 nm",  "resolution": "20 m",  "description": "Narrow near infrared. More precise than B08."},
            "B09": {"name": "Water Vapour",     "wavelength": "945 nm",  "resolution": "60 m",  "description": "Atmospheric water vapour correction band."},
            "B11": {"name": "SWIR 1",           "wavelength": "1610 nm", "resolution": "20 m",  "description": "Shortwave infrared. Soil moisture, snow, burn scars."},
            "B12": {"name": "SWIR 2",           "wavelength": "2190 nm", "resolution": "20 m",  "description": "Shortwave infrared. Geology, soil, mineral mapping."},
        },
        "presets": {
            "True Color":               {"r": "B04", "g": "B03", "b": "B02", "note": "Natural colour. What the eye sees from space."},
            "False Color — Vegetation": {"r": "B08", "g": "B04", "b": "B03", "note": "Vegetation = bright red. Best for forest vs. farmland vs. urban."},
            "False Color — Agriculture":{"r": "B11", "g": "B08", "b": "B02", "note": "Crop stress and irrigation. Fallow fields = bright. Active crops = green."},
            "False Color — Urban":      {"r": "B12", "g": "B11", "b": "B04", "note": "Built-up areas = purple/magenta. Bare soil = cyan. Vegetation = orange."},
            "Atmospheric Penetration":  {"r": "B12", "g": "B11", "b": "B08", "note": "Cuts through haze and smoke. Useful in dry or dusty regions."},
            "Burn Scar":                {"r": "B12", "g": "B08", "b": "B04", "note": "Recent fire damage = dark red. Healthy vegetation = bright green."},
            "Water Bodies":             {"r": "B08", "g": "B03", "b": "B02", "note": "Open water = very dark. Vegetation = bright. Good for flood mapping."},
            "Geology / Minerals":       {"r": "B12", "g": "B04", "b": "B02", "note": "Rock type discrimination. Useful for mineral and soil surveys."},
        },
    },

    "Landsat 8/9": {
        "collection":   "landsat-c2-l2",
        "cloud_field":  "eo:cloud_cover",
        "rescale":      "7000,22000",
        "sar":          False,
        "color_formula": "Gamma RGB 3.5",
        "bands": {
            "blue":   {"name": "Blue",          "wavelength": "485 nm",  "resolution": "30 m",  "description": "Visible blue. Water, aerosols, coastal areas."},
            "green":  {"name": "Green",         "wavelength": "560 nm",  "resolution": "30 m",  "description": "Visible green. Vegetation and water assessment."},
            "red":    {"name": "Red",           "wavelength": "660 nm",  "resolution": "30 m",  "description": "Visible red. Vegetation and soil discrimination."},
            "nir08":  {"name": "NIR",           "wavelength": "865 nm",  "resolution": "30 m",  "description": "Near infrared. Vegetation biomass. Used in NDVI."},
            "swir16": {"name": "SWIR 1",        "wavelength": "1610 nm", "resolution": "30 m",  "description": "Shortwave infrared. Soil moisture, snow, burn scars."},
            "swir22": {"name": "SWIR 2",        "wavelength": "2200 nm", "resolution": "30 m",  "description": "Shortwave infrared. Geology and mineral mapping."},
            "lwir11": {"name": "Thermal (TIRS)","wavelength": "10900 nm","resolution": "100 m", "description": "Land surface temperature. Urban heat islands, geothermal."},
        },
        "presets": {
            "True Color":               {"r": "red",   "g": "green", "b": "blue",   "note": "Natural colour."},
            "False Color — Vegetation": {"r": "nir08", "g": "red",   "b": "green",  "note": "Vegetation = bright red. 50-year archive for trend analysis."},
            "False Color — Urban":      {"r": "swir22","g": "swir16","b": "red",    "note": "Urban areas = purple/magenta. Same logic as Sentinel-2 urban preset."},
            "Thermal":                  {"r": "lwir11","g": "lwir11","b": "lwir11", "note": "Surface temperature as greyscale. Hot = bright. Cold = dark."},
            "Water Bodies":             {"r": "nir08", "g": "green", "b": "blue",   "note": "Water = very dark NIR. Good for reservoir and flood mapping."},
        },
    },

    "Sentinel-1 SAR": {
        "collection":   "sentinel-1-grd",
        "cloud_field":  None,
        "rescale":      "-25,0",
        "sar":          True,
        "color_formula": "",
        "bands": {
            "vv": {"name": "VV",          "wavelength": "C-band 5.5 cm", "resolution": "10 m", "description": "Co-polarisation. Strong return from buildings, calm water is dark."},
            "vh": {"name": "VH",          "wavelength": "C-band 5.5 cm", "resolution": "10 m", "description": "Cross-polarisation. Sensitive to volume scattering (vegetation, rough terrain)."},
        },
        "presets": {
            "SAR False Color (VV/VH/VV)": {"r": "vv", "g": "vh", "b": "vv", "note": "Standard SAR false color. Urban = bright blue/white. Vegetation = green. Water = black."},
            "VV Only (Grayscale)":         {"r": "vv", "g": "vv", "b": "vv", "note": "Single polarisation grayscale. Buildings = bright. Calm water = dark."},
            "VH Only (Grayscale)":         {"r": "vh", "g": "vh", "b": "vh", "note": "Cross-pol grayscale. More sensitive to vegetation texture."},
        },
    },

}

# Preset locations: name -> [min_lon, min_lat, max_lon, max_lat]
PRESET_LOCATIONS: dict = {
    "Zanzibar, Tanzania":             [39.10, -6.50, 39.55, -5.65],
    "Moscow, Russia":                 [37.20, 55.55, 37.90, 55.95],
    "Central Tanzania (Serengeti)":   [34.50, -3.00, 35.50, -2.00],
    "Alpharetta, Georgia USA":        [-84.35, 34.00, -84.15, 34.15],
    "California Central Valley":      [-119.80, 36.20, -119.00, 36.90],
    "Amazon Basin, Brazil":           [-62.00, -5.00, -60.50, -3.50],
    "Nile Delta, Egypt":              [30.00, 30.50, 32.50, 31.80],
}
