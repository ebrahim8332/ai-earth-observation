"""
geocoder.py — Convert a place name to a bounding box.

Uses two geocoding services in sequence:
  1. ArcGIS World Geocoding Service — free, no API key, very reliable under load.
  2. OpenStreetMap Nominatim — free, no API key, rate-limited (1 req/sec).

Returns a bounding box [min_lon, min_lat, max_lon, max_lat] suitable for STAC/GEE queries.
Returns None if both services fail.
"""

import time
import requests

# Default bbox size in degrees when a geocoder returns a point rather than a polygon.
# 0.5 degrees is roughly 50 km — a reasonable default for city-level queries.
DEFAULT_BBOX_SIZE_DEG = 0.5

# Nominatim requires a descriptive User-Agent.
NOMINATIM_HEADERS = {
    "User-Agent": "EOIL-Portal/1.5 (AI-Native Earth Observation Innovation Lab; contact: eoil@example.com)"
}


def geocode_place(place_name: str) -> list | None:
    """Convert a place name to a bounding box [min_lon, min_lat, max_lon, max_lat].

    Tries ArcGIS first (more reliable on shared IPs), then Nominatim as backup.
    Returns None if both services fail or return no results.
    """
    if not place_name or not place_name.strip():
        return None

    name = place_name.strip()

    bbox = _geocode_arcgis(name)
    if bbox:
        return bbox

    bbox = _geocode_nominatim(name)
    return bbox


# ---------------------------------------------------------------------------
# ArcGIS World Geocoding Service — primary
# Free for light use, no API key required. Very reliable under load.
# Returns an extent object (bounding box) for region-level queries.
# ---------------------------------------------------------------------------

def _geocode_arcgis(place_name: str) -> list | None:
    """Try the ArcGIS World Geocoding Service and return a bbox, or None."""
    url    = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    params = {
        "SingleLine": place_name,
        "f":          "json",
        "maxLocations": 1,
        "outFields":  "Addr_type",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data       = response.json()
        candidates = data.get("candidates", [])

        if not candidates:
            return None

        candidate = candidates[0]
        extent    = candidate.get("extent")

        if extent:
            # ArcGIS returns extent as {xmin, ymin, xmax, ymax} — already in lon/lat
            return [extent["xmin"], extent["ymin"], extent["xmax"], extent["ymax"]]

        # Fall back to building a bbox around the returned point
        loc  = candidate.get("location", {})
        lon  = float(loc.get("x", 0))
        lat  = float(loc.get("y", 0))
        half = DEFAULT_BBOX_SIZE_DEG / 2
        return [lon - half, lat - half, lon + half, lat + half]

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Nominatim (OpenStreetMap) — backup
# Requires 1-second delay between requests per Nominatim usage policy.
# May be rate-limited on shared cloud IPs under high load.
# ---------------------------------------------------------------------------

def _geocode_nominatim(place_name: str) -> list | None:
    """Try Nominatim and return a bbox, or None."""
    url    = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":              place_name,
        "format":         "json",
        "limit":          1,
        "addressdetails": 0,
    }
    try:
        time.sleep(1)   # Nominatim usage policy: max 1 request per second
        response = requests.get(url, params=params, headers=NOMINATIM_HEADERS, timeout=10)
        response.raise_for_status()
        results = response.json()

        if not results:
            return None

        result = results[0]

        # Nominatim boundingbox is [south, north, west, east]
        if "boundingbox" in result:
            south, north, west, east = [float(x) for x in result["boundingbox"]]
            return [west, south, east, north]

        lat  = float(result["lat"])
        lon  = float(result["lon"])
        half = DEFAULT_BBOX_SIZE_DEG / 2
        return [lon - half, lat - half, lon + half, lat + half]

    except Exception:
        return None


def bbox_dims_km(bbox: list) -> tuple:
    """
    Return the approximate (width_km, height_km) of a bounding box.
    Used to display size to the user and warn if the area is too large for SAR.
    """
    import math
    lon_diff  = abs(bbox[2] - bbox[0])
    lat_diff  = abs(bbox[3] - bbox[1])
    mid_lat   = (bbox[1] + bbox[3]) / 2
    km_per_lon = 111.0 * math.cos(math.radians(mid_lat))
    km_per_lat = 111.0
    return round(lon_diff * km_per_lon, 0), round(lat_diff * km_per_lat, 0)


def bbox_area_km2(bbox: list) -> float:
    """
    Estimate the area of a bounding box in square kilometres.
    Used to warn the user if the bbox is too large for a meaningful satellite render.
    Approximation only — accurate enough for display purposes.
    """
    import math
    lon_diff = abs(bbox[2] - bbox[0])
    lat_diff = abs(bbox[3] - bbox[1])
    # 1 degree latitude ≈ 111 km. 1 degree longitude ≈ 111 * cos(lat) km.
    mid_lat   = (bbox[1] + bbox[3]) / 2
    km_per_lon = 111.0 * math.cos(math.radians(mid_lat))
    km_per_lat = 111.0
    return round(lon_diff * km_per_lon * lat_diff * km_per_lat, 0)
