"""
geocoder.py — Convert a place name to a bounding box using OpenStreetMap Nominatim.

Nominatim is OpenStreetMap's free geocoding service. No API key required.
Returns a bounding box [min_lon, min_lat, max_lon, max_lat] suitable for STAC queries.
If geocoding fails, returns None and the caller falls back to a default location.
"""

import requests
import time

# Nominatim requires a User-Agent header identifying the application.
# Using the project name as the identifier.
NOMINATIM_HEADERS = {
    "User-Agent": "EOIL-SpectralExplorer/1.0 (AI-Native Earth Observation Innovation Lab)"
}

# Default bbox size in degrees when Nominatim returns a point rather than a polygon.
# 0.3 degrees is roughly 30 km — a reasonable default for city-level queries.
DEFAULT_BBOX_SIZE_DEG = 0.3


def geocode_place(place_name: str) -> list | None:
    """
    Search for a place name and return a bounding box [min_lon, min_lat, max_lon, max_lat].

    Uses the Nominatim API. If the place has a known polygon boundary, that is returned.
    If only a point is found, a square bbox of DEFAULT_BBOX_SIZE_DEG is built around it.
    Returns None if nothing is found or if the request fails.
    """
    if not place_name or not place_name.strip():
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":              place_name.strip(),
        "format":         "json",
        "limit":          1,
        "addressdetails": 0,
    }

    try:
        # Nominatim asks for a 1-second delay between requests.
        # In a web app this runs once per user action, so delay is minimal.
        response = requests.get(url, params=params, headers=NOMINATIM_HEADERS, timeout=10)
        response.raise_for_status()
        results = response.json()

        if not results:
            return None

        result = results[0]

        # Nominatim returns a boundingbox as [south, north, west, east] strings.
        # We reorder to [min_lon, min_lat, max_lon, max_lat] for STAC compatibility.
        if "boundingbox" in result:
            south, north, west, east = [float(x) for x in result["boundingbox"]]
            return [west, south, east, north]

        # If no bounding box, build one around the returned point.
        lat = float(result["lat"])
        lon = float(result["lon"])
        half = DEFAULT_BBOX_SIZE_DEG / 2
        return [lon - half, lat - half, lon + half, lat + half]

    except Exception:
        return None


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
