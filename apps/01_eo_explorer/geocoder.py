"""
geocoder.py — Convert a place name to a bounding box.

Uses two geocoding services in sequence:
  1. ArcGIS World Geocoding Service — free, no API key, very reliable under load.
  2. OpenStreetMap Nominatim — free, no API key, rate-limited (1 req/sec).

Returns a bounding box [min_lon, min_lat, max_lon, max_lat] suitable for STAC/GEE queries.
Returns (None, None) if both services fail.

Ambiguous or low-confidence matches (e.g. "Georgia" matching the country when a
US state was meant, or a place name with several similarly-named results) are
not silently resolved to whichever candidate the service ranks first. The
match is still returned — the caller always gets a usable bbox — but a
warning string describing exactly what was matched is returned alongside it,
so the UI can tell the user which location was actually used and let them
narrow the query if it is wrong.
"""

import time
import requests

# Default bbox size in degrees when a geocoder returns a point rather than a polygon.
# 0.5 degrees is roughly 50 km — a reasonable default for city-level queries.
DEFAULT_BBOX_SIZE_DEG = 0.5

# ArcGIS match score (0-100) below which a result is flagged as low-confidence
# rather than silently accepted as the intended location.
ARCGIS_CONFIDENCE_THRESHOLD = 80

# Nominatim requires a descriptive User-Agent.
NOMINATIM_HEADERS = {
    "User-Agent": "EOIL-Portal/1.5 (AI-Native Earth Observation Innovation Lab; contact: eoil@example.com)"
}


def geocode_place(place_name: str) -> tuple:
    """Convert a place name to a bounding box [min_lon, min_lat, max_lon, max_lat].

    Tries ArcGIS first (more reliable on shared IPs), then Nominatim as backup.

    Returns (bbox, warning). bbox is None if both services fail or return no
    results. warning is None when the match looks solid, or a plain-English
    string naming the actual matched location when the match was ambiguous
    or low-confidence — the caller should show this to the user rather than
    discard it, since a silently-wrong bbox feeds wrong coordinates into
    every downstream analytical module.
    """
    if not place_name or not place_name.strip():
        return None, None

    name = place_name.strip()

    bbox, warning = _geocode_arcgis(name)
    if bbox:
        return bbox, warning

    bbox, warning = _geocode_nominatim(name)
    return bbox, warning


# ---------------------------------------------------------------------------
# ArcGIS World Geocoding Service — primary
# Free for light use, no API key required. Very reliable under load.
# Returns an extent object (bounding box) for region-level queries.
# ---------------------------------------------------------------------------

def _geocode_arcgis(place_name: str) -> tuple:
    """Try the ArcGIS World Geocoding Service and return (bbox, warning), or (None, None).

    ArcGIS returns a 0-100 match "score" per candidate. A low score means the
    service is not confident this candidate is what the user meant (common
    for short or generic names — "Georgia" matches the country before the
    US state). Requesting 3 candidates also lets us tell a single clear match
    apart from several similarly-ranked ones.
    """
    url    = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    params = {
        "SingleLine": place_name,
        "f":          "json",
        "maxLocations": 3,
        "outFields":  "Addr_type",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data       = response.json()
        candidates = data.get("candidates", [])

        if not candidates:
            return None, None

        candidate = candidates[0]
        score     = candidate.get("score", 100)
        address   = candidate.get("address", place_name)

        warning = None
        if score < ARCGIS_CONFIDENCE_THRESHOLD:
            warning = (
                f"Low-confidence match for \"{place_name}\" — using \"{address}\" "
                f"(match score {score}/100). If this isn't the location you meant, "
                f"try adding a country or region name."
            )
        elif len(candidates) > 1:
            other = candidates[1].get("address", "")
            other_score = candidates[1].get("score", 0)
            # Only flag ambiguity when the second candidate is both a close
            # scoring match AND a genuinely different place — ArcGIS sometimes
            # returns near-duplicate candidates for the same location, which
            # would otherwise produce a confusing "second match" identical to
            # the one just used.
            if other_score >= score - 5 and other and other != address:
                warning = (
                    f"Multiple similar matches for \"{place_name}\" — using \"{address}\". "
                    f"A close second match was \"{other}\". Try a more specific name if "
                    f"this isn't the location you meant."
                )

        extent = candidate.get("extent")

        if extent:
            # ArcGIS returns extent as {xmin, ymin, xmax, ymax} — already in lon/lat
            bbox = [extent["xmin"], extent["ymin"], extent["xmax"], extent["ymax"]]
        else:
            # Fall back to building a bbox around the returned point
            loc  = candidate.get("location", {})
            lon  = float(loc.get("x", 0))
            lat  = float(loc.get("y", 0))
            half = DEFAULT_BBOX_SIZE_DEG / 2
            bbox = [lon - half, lat - half, lon + half, lat + half]

        return bbox, warning

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Nominatim (OpenStreetMap) — backup
# Requires 1-second delay between requests per Nominatim usage policy.
# May be rate-limited on shared cloud IPs under high load.
# ---------------------------------------------------------------------------

def _geocode_nominatim(place_name: str) -> tuple:
    """Try Nominatim and return (bbox, warning), or (None, None).

    Requests 3 results instead of 1 so a genuinely ambiguous query (multiple
    distinct places with similar names) can be flagged instead of silently
    resolved to whichever result Nominatim's internal ranking put first.
    """
    url    = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":              place_name,
        "format":         "json",
        "limit":          3,
        "addressdetails": 0,
    }
    try:
        time.sleep(1)   # Nominatim usage policy: max 1 request per second
        response = requests.get(url, params=params, headers=NOMINATIM_HEADERS, timeout=10)
        response.raise_for_status()
        results = response.json()

        if not results:
            return None, None

        result       = results[0]
        display_name = result.get("display_name", place_name)

        warning = None
        if len(results) > 1 and results[1].get("display_name") != display_name:
            warning = (
                f"Multiple possible matches for \"{place_name}\" — using "
                f"\"{display_name}\". Try a more specific name (e.g. add a country) "
                f"if this isn't the location you meant."
            )

        # Nominatim boundingbox is [south, north, west, east]
        if "boundingbox" in result:
            south, north, west, east = [float(x) for x in result["boundingbox"]]
            bbox = [west, south, east, north]
        else:
            lat  = float(result["lat"])
            lon  = float(result["lon"])
            half = DEFAULT_BBOX_SIZE_DEG / 2
            bbox = [lon - half, lat - half, lon + half, lat + half]

        return bbox, warning

    except Exception:
        return None, None


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
