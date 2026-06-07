"""
map_picker.py — Shared click-to-set-location map widget for the EOIL portal.

Used by all five portal modules as an optional override to text geocoding.
The user types a place name as normal. The map centres on that location.
They can then click anywhere on the map to set an exact centre point, and
use the slider to control the analysis bbox size.

Usage in each module (after geocoding block):

    import map_picker
    picked = map_picker.render_map_picker(
        centre_bbox = module_bbox,   # from geocoder
        picker_key  = "ii",          # module prefix — must be unique per module
        default_size_km = 50,
    )
    if picked:
        module_bbox = picked  # override geocoded bbox with map pick

Session state keys (all prefixed by picker_key):
    {key}_map_click   — (lat, lon) tuple of the last user click, or None
    {key}_map_size_km — current slider value in km
"""

import math
import folium
import streamlit as st
from streamlit_folium import st_folium


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

_KM_PER_DEG_LAT = 111.0  # approximately constant globally


def _km_to_half_deg(km, centre_lat):
    """
    Convert a km radius to half-side degrees for lat and lon.

    Latitude degrees are constant (~111 km each).
    Longitude degrees shrink towards the poles — adjusted by cos(lat).

    Returns (lat_half_deg, lon_half_deg) for the half-side of the bbox.
    """
    lat_half = (km / 2) / _KM_PER_DEG_LAT
    cos_lat  = math.cos(math.radians(centre_lat))
    # Guard against cos(lat) approaching 0 near the poles
    cos_lat  = max(cos_lat, 0.01)
    lon_half = (km / 2) / (_KM_PER_DEG_LAT * cos_lat)
    return lat_half, lon_half


def _build_bbox(lat, lon, size_km):
    """
    Build a [min_lon, min_lat, max_lon, max_lat] bbox centred on (lat, lon).
    size_km is the full side length — the bbox is size_km × size_km.
    """
    lat_half, lon_half = _km_to_half_deg(size_km, lat)
    return [
        lon - lon_half,  # min_lon (west)
        lat - lat_half,  # min_lat (south)
        lon + lon_half,  # max_lon (east)
        lat + lat_half,  # max_lat (north)
    ]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def render_map_picker(
    centre_bbox,
    picker_key,
    default_size_km = 50,
):
    """
    Render a click-to-set-location Folium map with a size slider.

    The map centres on the geocoded location (centre_bbox). The user clicks
    anywhere to place the analysis bbox. A blue rectangle shows what area
    will be analysed at the current slider size.

    Args:
        centre_bbox:     [min_lon, min_lat, max_lon, max_lat] from geocoder.
                         Used to centre the map. Not the returned bbox.
        picker_key:      Unique prefix for session state keys. Use the module
                         prefix (e.g. "ii", "sar", "cd", "ts", "se").
        default_size_km: Default bbox side length in km. Slider default.

    Returns:
        [min_lon, min_lat, max_lon, max_lat] if the user has clicked.
        None if the user has not clicked yet — caller uses geocoded bbox.
    """
    click_key = f"{picker_key}_map_click"
    size_key  = f"{picker_key}_map_size_km"

    # Compute map centre from the geocoded bbox
    centre_lat = (centre_bbox[1] + centre_bbox[3]) / 2
    centre_lon = (centre_bbox[0] + centre_bbox[2]) / 2

    # Estimate zoom level from bbox width
    bbox_width_deg = centre_bbox[2] - centre_bbox[0]
    if   bbox_width_deg < 0.1:  zoom = 13
    elif bbox_width_deg < 0.5:  zoom = 11
    elif bbox_width_deg < 1.5:  zoom = 9
    elif bbox_width_deg < 5.0:  zoom = 7
    elif bbox_width_deg < 15.0: zoom = 5
    else:                        zoom = 4

    # Size slider — above the map so the rectangle updates when it changes
    size_km = st.slider(
        "Analysis area size (km × km)",
        min_value = 25,
        max_value = 500,
        value     = st.session_state.get(size_key, default_size_km),
        step      = 25,
        key       = size_key,
        help      = (
            "Controls the bbox built around your clicked point. "
            "25 km = city district or port. "
            "100 km = large island or metropolitan area. "
            "300 km+ = regional analysis."
        ),
    )

    # Current click state
    last_click = st.session_state.get(click_key)

    # Build the Folium map
    m = folium.Map(
        location   = [centre_lat, centre_lon],
        zoom_start = zoom,
        tiles      = "CartoDB positron",  # clean light basemap, no API key needed
    )

    # If the user has already clicked, draw the bbox rectangle
    if last_click:
        click_lat, click_lon = last_click
        bbox = _build_bbox(click_lat, click_lon, size_km)

        # Draw the analysis rectangle
        folium.Rectangle(
            bounds       = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color        = "#1a73e8",
            weight       = 2,
            fill         = True,
            fill_color   = "#1a73e8",
            fill_opacity = 0.08,
            tooltip      = f"Analysis area — {size_km} km × {size_km} km",
        ).add_to(m)

        # Draw a small marker at the click centre
        folium.CircleMarker(
            location    = [click_lat, click_lon],
            radius      = 5,
            color       = "#1a73e8",
            fill        = True,
            fill_color  = "#1a73e8",
            fill_opacity= 1.0,
            tooltip     = f"Centre: {click_lat:.4f}°, {click_lon:.4f}°",
        ).add_to(m)

    # Render the interactive map and capture click events
    map_output = st_folium(
        m,
        width            = 700,
        height           = 360,
        returned_objects = ["last_clicked"],
        key              = f"{picker_key}_folium_picker",
    )

    # Process a new click — update session state and rerun to redraw the rectangle
    if map_output and map_output.get("last_clicked"):
        clicked  = map_output["last_clicked"]
        new_lat  = clicked["lat"]
        new_lon  = clicked["lng"]
        new_click = (new_lat, new_lon)

        # Only trigger a rerun if the click point actually changed
        if new_click != last_click:
            st.session_state[click_key] = new_click
            st.rerun()

    # Instruction / status line below the map
    if last_click:
        click_lat, click_lon = last_click
        bbox = _build_bbox(click_lat, click_lon, size_km)
        st.caption(
            f"✅ Map area active — centre {click_lat:.4f}°, {click_lon:.4f}° — "
            f"{size_km} km × {size_km} km — "
            f"bbox {[round(x, 3) for x in bbox]}. "
            f"Click elsewhere on the map to move the area."
        )
        return bbox
    else:
        st.caption(
            "Click anywhere on the map to set the exact analysis area. "
            "The blue rectangle shows what will be analysed. "
            "Use the slider above to adjust the area size."
        )
        return None


# ---------------------------------------------------------------------------
# Helper: clear click state for a module (call when location text changes)
# ---------------------------------------------------------------------------

def clear_click(picker_key):
    """
    Clear the stored map click for a module.
    Call this when the user types a new location so the old map click
    does not carry over to the new location.
    """
    click_key = f"{picker_key}_map_click"
    if click_key in st.session_state:
        del st.session_state[click_key]
