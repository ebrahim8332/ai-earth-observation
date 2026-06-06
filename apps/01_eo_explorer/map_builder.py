"""
map_builder.py — Builds and returns Folium map objects.
Accepts a location name, centers the map, adds all sample layers with styling,
and returns a ready-to-render Folium map object.
app.py passes the returned object directly to st_folium().
"""

import folium
from folium.plugins import MeasureControl
import sample_layers

# LOCATIONS: display name -> [latitude, longitude, zoom level]
# These are the four locations available in the sidebar location selector.
LOCATIONS: dict = {
    "Georgia, USA": [32.5, -83.5, 6],
    "California Central Valley": [36.5, -120.0, 7],
    "Houston Gulf Coast": [29.76, -95.37, 8],
    "Amazon Basin": [-5.0, -60.0, 5],
}

# Layer styling constants — centralised here so any future style change
# only requires editing this section.
PIPELINE_STYLE = {"color": "orange", "weight": 3}
AGRICULTURE_STYLE = {"color": "green", "fillColor": "green", "fillOpacity": 0.4, "weight": 2}
HOTSPOT_RADIUS = 10
HOTSPOT_COLOR = "red"
AMAZON_STYLE = {"color": "darkgreen", "fillColor": "darkgreen", "fillOpacity": 0.4, "weight": 2}


def build_map(location_name: str) -> folium.Map:
    """
    Build a Folium map centred on the given location with all sample layers added.

    Args:
        location_name: must match a key in LOCATIONS.

    Returns:
        A configured folium.Map object ready to pass to st_folium().
    """
    # Look up centre coordinates and zoom for the selected location.
    # Fall back to Georgia if the name is somehow not in the dict.
    lat, lon, zoom = LOCATIONS.get(location_name, LOCATIONS["Georgia, USA"])

    # Create the base map using OpenStreetMap tiles (free, no token required).
    fmap = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")

    # Add a scale bar and measurement tool so users can gauge distances.
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(fmap)

    # Add each sample layer to its own FeatureGroup so the layer control
    # lets users toggle them on and off independently.
    _add_pipeline_layer(fmap)
    _add_agriculture_layer(fmap)
    _add_hotspot_layer(fmap)
    _add_amazon_layer(fmap)

    # LayerControl renders a toggle panel in the top-right corner of the map.
    folium.LayerControl(collapsed=False).add_to(fmap)

    return fmap


def _add_pipeline_layer(fmap: folium.Map) -> None:
    """Add the Georgia pipeline corridor as an orange LineString layer."""
    group = folium.FeatureGroup(name="Georgia Infrastructure Corridor", show=True)
    folium.GeoJson(
        sample_layers.GEORGIA_PIPELINE,
        style_function=lambda feature: PIPELINE_STYLE,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "description"],
            aliases=["Layer", "Description"],
        ),
    ).add_to(group)
    group.add_to(fmap)


def _add_agriculture_layer(fmap: folium.Map) -> None:
    """Add the California agriculture polygon as a green-filled layer."""
    group = folium.FeatureGroup(name="California Agriculture Region", show=True)
    folium.GeoJson(
        sample_layers.CALIFORNIA_AGRICULTURE,
        style_function=lambda feature: AGRICULTURE_STYLE,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "description"],
            aliases=["Layer", "Description"],
        ),
    ).add_to(group)
    group.add_to(fmap)


def _add_hotspot_layer(fmap: folium.Map) -> None:
    """Add the Houston atmospheric hotspot as a red circle marker."""
    group = folium.FeatureGroup(name="Houston Atmospheric Hotspot", show=True)
    coords = sample_layers.HOUSTON_HOTSPOT["geometry"]["coordinates"]
    props = sample_layers.HOUSTON_HOTSPOT["properties"]
    # GeoJSON uses [lon, lat] order; Folium uses [lat, lon]
    folium.CircleMarker(
        location=[coords[1], coords[0]],
        radius=HOTSPOT_RADIUS,
        color=HOTSPOT_COLOR,
        fill=True,
        fill_color=HOTSPOT_COLOR,
        fill_opacity=0.7,
        tooltip=f"{props['name']}: {props['description']}",
    ).add_to(group)
    group.add_to(fmap)


def _add_amazon_layer(fmap: folium.Map) -> None:
    """Add the Amazon monitoring zone as a dark green filled polygon."""
    group = folium.FeatureGroup(name="Amazon Environmental Monitoring Zone", show=True)
    folium.GeoJson(
        sample_layers.AMAZON_MONITORING,
        style_function=lambda feature: AMAZON_STYLE,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "description"],
            aliases=["Layer", "Description"],
        ),
    ).add_to(group)
    group.add_to(fmap)
