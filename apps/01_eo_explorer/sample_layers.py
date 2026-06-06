"""
sample_layers.py — Hardcoded sample GeoJSON features for the Day 1 map.
These stand in for live satellite data layers until real data pipelines are added.
Each feature includes name, description, and type properties for the layer control.
"""

# Georgia Infrastructure Corridor
# A LineString tracing a sample gas transmission pipeline route across Georgia.
GEORGIA_PIPELINE: dict = {
    "type": "Feature",
    "properties": {
        "name": "Georgia Infrastructure Corridor",
        "description": "Sample gas transmission pipeline corridor in Georgia",
        "type": "Infrastructure",
    },
    "geometry": {
        "type": "LineString",
        "coordinates": [
            [-84.39, 33.75],
            [-83.80, 33.45],
            [-83.20, 33.10],
            [-82.60, 32.80],
            [-82.00, 32.50],
        ],
    },
}

# California Agriculture Region
# A Polygon covering a sample irrigated agriculture area in the Central Valley.
CALIFORNIA_AGRICULTURE: dict = {
    "type": "Feature",
    "properties": {
        "name": "California Agriculture Region",
        "description": "Sample irrigated agriculture area in California Central Valley",
        "type": "Agriculture",
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [-120.50, 36.80],
                [-119.80, 36.80],
                [-119.80, 36.20],
                [-120.50, 36.20],
                [-120.50, 36.80],
            ]
        ],
    },
}

# Houston Atmospheric Hotspot
# A Point marking a sample industrial emissions monitoring location near Houston.
HOUSTON_HOTSPOT: dict = {
    "type": "Feature",
    "properties": {
        "name": "Houston Atmospheric Hotspot",
        "description": "Sample industrial emissions monitoring location near Houston",
        "type": "Atmospheric",
    },
    "geometry": {
        "type": "Point",
        "coordinates": [-95.37, 29.76],
    },
}

# Amazon Environmental Monitoring Zone
# A Polygon covering a sample deforestation monitoring area in the Amazon Basin.
AMAZON_MONITORING: dict = {
    "type": "Feature",
    "properties": {
        "name": "Amazon Environmental Monitoring Zone",
        "description": "Sample deforestation monitoring area in the Amazon Basin",
        "type": "Environmental",
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [-62.00, -3.00],
                [-58.00, -3.00],
                [-58.00, -7.00],
                [-62.00, -7.00],
                [-62.00, -3.00],
            ]
        ],
    },
}

# All four features collected into a GeoJSON FeatureCollection.
# map_builder.py uses this to add all layers in one call.
ALL_LAYERS: dict = {
    "type": "FeatureCollection",
    "features": [
        GEORGIA_PIPELINE,
        CALIFORNIA_AGRICULTURE,
        HOUSTON_HOTSPOT,
        AMAZON_MONITORING,
    ],
}
