import streamlit as st

st.set_page_config(page_title="EO Explorer", page_icon="🛰️", layout="wide")
st.title("🛰️ EO Explorer — Import Diagnostics")

modules = [
    ("streamlit_folium", "from streamlit_folium import st_folium"),
    ("folium",           "import folium"),
    ("folium.plugins",   "from folium.plugins import MeasureControl"),
    ("numpy",            "import numpy as np"),
    ("plotly",           "import plotly.graph_objects as go"),
    ("PIL",              "from PIL import Image"),
    ("requests",         "import requests"),
    ("dotenv",           "from dotenv import load_dotenv"),
    ("pystac_client",    "import pystac_client"),
    ("planetary_computer","import planetary_computer"),
    ("groq",             "from groq import Groq"),
    ("google.genai",     "from google import genai"),
    ("config",           "import config"),
    ("data_catalog",     "import data_catalog"),
    ("sample_layers",    "import sample_layers"),
    ("map_builder",      "import map_builder"),
    ("satellite_catalog","import satellite_catalog"),
    ("geocoder",         "import geocoder"),
    ("ai_assistant",     "import ai_assistant"),
    ("spectral_explorer","import spectral_explorer"),
]

for name, stmt in modules:
    try:
        exec(stmt)
        st.success(f"✅ {name}")
    except Exception as e:
        st.error(f"❌ {name}: {e}")
