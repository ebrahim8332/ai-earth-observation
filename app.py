"""
app.py — EOIL Portal v1.3
Streamlit layout and component wiring only. No business logic here.
All logic is imported from other modules.

v1.0  Day 1: EO Explorer foundational app
v1.1  Day 3: Spectral Explorer tab added
v1.2  Day 6: Time Series Explorer module added; sidebar navigation replaces tab navigation
v1.3  Day 6: Spectral Explorer promoted to standalone sidebar module; tabs removed
"""

import streamlit as st
from streamlit_folium import st_folium
from datetime import date, timedelta
import numpy as np

import config
import data_catalog
import sample_layers
import map_builder
import ai_assistant
import satellite_catalog
import geocoder
import spectral_explorer
import gee_timeseries

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EOIL Portal",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 EOIL Portal")
st.caption("Earth Observation Innovation Lab — satellite data analysis and AI interpretation.")

# Pointer cursor on all interactive controls
st.markdown("""
<style>
div[data-baseweb="select"] > div,
div[data-baseweb="select"] input,
button[kind="primary"],
button[kind="secondary"],
.stButton > button,
.stDownloadButton > button,
input[type="range"] {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — module navigation (always visible at the top)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Module")
    selected_module = st.radio(
        "Navigate",
        ["🗺️ EO Explorer", "🔬 Spectral Explorer", "📈 Time Series Explorer"],
        label_visibility="collapsed",
    )
    st.divider()

# ---------------------------------------------------------------------------
# GEE status — check once per session and cache the result
# If GEE credentials are in Streamlit secrets, live queries are available.
# If not, the Time Series Explorer uses sample data.
# ---------------------------------------------------------------------------
if "gee_available" not in st.session_state:
    st.session_state.gee_available = gee_timeseries.init_gee()

# ---------------------------------------------------------------------------
# Helper: "What am I looking at?" expander content
# ---------------------------------------------------------------------------

def render_what_am_i_looking_at(sat_key: str, r_band: str, g_band: str, b_band: str,
                                  preset_name: str = "", location_label: str = "",
                                  is_contact_sheet: bool = False):
    """
    Render an expandable explanation panel below a satellite image.
    Contains two sections:
      1. A fixed primer on what satellite imagery is (vs a photograph)
      2. A dynamic explanation of the specific band combination displayed
    """
    with st.expander("ℹ️ What am I looking at? — Understanding satellite imagery", expanded=False):

        # -------------------------------------------------------------------
        # Section 1: What satellite imagery is
        # -------------------------------------------------------------------
        st.markdown("### Satellite imagery is not a photograph")
        st.markdown("""
A camera — in a phone, a drone, or an aircraft — captures what the human eye sees.
It records reflected light in three wavelength ranges: red, green, and blue. That is
the full extent of what a photograph captures.

A satellite multispectral sensor is fundamentally different. It measures reflected
energy across many separate wavelength ranges simultaneously — including ranges that
are completely invisible to the human eye. Sentinel-2 measures 12 separate wavelength
ranges. Landsat measures 11. Each range reveals different physical properties of the
ground surface.

When you look at a satellite image on this screen, you are not looking at a photograph.
You are looking at a scientific measurement rendered as colour to make it interpretable.
The colour assigned to each pixel is a choice — it represents the intensity of a
particular wavelength measurement, mapped onto a colour channel so human eyes can
perceive the difference.
        """)

        st.markdown("### What is a band?")
        st.markdown("""
Each wavelength range that a sensor measures is called a **band**. Think of it like
a filter on a camera — except instead of blocking colour, each band captures energy
at a precise part of the electromagnetic spectrum.

Some bands measure visible light (blue at 490 nm, green at 560 nm, red at 665 nm).
Others measure near-infrared (842 nm), which healthy vegetation reflects strongly but
human eyes cannot see at all. Others measure shortwave infrared (1610 nm, 2190 nm),
which reveals moisture content in soil, snow cover, and burn scars. Landsat even
includes a thermal infrared band that measures surface temperature directly.

Every band has a specific physical meaning. Choosing which band to display in which
colour channel is how you ask the satellite a specific question.
        """)

        st.markdown("### True color vs. false color")
        st.markdown("""
**True color** is when you assign the red band to the Red channel, the green band to
the Green channel, and the blue band to the Blue channel. The result mimics what a
camera would capture — it looks natural to the human eye. You see the ground roughly
as you would from an aircraft.

**False color** is when you assign any other combination of bands to the RGB channels.
The result does not look natural, but it reveals things a camera cannot show.

The most common example: assign the near-infrared band (NIR) to the Red channel.
Healthy vegetation reflects NIR strongly, so it appears bright red. Bare soil reflects
NIR weakly, so it appears dark. Open water absorbs NIR almost completely, so it appears
black. In a true color image, all three of these might look similar shades of brown and
green. In false color NIR, they are immediately and unambiguously distinguishable.

This is the core power of multispectral imaging. You are not making the image look
different for aesthetic reasons. You are selecting which physical measurement to
visualise in order to answer a specific question about the ground.
        """)

        st.markdown("### The tilt you may notice")
        st.markdown("""
Satellite images often appear tilted or diamond-shaped rather than as neat rectangles.
This is normal and reflects real satellite geometry.

Landsat orbits at approximately 705 km altitude on a sun-synchronous orbit inclined at
98 degrees relative to the equator. The satellite travels diagonally across the Earth
while the planet rotates underneath it. Each pass captures a diagonal strip of ground.
Sentinel-2 follows the same orbital geometry.

The black triangles in the corners of tilted images are simply areas the sensor did not
cover on that particular orbital pass. They contain no data. Professional GIS software
hides this by mosaicking many scenes together and reprojecting them into a flat grid —
but that requires additional processing. What you see here is the raw scene exactly as
the satellite captured it.
        """)

        # -------------------------------------------------------------------
        # Section 2: Dynamic explanation of this specific combination
        # -------------------------------------------------------------------
        st.divider()

        if is_contact_sheet:
            st.markdown("### How to read the comparison grid")
            st.markdown(f"""
The grid below shows **{sat_key}** imagery of **{location_label}** rendered in every
available band combination simultaneously. Each thumbnail uses a different assignment
of bands to the Red, Green, and Blue display channels.

Look at the same feature — a forest patch, a city, a river — across all thumbnails.
Notice how dramatically its colour changes between combinations. That is not an
artefact. Each colour change represents a real difference in what that combination
is measuring about the surface.

The last two thumbnails (NDVI and NDWI) are different from the others. They are not
RGB composites — they are mathematical indices computed from band ratios and displayed
using a fixed colour scale (green-to-red for vegetation health, blue for water
extent). These are the most analytically useful outputs: they give you a direct numeric
signal rather than an interpretive colour image.
            """)

            st.markdown("**What each preset combination is measuring:**")
            sat_info_local = satellite_catalog.SATELLITES[sat_key]
            for preset_name_item, preset_vals in sat_info_local["presets"].items():
                r = preset_vals["r"]
                g = preset_vals["g"]
                b = preset_vals["b"]
                note = preset_vals["note"]
                r_name = sat_info_local["bands"].get(r, {}).get("name", r)
                g_name = sat_info_local["bands"].get(g, {}).get("name", g)
                b_name = sat_info_local["bands"].get(b, {}).get("name", b)
                st.markdown(
                    f"**{preset_name_item}** — R: {r} ({r_name}), G: {g} ({g_name}), "
                    f"B: {b} ({b_name}). *{note}*"
                )

        else:
            sat_info_local = satellite_catalog.SATELLITES[sat_key]
            r_info = sat_info_local["bands"].get(r_band, {})
            g_info = sat_info_local["bands"].get(g_band, {})
            b_info = sat_info_local["bands"].get(b_band, {})

            r_name = r_info.get("name", r_band)
            g_name = g_info.get("name", g_band)
            b_name = b_info.get("name", b_band)
            r_wl   = r_info.get("wavelength", "")
            g_wl   = g_info.get("wavelength", "")
            b_wl   = b_info.get("wavelength", "")
            r_desc = r_info.get("description", "")
            g_desc = g_info.get("description", "")
            b_desc = b_info.get("description", "")

            is_true_color = (
                r_band in ("B04", "red") and
                g_band in ("B03", "green") and
                b_band in ("B02", "blue")
            )

            if preset_name and preset_name != "Manual":
                st.markdown(f"### This combination: {preset_name}")
            else:
                st.markdown("### This combination")

            if is_true_color:
                st.markdown(f"""
You are viewing a **true color** composite. This is the closest satellite imagery
gets to a photograph. The Red channel carries {r_band} ({r_name}, {r_wl}), the Green
channel carries {g_band} ({g_name}, {g_wl}), and the Blue channel carries {b_band}
({b_name}, {b_wl}) — matching the three colour ranges the human eye uses.

The result is a natural-looking image where vegetation appears green, bare soil appears
brown or tan, water appears dark blue, and built-up areas appear grey. Snow and clouds
appear white because they reflect strongly across all three visible bands.

Even though it looks like a photograph, it is still a processed scientific product.
The raw sensor data has been atmospherically corrected (to remove the effect of the
atmosphere on reflectance values), radiometrically calibrated (to convert raw digital
numbers to physically meaningful reflectance percentages), and display-stretched
(to map the actual reflectance range onto the 0-255 display range).
                """)
            else:
                st.markdown(f"""
You are viewing a **false color** composite. The colour in this image does not
correspond to what the human eye would see — each channel carries a measurement
from outside the visible spectrum, or a different visible band than usual.

**Red channel — {r_band} ({r_name}, {r_wl}):**
{r_desc}
In this image, anything that appears **red or bright** is a surface that reflects
strongly in the {r_wl} wavelength range.

**Green channel — {g_band} ({g_name}, {g_wl}):**
{g_desc}
In this image, anything that contributes to a **green tone** is reflecting strongly
in the {g_wl} range.

**Blue channel — {b_band} ({b_name}, {b_wl}):**
{b_desc}
In this image, anything that appears **blue** is reflecting strongly in the {b_wl}
range.

When all three channels combine, the colour of any given pixel tells you the relative
reflectance of that surface across all three wavelength ranges simultaneously. A pixel
that appears white is reflecting strongly in all three. A pixel that appears black is
absorbing energy in all three. A pixel that appears bright red is reflecting strongly
only in the {r_wl} range.
                """)

            if location_label:
                st.markdown(f"### What to look for over {location_label}")
                if "zanzibar" in location_label.lower() or "tanzania" in location_label.lower():
                    st.markdown("""
- **The island outline:** The boundary between land and ocean is sharp. Ocean absorbs
  near-infrared strongly — it will appear very dark in any combination that includes NIR.
- **Coral reefs:** In true color, shallow reef areas appear turquoise or light blue due
  to the sandy substrate and clear shallow water. In NIR combinations they disappear
  entirely as water absorbs all NIR.
- **Vegetation:** The island has a mix of dense coastal forest, smallholder farms, and
  the urban area of Stone Town. In false color NIR, the forest appears brighter red than
  the farmland, which appears paler. Urban Stone Town appears grey-dark regardless of
  combination.
- **Cloud cover:** Clouds appear white in every combination because they reflect all
  wavelengths strongly. Cloud shadows on the ground appear dark.
                    """)
                elif "moscow" in location_label.lower() or "russia" in location_label.lower():
                    st.markdown("""
- **The urban core:** Moscow is one of the largest cities in Europe. Dense urban areas
  appear grey in true color and purple-magenta in urban false color combinations. The
  contrast with surrounding forest is stark.
- **The Moscow River:** The river running through the city appears dark in any NIR
  combination. Water absorbs NIR almost completely.
- **Seasonal vegetation:** If the scene was captured in summer, the surrounding mixed
  forest appears bright green (true color) or bright red (NIR false color). Winter scenes
  will show snow (white across all combinations) and bare deciduous trees.
                    """)
                elif "alpharetta" in location_label.lower() or "georgia" in location_label.lower():
                    st.markdown("""
- **Suburban development pattern:** Alpharetta is a rapidly growing suburb north of
  Atlanta. The development pattern — road networks, residential subdivisions, commercial
  zones — is clearly visible in true color as a grey-tan grid overlaid on forest.
- **Urban heat:** In Landsat thermal band, developed areas appear warmer (brighter) than
  the surrounding forest. This is the urban heat island effect.
- **Forest fragments:** The Atlanta metro area retains significant forest between
  developments. In NIR false color, these fragments appear bright red against the grey
  built-up areas.
                    """)
                elif "serengeti" in location_label.lower() or "central tanzania" in location_label.lower():
                    st.markdown("""
- **Savanna texture:** The Serengeti ecosystem appears as a mosaic of grass, shrub, and
  occasional forest. In true color it looks uniformly tan-brown. In NIR false color the
  variation in vegetation density becomes clearly visible.
- **Seasonal water:** Ephemeral rivers and seasonal lakes appear as dark lines and patches.
  In NIR and SWIR combinations, moisture in the soil around these features makes them
  stand out.
- **Burned areas:** Controlled burns are common in East African savanna management.
  Recent burn scars appear very dark in NIR and bright in SWIR combinations.
                    """)
                else:
                    st.markdown("""
- Look for the **boundary between land and water** — water absorbs NIR almost completely
  and appears very dark in any combination that includes a NIR band.
- Look for **vegetation density gradients** — dense healthy vegetation reflects NIR more
  strongly than stressed or sparse vegetation. In NIR false color, this appears as varying
  intensities of red.
- Look for **built-up areas** — urban surfaces reflect SWIR more strongly than vegetation.
  In SWIR combinations, cities and roads appear brighter than surrounding land.
                    """)


# ===========================================================================
# MODULE ROUTING
# The Time Series Explorer renders first and calls st.stop() when active.
# That prevents the EO Explorer code below from running.
# The EO Explorer code needs no indentation changes — it just runs when
# the Time Series module is not selected.
# ===========================================================================

if selected_module == "📈 Time Series Explorer":

    # -----------------------------------------------------------------------
    # MODULE 2 — Time Series Explorer
    # All logic lives in gee_timeseries.py. This block handles layout only.
    # -----------------------------------------------------------------------

    gee_available = st.session_state.gee_available

    # --- Header ---
    st.subheader("📈 Time Series Explorer")
    st.caption(
        "Select a dataset, location, and year range, then click Run Analysis."
    )

    with st.expander("ℹ️ What does this module do?", expanded=False):
        st.markdown("""
**This module answers the question: how has this location changed over time?**

It pulls 10+ years of satellite measurements for any location on Earth and shows
you how a chosen index — vegetation greenness, land surface temperature, or others —
has shifted across the full period.

**What you will see after running an analysis:**

- **Summary metrics** — mean value, long-term trend per year, and seasonal amplitude at a glance
- **Time series chart** — every data point plotted across the full date range, with a smoothed trend line and a linear trend overlay
- **Seasonal cycle chart** — the average pattern by month, showing when values peak and trough across a typical year
- **Annual comparison chart** — one bar per year, colour-coded to highlight how early years compare to recent years
- **AI interpretation** — a plain-language explanation of what the data shows, why the pattern exists, and what it means in practice

**How to use it:**

1. Choose a dataset from the dropdown
2. Type any city, country, or region in the location box
3. Set the year range
4. Click Run Analysis
        """)

    # --- Row 1: Dataset + Location (mirrors Spectral Explorer layout) ---
    col_ds, col_loc = st.columns([1, 3])

    with col_ds:
        ts_dataset = st.selectbox(
            "Dataset",
            list(gee_timeseries.DATASETS.keys()),
            key="ts_dataset",
        )

    with col_loc:
        ts_custom_place = st.text_input(
            "Location — type any city, country, or region",
            placeholder="e.g. Sahel, West Africa   |   Tanzania   |   Patagonia   |   Fergana Valley",
            key="ts_custom_place",
        )

    # --- Row 2: Year range + Run button ---
    yr_col1, yr_col2, yr_col3 = st.columns([1, 1, 1], vertical_alignment="bottom")

    with yr_col1:
        ts_start_year = st.number_input(
            "From year", min_value=2000, max_value=2023,
            value=2014, step=1, key="ts_start_year",
        )

    with yr_col2:
        ts_end_year = st.number_input(
            "To year", min_value=2001, max_value=2024,
            value=2024, step=1, key="ts_end_year",
        )

    with yr_col3:
        run_btn = st.button(
            "▶ Run Analysis", type="primary",
            use_container_width=True, key="ts_run",
        )

    # Dataset educational info — collapsible, sits just below the controls
    with st.expander("ℹ️ About this dataset", expanded=False):
        ds_info = gee_timeseries.DATASETS[ts_dataset]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Sensor:** {ds_info['sensor']}")
            st.markdown(f"**Resolution:** {ds_info['resolution']}")
            st.markdown(f"**Revisit:** {ds_info['revisit']}")
            st.markdown(f"**Available from:** {ds_info['available_from']}")
        with col_b:
            st.markdown(f"**Measures:** {ds_info['measures']}")
            st.markdown(f"**Best for:** {ds_info['best_for']}")
            st.markdown(f"**Limitation:** {ds_info['limitation']}")

    st.divider()

    # --- Resolve region from typed location ---
    ts_bbox        = None
    ts_region_name = ""

    if ts_custom_place.strip():
        cached_place = st.session_state.get("ts_geocoded_place", "")
        cached_bbox  = st.session_state.get("ts_geocoded_bbox",  None)

        if ts_custom_place.strip() != cached_place:
            with st.spinner(f"Looking up '{ts_custom_place}'..."):
                result_bbox = geocoder.geocode_place(ts_custom_place)
            if result_bbox:
                st.session_state.ts_geocoded_place = ts_custom_place.strip()
                st.session_state.ts_geocoded_bbox  = result_bbox
                cached_bbox = result_bbox
            else:
                st.session_state.ts_geocoded_place = ""
                st.session_state.ts_geocoded_bbox  = None
                cached_bbox = None
                st.error(f"Could not geocode '{ts_custom_place}'. Using preset region.")

        if cached_bbox:
            ts_bbox        = cached_bbox
            ts_region_name = ts_custom_place.strip()
            st.caption(f"📍 {ts_region_name}")

    # GEE status — compact caption so it doesn't dominate the page
    if gee_available:
        st.caption("🟢 GEE connected — live data active.")
    else:
        st.caption(
            "🔵 No GEE credentials — showing sample data modeled from MODIS climatology. "
            "Add GEE_SERVICE_ACCOUNT_JSON to Streamlit secrets to enable live queries."
        )

    # --- Session state initialisation for Time Series results ---
    for _k, _v in [
        ("ts_df",     None), ("ts_stats",   None),
        ("ts_result_dataset", None), ("ts_result_region", None),
        ("ts_result_start",   None), ("ts_result_end",    None),
        ("ts_is_sample", True),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # --- Run Analysis ---
    # Two-step pattern: button click clears results and queues a run, then
    # st.rerun() forces a full re-render (blank page). The pending run block
    # below picks up on the next render and executes the analysis.
    if run_btn:
        if not ts_bbox:
            st.warning("Enter a location first.")
        elif ts_start_year >= ts_end_year:
            st.error("Start year must be before end year.")
        else:
            # Step 1: clear old results and queue the analysis parameters
            st.session_state.ts_df        = None
            st.session_state.ts_ai_result = None
            st.session_state.ts_pending_run = {
                "bbox":    ts_bbox,
                "dataset": ts_dataset,
                "start":   ts_start_year,
                "end":     ts_end_year,
                "region":  ts_region_name,
            }
            st.rerun()

    # Step 2: execute the queued analysis (runs on the render AFTER the button click)
    if st.session_state.get("ts_pending_run"):
        p = st.session_state.ts_pending_run
        st.session_state.ts_pending_run = None  # consume the queue entry
        with st.spinner(f"Fetching {p['dataset']} data for {p['region']}..."):
            try:
                if gee_available:
                    df       = gee_timeseries.extract_time_series_gee(
                        p["bbox"], p["dataset"], p["start"], p["end"]
                    )
                    is_sample = False
                else:
                    df       = gee_timeseries.generate_sample_data(
                        p["region"], p["dataset"], p["start"], p["end"]
                    )
                    is_sample = True

                stats = gee_timeseries.compute_statistics(df, p["dataset"])
                st.session_state.ts_df             = df
                st.session_state.ts_stats          = stats
                st.session_state.ts_result_dataset = p["dataset"]
                st.session_state.ts_result_region  = p["region"]
                st.session_state.ts_result_start   = p["start"]
                st.session_state.ts_result_end     = p["end"]
                st.session_state.ts_is_sample      = is_sample

                # Reset AI prompt to match the new region and dataset
                st.session_state.ts_ai_prompt = (
                    f"Interpret this {p['dataset']} time series for {p['region']} "
                    f"from {p['start']} to {p['end']}. "
                    f"Cover: what the data shows, why the pattern exists, "
                    f"one practical application, and one limitation."
                )
                st.success(
                    f"Analysis complete — {stats['count']} data points, "
                    f"{p['start']} to {p['end']}."
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # --- Display results (from session state so they survive widget interactions) ---
    if st.session_state.ts_df is not None:
        df      = st.session_state.ts_df
        stats   = st.session_state.ts_stats
        r_ds    = st.session_state.ts_result_dataset
        r_reg   = st.session_state.ts_result_region
        r_start = st.session_state.ts_result_start
        r_end   = st.session_state.ts_result_end

        if st.session_state.ts_is_sample:
            st.caption("Showing sample data. Add GEE credentials to switch to live data.")

        # Reusable thick section divider — heavier than st.divider()
        def section_break():
            st.markdown(
                '<hr style="border: none; border-top: 3px solid #d0d0d0; margin: 28px 0 20px 0;">',
                unsafe_allow_html=True,
            )

        # --- SECTION 1: Summary metrics ---
        unit = gee_timeseries.DATASETS[r_ds]["unit"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data points", stats["count"])
        c2.metric(f"Mean {unit}", f"{stats['mean']:.3f}")
        c3.metric("Trend / year",
                  f"{stats['slope_per_year']:+.4f} {unit}",
                  delta_color="normal")
        c4.metric("Seasonal amplitude", f"{stats['amplitude']:.3f} {unit}")

        section_break()

        # --- SECTION 2: Time series chart ---
        st.subheader("📈 Time Series")
        fig_ts = gee_timeseries.build_timeseries_chart(
            df, stats, r_ds, r_reg, r_start, r_end
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        section_break()

        # --- SECTION 3: Seasonal cycle chart + statistics panel ---
        st.subheader("🗓️ Seasonal Cycle")
        col_seasonal, col_stats = st.columns([2, 1])

        with col_seasonal:
            fig_seasonal = gee_timeseries.build_seasonal_chart(stats, r_ds)
            st.plotly_chart(fig_seasonal, use_container_width=True)

        with col_stats:
            st.markdown("**Seasonal statistics**")
            st.markdown(f"- Peak month: **{stats['peak_month']}** ({stats['peak_value']:.3f} {unit})")
            st.markdown(f"- Trough month: **{stats['trough_month']}** ({stats['trough_value']:.3f} {unit})")
            st.markdown(f"- Amplitude: **{stats['amplitude']:.3f} {unit}**")
            st.divider()
            st.markdown("**Long-term trend**")
            direction_emoji = "📈" if stats["slope_per_year"] > 0 else "📉"
            st.markdown(
                f"{direction_emoji} {stats['trend_direction'].capitalize()} "
                f"at {stats['slope_per_year']:+.4f} {unit}/year"
            )

        section_break()

        # --- SECTION 4: Annual means chart ---
        st.subheader("📊 Annual Comparison")
        early_boundary  = r_start + max(1, (r_end - r_start) // 4)
        recent_boundary = r_end   - max(1, (r_end - r_start) // 4)

        # Compute change statistics for the summary panel
        early_mean  = df[df["date"].dt.year <= early_boundary]["value"].mean()
        recent_mean = df[df["date"].dt.year >= recent_boundary]["value"].mean()
        diff_val    = recent_mean - early_mean
        pct_change  = (diff_val / early_mean * 100) if early_mean != 0 else 0

        col_chart, col_summary = st.columns([3, 1])

        with col_chart:
            fig_annual = gee_timeseries.build_annual_means_chart(
                df, r_ds, r_start, r_end, early_boundary, recent_boundary
            )
            st.plotly_chart(fig_annual, use_container_width=True)

        with col_summary:
            st.markdown("**Period comparison**")
            st.metric(
                label=f"Early mean ({r_start}–{early_boundary})",
                value=f"{early_mean:.3f} {unit}",
            )
            st.metric(
                label=f"Recent mean ({recent_boundary}–{r_end})",
                value=f"{recent_mean:.3f} {unit}",
                delta=f"{diff_val:+.3f} {unit}",
            )
            direction = "increase" if diff_val > 0 else "decrease"
            emoji     = "🟢" if diff_val > 0 else "🔴"
            st.markdown(
                f"{emoji} **{abs(pct_change):.1f}% {direction}** "
                f"over the full period."
            )

        section_break()

        # --- SECTION 5: AI Interpretation ---
        st.subheader("🤖 AI Interpretation")
        default_prompt = (
            f"Interpret this {r_ds} time series for {r_reg} "
            f"from {r_start} to {r_end}. "
            f"Cover: what the data shows, why the pattern exists, "
            f"one practical application, and one limitation."
        )
        ts_user_prompt = st.text_area(
            "Edit the prompt before submitting (optional):",
            value=default_prompt,
            height=100,
            key="ts_ai_prompt",
        )
        if st.button("Get AI Interpretation", type="primary", key="ts_ai_btn"):
            with st.spinner("Thinking..."):
                interpretation = gee_timeseries.get_ai_interpretation(
                    stats, r_ds, r_reg, r_start, r_end,
                    custom_prompt=ts_user_prompt,
                    api_key=config.GROQ_API_KEY or None,
                )
                st.session_state.ts_ai_result = interpretation

        if "ts_ai_result" in st.session_state and st.session_state.ts_ai_result:
            st.markdown(st.session_state.ts_ai_result)

    else:
        # No analysis run yet — show a prompt to get started
        st.markdown("---")
        st.markdown(
            "**Type a location above, then click Run Analysis.**\n\n"
            "The module will produce a time series chart, a seasonal cycle chart, "
            "two comparison maps, and an AI interpretation of what the data shows."
        )

    # Stop here — do not render the EO Explorer below
    st.stop()

# ---------------------------------------------------------------------------
# MODULE 2 — Spectral Explorer (reached only when Spectral Explorer is selected)
# ---------------------------------------------------------------------------

if selected_module == "🔬 Spectral Explorer":

    st.subheader("🔬 Spectral Explorer")
    st.caption(
        "Choose any location, satellite, and band combination. "
        "Fetch real imagery and compare how the same area looks in different spectral views."
    )

    with st.expander("ℹ️ What does this module do?", expanded=False):
        st.markdown("""
**This module answers the question: what can satellites see that cameras cannot?**

It fetches real satellite imagery from NASA and ESA archives for any location on Earth
and lets you render the same scene through different combinations of spectral bands.
Each combination reveals different physical properties of the ground surface — vegetation
health, water extent, urban heat, burn scars, soil moisture, and more.

**What you will see after fetching a scene:**

- **Scene timeline** — a chart of all available satellite passes in your date range, scored by how much of your area they cover
- **Rendered image** — the actual satellite data displayed using your chosen band combination, with a download button
- **Scene details** — date, cloud cover, satellite, and an explanation of what each channel is measuring
- **Compare all views** — every available band combination rendered side by side so you can see how the same place looks across the full electromagnetic spectrum
- **AI explanation** — a plain-language interpretation of what the chosen combination reveals and why

**How to use it:**

1. Choose a satellite from the dropdown
2. Type any location in the search box
3. Set a date range and maximum cloud cover
4. Click Search for scenes
5. Choose a band combination and click Render
        """)

    # -----------------------------------------------------------------------
    # Initialise all session state keys up front so they always exist.
    # This separates state management from display logic.
    # -----------------------------------------------------------------------
    for key, default in [
        ("se_items",           []),
        ("se_best_item",       None),
        ("se_coverage",        []),
        ("se_timeline_items",  []),
        ("se_prev_satellite",  None),
        ("se_prev_preset",     None),
        ("se_rendered_arr",    None),
        ("se_rendered_info",   None),
        ("se_contact_results", None),
        ("se_contact_info",    None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # -----------------------------------------------------------------------
    # Row 1: Satellite + Location
    # -----------------------------------------------------------------------
    col_sat, col_place = st.columns([1, 3])

    with col_sat:
        sat_key  = st.selectbox("Satellite", list(satellite_catalog.SATELLITES.keys()), key="se_satellite")
        sat_info = satellite_catalog.SATELLITES[sat_key]

    with col_place:
        place_input = st.text_input(
            "Location — type any city, country, or region",
            placeholder="e.g. Zanzibar   |   Moscow, Russia   |   Serengeti, Tanzania   |   Alpharetta, Georgia",
            key="se_place_input",
        )

    # Resolve bounding box from the typed place name.
    # Geocoding runs on every render while the text box has content.
    # The spinner only shows on the first lookup — subsequent renders
    # use the cached result already stored in session state.
    bbox_se        = None
    location_label = ""

    if place_input.strip():
        # Only call the geocoder when the place text has changed
        cached_place = st.session_state.get("se_geocoded_place", "")
        cached_bbox  = st.session_state.get("se_geocoded_bbox",  None)

        if place_input.strip() != cached_place:
            with st.spinner(f"Looking up '{place_input}'..."):
                result_bbox = geocoder.geocode_place(place_input)
            if result_bbox:
                st.session_state.se_geocoded_place = place_input.strip()
                st.session_state.se_geocoded_bbox  = result_bbox
                cached_bbox = result_bbox
            else:
                st.session_state.se_geocoded_place = ""
                st.session_state.se_geocoded_bbox  = None
                cached_bbox = None
                st.error(f"Could not find '{place_input}'. Check the spelling or try a broader name (e.g. country instead of city).")

        if cached_bbox:
            bbox_se        = cached_bbox
            location_label = place_input.strip()
            area = geocoder.bbox_area_km2(bbox_se)
            st.caption(f"📍 {location_label} — bbox {[round(x, 3) for x in bbox_se]} — {area:,.0f} km²")
    else:
        # Text box is empty — clear any cached geocode result
        st.session_state.se_geocoded_place = ""
        st.session_state.se_geocoded_bbox  = None

    # Clear all search results when the user changes the location.
    # Compare the typed text against what was last searched — not against
    # previous widget states, which was the source of the clearing bugs.
    last_searched = st.session_state.get("se_last_searched_location", "")
    if last_searched and location_label and location_label.lower() != last_searched.lower():
        st.session_state.se_items            = []
        st.session_state.se_timeline_items   = []
        st.session_state.se_coverage         = []
        st.session_state.se_best_item        = None
        st.session_state.se_rendered_arr     = None
        st.session_state.se_contact_results  = None

    # -----------------------------------------------------------------------
    # Row 2: Date range + Cloud cover + Search button
    # The Search button lives here because it acts on these parameters.
    # Once the user has set satellite, location, dates and cloud cover,
    # the next natural action is to search — not scroll past band controls.
    # -----------------------------------------------------------------------
    # vertical_alignment="bottom" aligns the button with the bottom edge of
    # the date inputs and slider, matching their visual baseline precisely.
    col_d1, col_d2, col_cloud, col_search = st.columns([1, 1, 1, 1], vertical_alignment="bottom")
    with col_d1:
        date_start = st.date_input("From date", value=date.today() - timedelta(days=180), key="se_date_start")
    with col_d2:
        date_end   = st.date_input("To date",   value=date.today(), key="se_date_end")
    with col_cloud:
        if sat_info.get("sar"):
            st.info("SAR is cloud-independent. No cloud filter applied.")
            max_cloud = 100
        else:
            max_cloud = st.slider("Max cloud cover %", 0, 50, 20, key="se_cloud")
    with col_search:
        fetch_btn = st.button("🔍 Search for scenes", type="primary", key="se_fetch", use_container_width=True)

    date_range_str = f"{date_start.isoformat()}/{date_end.isoformat()}"

    # -----------------------------------------------------------------------
    # SEARCH — runs when button clicked, stores results in session state
    # -----------------------------------------------------------------------
    if fetch_btn:
        if not bbox_se:
            st.warning("Select a location first.")
        else:
            with st.spinner("Searching Planetary Computer..."):
                try:
                    cat   = spectral_explorer.get_catalog()
                    items = spectral_explorer.search_scenes(
                        cat, sat_info["collection"], bbox_se,
                        date_range_str, max_cloud, sat_info["cloud_field"],
                    )
                    st.session_state.se_items          = items
                    st.session_state.se_timeline_items = items
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    items = []

            if items:
                with st.spinner("Scoring scenes for coverage..."):
                    # Use True Color bands for scoring — they are always valid
                    # for the current satellite and don't depend on what the
                    # user has selected in the band dropdowns below.
                    _tc      = sat_info["presets"].get("True Color", {})
                    _band_ids = list(sat_info["bands"].keys())
                    _score_r  = _tc.get("r", _band_ids[0])
                    _score_g  = _tc.get("g", _band_ids[0])
                    _score_b  = _tc.get("b", _band_ids[0])
                    best_item, best_pct, scores = spectral_explorer.find_best_scene(
                        items, _score_r, _score_g, _score_b, sat_key, max_to_check=6
                    )
                st.session_state.se_best_item = best_item
                st.session_state.se_coverage  = scores
                st.session_state.se_rendered_arr    = None
                st.session_state.se_contact_results = None
                # Stamp the location so future renders know what was searched
                st.session_state.se_last_searched_location = location_label
                st.success(
                    f"Found {len(items)} scenes. "
                    f"Best: {best_item.datetime.strftime('%Y-%m-%d')} — {best_pct:.0f}% coverage. "
                    f"Now choose a band combination below and click Render."
                )
            else:
                st.warning("No scenes found. Try a wider date range or higher cloud cover limit.")

    # -----------------------------------------------------------------------
    # DISPLAY SEARCH RESULTS — shown here, directly below the search section,
    # so the user can see what is available before choosing how to visualise it
    # -----------------------------------------------------------------------
    if st.session_state.se_timeline_items:
        fig = spectral_explorer.build_timeline(
            st.session_state.se_timeline_items,
            sat_info["cloud_field"],
            st.session_state.se_coverage,
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.se_coverage:
            st.markdown("**Top scenes scored by coverage:**")
            for date_str, cloud, pct, _ in st.session_state.se_coverage:
                bar       = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                cloud_str = f"Cloud {cloud:.0f}%" if sat_info["cloud_field"] else "SAR"
                st.caption(f"`{date_str}`  {cloud_str}  Coverage: {bar} {pct:.0f}%")

    # -----------------------------------------------------------------------
    # Row 3: Band assignment
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("**Band combination — choose a preset or assign bands manually**")

    band_ids      = list(sat_info["bands"].keys())
    band_names    = [f"{k} — {v['name']} ({v['wavelength']})" for k, v in sat_info["bands"].items()]
    band_map      = dict(zip(band_names, band_ids))
    id_to_display = {v: k for k, v in band_map.items()}

    # True color defaults for this satellite
    tc_preset    = sat_info["presets"].get("True Color", {})
    default_r_id = tc_preset.get("r", band_ids[min(2, len(band_ids)-1)])
    default_g_id = tc_preset.get("g", band_ids[min(1, len(band_ids)-1)])
    default_b_id = tc_preset.get("b", band_ids[0])

    # Reset R/G/B when satellite changes
    satellite_changed = (sat_key != st.session_state.se_prev_satellite)
    if satellite_changed:
        st.session_state.se_prev_satellite = sat_key
        st.session_state.se_r = id_to_display.get(default_r_id, band_names[0])
        st.session_state.se_g = id_to_display.get(default_g_id, band_names[0])
        st.session_state.se_b = id_to_display.get(default_b_id, band_names[0])
        st.session_state.se_prev_preset = None
        # Clear stored results — they belong to the old satellite
        st.session_state.se_rendered_arr    = None
        st.session_state.se_contact_results = None

    # Initialise R/G/B to true color on very first load
    if "se_r" not in st.session_state:
        st.session_state.se_r = id_to_display.get(default_r_id, band_names[0])
    if "se_g" not in st.session_state:
        st.session_state.se_g = id_to_display.get(default_g_id, band_names[0])
    if "se_b" not in st.session_state:
        st.session_state.se_b = id_to_display.get(default_b_id, band_names[0])

    # Preset dropdown.
    # "— Custom —" is a read-only indicator that appears automatically when
    # the R/G/B channels no longer match any named preset. The user never
    # needs to pick it — it switches on its own.
    #
    # Streamlit rule: a widget's session state key can only be set BEFORE
    # the widget is instantiated in a given render cycle. So the mismatch
    # check must happen here, before st.selectbox() is called.
    CUSTOM_LABEL = "— Custom —"
    preset_names = [CUSTOM_LABEL] + list(sat_info["presets"].keys())

    # Read the current R/G/B band IDs from session state (set by previous render)
    cur_r = band_map.get(st.session_state.get("se_r", band_names[0]), band_ids[0])
    cur_g = band_map.get(st.session_state.get("se_g", band_names[0]), band_ids[0])
    cur_b = band_map.get(st.session_state.get("se_b", band_names[0]), band_ids[0])

    stored_preset = st.session_state.get("se_preset", "True Color")
    prev_preset   = st.session_state.get("se_prev_preset")

    # Only check for a channel/preset mismatch when the preset is STABLE —
    # i.e. the user did not just change it in this render cycle.
    # If the preset just changed, the R/G/B channels still hold the OLD values
    # (they haven't been updated yet). Running the check at that moment would
    # wrongly detect a mismatch and flip straight to Custom.
    preset_just_changed = (stored_preset != prev_preset)

    if not preset_just_changed and stored_preset != CUSTOM_LABEL and stored_preset in sat_info["presets"]:
        pv = sat_info["presets"][stored_preset]
        if cur_r != pv["r"] or cur_g != pv["g"] or cur_b != pv["b"]:
            st.session_state.se_preset      = CUSTOM_LABEL
            st.session_state.se_prev_preset = CUSTOM_LABEL

    col_preset, col_r, col_g, col_b = st.columns([2, 1, 1, 1])

    with col_preset:
        # Default index points to True Color, not Custom (index 0).
        # The index is only used on first load when se_preset is not yet
        # in session state. After that, session state controls the value.
        tc_idx = preset_names.index("True Color") if "True Color" in preset_names else 1
        chosen_preset = st.selectbox(
            "Named preset (auto-fills R/G/B below)",
            preset_names,
            index=tc_idx,
            key="se_preset",
        )

    # When a real preset is chosen, push its bands into session state before
    # the R/G/B dropdowns render so they display the updated values immediately.
    preset_switched = (chosen_preset != st.session_state.se_prev_preset)
    if preset_switched and chosen_preset != CUSTOM_LABEL:
        st.session_state.se_prev_preset = chosen_preset
        preset_vals = sat_info["presets"][chosen_preset]
        if preset_vals["r"] in id_to_display:
            st.session_state.se_r = id_to_display[preset_vals["r"]]
        if preset_vals["g"] in id_to_display:
            st.session_state.se_g = id_to_display[preset_vals["g"]]
        if preset_vals["b"] in id_to_display:
            st.session_state.se_b = id_to_display[preset_vals["b"]]
    elif preset_switched:
        st.session_state.se_prev_preset = chosen_preset

    # Caption: description when a preset is active, instruction when Custom
    if chosen_preset != CUSTOM_LABEL:
        st.caption(f"ℹ️ {sat_info['presets'][chosen_preset]['note']}")
    else:
        st.caption("ℹ️ Channels set manually. Pick a named preset above to reset to a standard combination.")

    with col_r:
        r_display = st.selectbox("Red channel",   band_names, key="se_r")
        r_band    = band_map[r_display]
    with col_g:
        g_display = st.selectbox("Green channel", band_names, key="se_g")
        g_band    = band_map[g_display]
    with col_b:
        b_display = st.selectbox("Blue channel",  band_names, key="se_b")
        b_band    = band_map[b_display]

    st.caption(
        f"🔴 **{r_band}:** {sat_info['bands'][r_band]['description']}  |  "
        f"🟢 **{g_band}:** {sat_info['bands'][g_band]['description']}  |  "
        f"🔵 **{b_band}:** {sat_info['bands'][b_band]['description']}"
    )

    # Render and Compare buttons sit here — after the band controls,
    # because they act on the band selection the user just made.
    st.divider()
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        render_btn  = st.button("🖼️ Render selected combination", key="se_render",  use_container_width=True)
    with col_btn2:
        contact_btn = st.button("📋 Compare All Views",           key="se_contact", use_container_width=True)

    # -----------------------------------------------------------------------
    # RENDER — runs only when button clicked, stores result in session state
    # -----------------------------------------------------------------------
    if render_btn:
        item = st.session_state.se_best_item
        if item is None and st.session_state.se_items:
            item = st.session_state.se_items[0]

        if item is None:
            st.warning("Search for scenes first.")
        else:
            scene_date = item.datetime.strftime("%B %d, %Y")
            cloud      = item.properties.get("eo:cloud_cover", 0)

            with st.spinner(f"Rendering {r_band}/{g_band}/{b_band} for {scene_date}..."):
                arr = spectral_explorer.render_combination(item, r_band, g_band, b_band, sat_key, width=700)

            if arr is not None and arr.max() > 0:
                # Store everything needed to redisplay after any widget interaction
                st.session_state.se_rendered_arr  = arr
                st.session_state.se_rendered_info = {
                    "scene_date":    scene_date,
                    "cloud":         cloud,
                    "r_band":        r_band,
                    "g_band":        g_band,
                    "b_band":        b_band,
                    "sat_key":       sat_key,
                    "location_label":location_label,
                    "preset_name":   chosen_preset,
                    "item_date":     item.datetime.strftime("%Y-%m-%d"),
                }
            else:
                st.warning("No valid data returned. Try a different scene or location.")

    # -----------------------------------------------------------------------
    # DISPLAY RENDERED IMAGE — always shown if a result exists in session state
    # -----------------------------------------------------------------------
    if st.session_state.se_rendered_arr is not None:
        info = st.session_state.se_rendered_info
        arr  = st.session_state.se_rendered_arr

        # Use stored info so it does not change when user fiddles with dropdowns
        r_b   = info["r_band"];   g_b  = info["g_band"];   b_b  = info["b_band"]
        s_key = info["sat_key"];  s_inf = satellite_catalog.SATELLITES[s_key]

        st.divider()
        col_img, col_info = st.columns([3, 2])

        with col_img:
            st.image(arr, caption=f"{r_b} / {g_b} / {b_b} — {info['scene_date']}", use_container_width=True)

            from PIL import Image as PILImage
            import io as _io
            buf = _io.BytesIO()
            PILImage.fromarray(arr).save(buf, format="PNG")
            fname = (
                f"{info['location_label'].replace(' ','_').replace(',','')}_"
                f"{s_key.replace(' ','_').replace('/','_')}_"
                f"{r_b}-{g_b}-{b_b}_{info['item_date']}.png"
            )
            st.download_button("⬇️ Download image", data=buf.getvalue(),
                               file_name=fname, mime="image/png", key="se_dl_single")

        with col_info:
            st.markdown("**Scene details**")
            st.markdown(f"- **Date:** {info['scene_date']}")
            st.markdown(f"- **Cloud cover:** {info['cloud']:.1f}%")
            st.markdown(f"- **Satellite:** {s_key}")
            st.markdown(f"- **Location:** {info['location_label']}")
            st.markdown(f"- **R channel:** {r_b} — {s_inf['bands'][r_b]['name']} ({s_inf['bands'][r_b]['wavelength']})")
            st.markdown(f"- **G channel:** {g_b} — {s_inf['bands'][g_b]['name']} ({s_inf['bands'][g_b]['wavelength']})")
            st.markdown(f"- **B channel:** {b_b} — {s_inf['bands'][b_b]['name']} ({s_inf['bands'][b_b]['wavelength']})")
            st.divider()
            st.markdown("**What this combination reveals**")
            st.markdown(f"🔴 {s_inf['bands'][r_b]['description']}")
            st.markdown(f"🟢 {s_inf['bands'][g_b]['description']}")
            st.markdown(f"🔵 {s_inf['bands'][b_b]['description']}")

            if config.has_any_key():
                st.divider()
                if st.button("🤖 AI: Explain this view", key="se_explain_single"):
                    with st.spinner("Asking AI..."):
                        explanation = spectral_explorer.explain_combination(
                            r_b, g_b, b_b, s_key, info["location_label"]
                        )
                    st.markdown(explanation)

        st.divider()
        render_what_am_i_looking_at(
            s_key, r_b, g_b, b_b,
            preset_name=info["preset_name"],
            location_label=info["location_label"],
            is_contact_sheet=False,
        )

    # -----------------------------------------------------------------------
    # COMPARE ALL VIEWS — runs only when button clicked
    # -----------------------------------------------------------------------
    if contact_btn:
        item = st.session_state.se_best_item
        if item is None and st.session_state.se_items:
            item = st.session_state.se_items[0]

        if item is None:
            st.warning("Search for scenes first.")
        else:
            with st.spinner("Rendering all combinations (30-60 seconds)..."):
                results = spectral_explorer.render_contact_sheet(item, sat_key)

            valid = [r for r in results if r["array"] is not None and r["array"].max() > 0]
            st.session_state.se_contact_results = valid
            st.session_state.se_contact_info    = {
                "scene_date":    item.datetime.strftime("%B %d, %Y"),
                "sat_key":       sat_key,
                "location_label":location_label,
            }

    # -----------------------------------------------------------------------
    # DISPLAY COMPARE ALL VIEWS — always shown if results exist in session state
    # -----------------------------------------------------------------------
    if st.session_state.se_contact_results:
        c_info  = st.session_state.se_contact_info
        valid   = st.session_state.se_contact_results

        st.divider()
        st.markdown(
            f"**All spectral views — {c_info['location_label']} — "
            f"{c_info['scene_date']} — {c_info['sat_key']}**"
        )
        st.caption("Every named band combination + NDVI + NDWI rendered for comparison.")

        cols_per_row = 3
        for i in range(0, len(valid), cols_per_row):
            row_items = valid[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, result in zip(cols, row_items):
                with col:
                    st.image(result["array"], caption=result["label"], use_container_width=True)
                    st.caption(result["note"])

        st.divider()
        render_what_am_i_looking_at(
            c_info["sat_key"], r_band, g_band, b_band,
            preset_name="",
            location_label=c_info["location_label"],
            is_contact_sheet=True,
        )

    # Stop here — do not render the EO Explorer below
    st.stop()

# ---------------------------------------------------------------------------
# MODULE 1 — EO Explorer (reached only when EO Explorer is selected)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("EO Explorer Controls")

    theme_options   = list(data_catalog.THEMES.keys())
    selected_theme  = st.selectbox("Theme", theme_options)

    location_options   = list(map_builder.LOCATIONS.keys())
    selected_location  = st.selectbox("Location", location_options)

    dataset_options   = data_catalog.THEMES[selected_theme]["datasets"]
    selected_dataset  = st.selectbox("Dataset", dataset_options)

    ai_mode_options = [
        "Explain selected theme",
        "Explain selected dataset",
        "Explain business use case",
        "Explain limitations",
        "Suggest next analysis",
    ]
    selected_ai_mode = st.selectbox("AI Mode", ai_mode_options)

    st.divider()
    provider_status = ai_assistant.get_provider_status()
    st.caption(f"AI: **{provider_status}**")

col_map, col_explain = st.columns([3, 2])

with col_map:
    st.subheader("Interactive Map")
    fmap = map_builder.build_map(selected_location)
    st_folium(fmap, use_container_width=True, height=550, returned_objects=[])

with col_explain:
    st.subheader("Dataset and Theme Overview")
    theme_data = data_catalog.THEMES[selected_theme]
    st.markdown(f"**Theme:** {selected_theme}")
    st.markdown(theme_data["description"])
    st.divider()
    dataset_data = data_catalog.DATASETS.get(selected_dataset, {})
    if dataset_data:
        st.markdown(f"**Dataset:** {selected_dataset}")
        st.markdown(f"- **What it measures:** {dataset_data['measures']}")
        st.markdown(f"- **Sensors:** {dataset_data['sensors']}")
        st.markdown(f"- **Resolution:** {dataset_data['resolution']}")
        st.markdown(f"- **Revisit frequency:** {dataset_data['revisit']}")
        st.markdown(f"- **Typical use cases:** {dataset_data['use_cases']}")
        st.markdown(f"- **Key limitations:** {dataset_data['limitations']}")

st.divider()
st.subheader("AI Assistant")
st.caption(f"Mode: **{selected_ai_mode}** | Provider: **{provider_status}**")

user_question = st.text_input(
    "Ask a question:",
    placeholder="e.g. How does Sentinel-2 detect vegetation stress?",
    key="tab1_question",
)

if st.button("Ask", type="primary", key="tab1_ask"):
    if user_question.strip():
        with st.spinner("Thinking..."):
            try:
                response = ai_assistant.ask(
                    question=user_question,
                    theme=selected_theme,
                    dataset=selected_dataset,
                    location=selected_location,
                    mode=selected_ai_mode,
                )
                st.markdown(response)
            except Exception as e:
                st.error(f"AI call failed: {e}")
    else:
        st.warning("Type a question before submitting.")
else:
    # Show a static prompt instead of making a live AI call on every render.
    # Auto-calling AI on every render was causing the app to hang on load.
    st.caption("Type a question above and click Ask to get an AI response.")
