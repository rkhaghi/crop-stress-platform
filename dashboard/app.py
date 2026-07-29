#%%
"""
dashboard/app.py
----------------
Streamlit dashboard for exploring crop-stress predictions.

Run:
    streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from extract import extract_all
from features.weather_features import build_weather_features
from features.build_features import build_feature_vector
from inference.predictor import predict

st.set_page_config(page_title="TerraSignal — Crop Stress Monitor", layout="wide")
st.title("🌾 TerraSignal — Crop Stress Monitor")

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Location & Period")
    lat        = st.number_input("Latitude",   value=51.5,  step=0.01, format="%.4f")
    lon        = st.number_input("Longitude",  value=-1.2,  step=0.01, format="%.4f")
    start_date = st.date_input("Start date",  value=pd.Timestamp("2024-04-01"))
    end_date   = st.date_input("End date",    value=pd.Timestamp("2024-06-30"))
    max_cloud  = st.slider("Max cloud cover (%)", 0, 100, 30)
    st.divider()
    st.header("Model")
    model_path = st.text_input(
        "Model path", value="", placeholder="models/crop_stress_model.json"
    )
    run_btn    = st.button("Run analysis", type="primary")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if run_btn:
    with st.spinner("Fetching data…"):
        raw = extract_all(
            lat=lat, lon=lon,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            max_cloud=max_cloud,
        )

    # --- Weather chart ------------------------------------------------------
    st.subheader("Weather Overview")
    wx_df = pd.DataFrame(raw["weather"])
    wx_df["time"] = pd.to_datetime(wx_df["time"])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(wx_df, x="time", y="precipitation_sum",
                     title="Daily Precipitation (mm)", labels={"precipitation_sum": "mm"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(wx_df, x="time", y=["temperature_2m_mean", "temperature_2m_max"],
                      title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)

    # Water deficit
    wx_df["water_deficit"] = wx_df["et0_fao_evapotranspiration"] - wx_df["precipitation_sum"]
    wx_df["cumulative_deficit"] = wx_df["water_deficit"].cumsum()
    fig = px.area(wx_df, x="time", y="cumulative_deficit",
                  title="Cumulative Water Deficit (ET₀ − Precip, mm)",
                  color_discrete_sequence=["#e05c2e"])
    st.plotly_chart(fig, use_container_width=True)

    # --- Soil properties ----------------------------------------------------
    st.subheader("Soil Properties")
    soil_df = pd.DataFrame(
        [{"property_depth": k, "value": v} for k, v in raw["soil"].items()]
    )
    st.dataframe(soil_df, use_container_width=True)

    # --- Sentinel scenes ----------------------------------------------------
    st.subheader(f"Sentinel-2 Scenes ({len(raw['sentinel'])} found)")
    if raw["sentinel"]:
        scenes_df = pd.DataFrame([
            {"id": s["id"], "date": s["date"], "cloud_cover": s["cloud_cover"]}
            for s in raw["sentinel"]
        ])
        st.dataframe(scenes_df, use_container_width=True)
    else:
        st.info("No Sentinel-2 scenes found for this location and period.")

    # --- Stress prediction --------------------------------------------------
    if model_path:
        st.subheader("Crop Stress Prediction")
        try:
            ref_date      = end_date.isoformat()
            weather_feats = build_weather_features(raw["weather"], ref_date=ref_date)
            soil_feats    = raw["soil"]
            doy           = end_date.timetuple().tm_yday
            feature_vec   = build_feature_vector(
                {}, weather_feats, soil_feats, metadata={"doy": doy}
            )
            result = predict(feature_vec, model_path=model_path)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Stress Index", f"{result['stress_index']:.3f}")
            with col_b:
                colour = {"low": "\U0001f7e2", "moderate": "\U0001f7e1", "high": "\U0001f534"}.get(
                    result["stress_level"], ""
                )
                st.metric("Stress Level", f"{colour} {result['stress_level'].capitalize()}")
            st.caption(
                "Prediction uses weather + soil features only "
                "(satellite bands are not downloaded in dashboard mode)."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.info("Set your location and date range in the sidebar, then click **Run analysis**.")

# %%
