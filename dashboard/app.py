#%%
"""
dashboard/app.py
----------------
Streamlit dashboard for exploring crop-stress predictions.

Run:
    streamlit run dashboard/app.py

Set LAMBDA_URL environment variable to the API Gateway endpoint, or enter it
in the sidebar at runtime.
"""
import json
import os

import requests
import streamlit as st
import pandas as pd
import plotly.express as px

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
    st.header("Lambda endpoint")
    lambda_url = st.text_input(
        "API Gateway URL",
        value=os.getenv("LAMBDA_URL", "https://bzeqptxosby22wro4bt6v5ea3y0gtdyn.lambda-url.eu-west-1.on.aws/"),
        placeholder="https://<id>.lambda-url.eu-west-1.on.aws/",
    )
    run_btn = st.button("Run analysis", type="primary")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if run_btn:
    if not lambda_url:
        st.error("Enter the API Gateway URL in the sidebar.")
        st.stop()

    payload = {
        "lat": lat,
        "lon": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    with st.spinner("Running inference via Lambda…"):
        try:
            resp = requests.post(lambda_url, json=payload, timeout=330)
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"Lambda returned {resp.status_code}: {resp.text}")
        st.stop()

    # Lambda Function URL returns body directly; API Gateway wraps it in {"body": "..."}
    raw_body = resp.json()
    data = json.loads(raw_body["body"]) if isinstance(raw_body.get("body"), str) else raw_body

    # --- Stress result ------------------------------------------------------
    st.subheader("Crop Stress Prediction")
    col_a, col_b = st.columns(2)
    colour = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(data.get("stress_level", ""), "")
    with col_a:
        st.metric("Stress Index", f"{data['stress_index']:.3f}")
    with col_b:
        st.metric("Stress Level", f"{colour} {data['stress_level'].capitalize()}")

    # --- Weather charts -----------------------------------------------------
    st.subheader("Weather Overview")
    wx_df = pd.DataFrame(data["weather"])
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

    wx_df["water_deficit"] = wx_df["et0_fao_evapotranspiration"] - wx_df["precipitation_sum"]
    wx_df["cumulative_deficit"] = wx_df["water_deficit"].cumsum()
    fig = px.area(wx_df, x="time", y="cumulative_deficit",
                  title="Cumulative Water Deficit (ET₀ − Precip, mm)",
                  color_discrete_sequence=["#e05c2e"])
    st.plotly_chart(fig, use_container_width=True)

    # --- Soil properties ----------------------------------------------------
    st.subheader("Soil Properties")
    if data["soil"]:
        soil_df = pd.DataFrame(
            [{"property_depth": k, "value": v} for k, v in data["soil"].items()]
        )
        st.dataframe(soil_df, use_container_width=True)
    else:
        st.info("No soil data available (SoilGrids API unavailable).")

    # --- Sentinel scenes ----------------------------------------------------
    st.subheader(f"Sentinel-2 Scenes ({len(data['sentinel'])} found)")
    if data["sentinel"]:
        scenes_df = pd.DataFrame(data["sentinel"])
        st.dataframe(scenes_df, use_container_width=True)
    else:
        st.info("No Sentinel-2 scenes found for this location and period.")

else:
    st.info("Set your location and date range in the sidebar, then click **Run analysis**.")

# %%

# %%
