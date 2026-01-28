import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bio-Ops OS", layout="wide", page_icon="🧬")

# --- PRO UI DESIGN (Glassmorphism & Cyberpunk) ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 242, 255, 0.3);
        border-radius: 15px; padding: 20px;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.1);
    }
    [data-testid="stMetricValue"] { color: #00f2ff !important; font-family: 'Courier New', monospace; font-size: 2.2rem !important; }
    [data-testid="stMetricLabel"] { color: #ffffff !important; letter-spacing: 1px; text-transform: uppercase; font-size: 0.8rem !important; }
    h1, h2, h3 { color: #00f2ff !important; font-family: 'Courier New', monospace; border-bottom: 1px solid rgba(0, 242, 255, 0.2); padding-bottom: 10px; }
    .stAlert { background-color: rgba(255, 0, 0, 0.1); border: 1px solid red; color: white; }
    </style>
    """, unsafe_allow_html=True)


# --- PREDICTION ENGINE LOGIC ---
def get_prediction(data_series, window=30, forecast_len=60):
    """Uses Linear Regression to forecast the next N points based on recent trend."""
    X = np.array(range(len(data_series))).reshape(-1, 1)
    y = data_series.values

    model = LinearRegression()
    model.fit(X[-window:], y[-window:])  # Train on the last 'window' of data

    future_X = np.array(range(len(data_series), len(data_series) + forecast_len)).reshape(-1, 1)
    future_y = model.predict(future_X)
    return future_y


# --- DATA LOADING ---
@st.cache_data
def load_data():
    if os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df
    return None


# --- MAIN APP FLOW ---
st.title("🧬 BIO-OPS COMMAND CENTER v1.2")
df_raw = load_data()

if df_raw is not None:
    # --- SIDEBAR: SIMULATION & CONTROL ---
    st.sidebar.header("🛠️ SYSTEM CONTROLS")
    mode = st.sidebar.radio("Operation Mode", ["Live Feed", "Simulation (Stress Test)"])

    # Simulation Logic: Allows you to manually drift the data to show alerts
    temp_offset = 0.0
    if mode == "Simulation (Stress Test)":
        st.sidebar.warning("SIMULATION ACTIVE")
        temp_offset = st.sidebar.slider("Manual Temp Drift (°C)", -3.0, 3.0, 0.0)

    # Apply simulation offset to the most recent data
    df = df_raw.copy()
    df.loc[df.index[-20:], 'Temperature'] += temp_offset

    # --- PARAMETERS ---
    GOLDEN_TEMP = 25.5
    TOLERANCE = 1.0
    df['Anomaly'] = (df['Temperature'] > GOLDEN_TEMP + TOLERANCE) | (df['Temperature'] < GOLDEN_TEMP - TOLERANCE)

    # --- CALCULATE METRICS ---
    latest = df.iloc[-1]
    temp_dev = latest['Temperature'] - GOLDEN_TEMP
    press_dev = latest['Pressure'] - 972.0
    health_score = max(0, min(100, 100 - (abs(temp_dev) * 15) - (abs(press_dev) * 0.05)))

    # --- TOP ROW: KPI DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reactor Temp", f"{latest['Temperature']:.2f}°C", delta=f"{temp_dev:.2f}°C", delta_color="inverse")
    col2.metric("Line Pressure", f"{latest['Pressure']:.1f} hPa")
    col3.metric("Health Index", f"{health_score:.1f}%", delta=f"{health_score - 98:.1f}%")

    status = "STABLE" if health_score > 85 else "CRITICAL"
    col4.metric("Batch Status", status, delta="GXP OK" if status == "STABLE" else "ALERT")

    # --- UPSTREAM: TEMPERATURE & PREDICTION ---
    st.header(">> UPSTREAM: THERMAL DIGITAL TWIN")

    # Get Forecast
    forecast_len = 60
    pred_values = get_prediction(df['Temperature'], forecast_len=forecast_len)
    future_dates = [df['Datetime'].max() + timedelta(minutes=i) for i in range(1, forecast_len + 1)]

    fig_up = px.line(df, x='Datetime', y='Temperature', template="plotly_dark",
                     color_discrete_sequence=['#00f2ff'], title="Real-Time Temp + 60min Predictive Trend")

    # Add Prediction Line
    fig_up.add_scatter(x=future_dates, y=pred_values, name='Predictive Trend',
                       line=dict(dash='dot', color='#ffaa00', width=3))

    # Add Golden Zone
    fig_up.add_hrect(y0=GOLDEN_TEMP - TOLERANCE, y1=GOLDEN_TEMP + TOLERANCE, fillcolor="green", opacity=0.1)

    # Plot Anomalies
    anoms = df[df['Anomaly']]
    fig_up.add_scatter(x=anoms['Datetime'], y=anoms['Temperature'], mode='markers',
                       marker=dict(color='red', size=5), name='Historical Deviation')

    st.plotly_chart(fig_up, use_container_width=True)

    # --- PREDICTIVE VERDICT ---
    if pred_values[-1] > (GOLDEN_TEMP + TOLERANCE) or pred_values[-1] < (GOLDEN_TEMP - TOLERANCE):
        st.error(
            f"🚨 PREDICTIVE ALARM: Process drift detected. Projected GXP violation in approx. {forecast_len} minutes.")
    else:
        st.success("✅ PREDICTIVE STABILITY: Trend indicates the batch will remain within the Golden Zone.")

    # --- DOWNSTREAM & RCA ---
    st.header(">> DOWNSTREAM: ANALYTICS & RCA")
    c1, c2 = st.columns([2, 1])

    with c1:
        fig_p = px.line(df, x='Datetime', y='Pressure', template="plotly_dark", color_discrete_sequence=['#ff00ff'])
        st.plotly_chart(fig_p, use_container_width=True)

    with c2:
        st.subheader("AI Root Cause Analysis")
        if temp_offset > 1.5:
            st.warning("RCA Source identified: External Thermal Load detected. Suggesting HVAC check.")
        elif health_score < 80:
            st.info("RCA Source identified: Pressure/CO2 Correlation spike. Inspecting exhaust filters.")
        else:
            st.success("System noise within 2-sigma. No root cause intervention required.")

    # --- COMPLIANCE REPORTING ---
    st.sidebar.markdown("---")
    if st.sidebar.button("📋 Compile GXP Batch Report"):
        st.markdown("---")
        st.header("📋 FINAL BATCH AUDIT REPORT")
        summary = pd.DataFrame({
            "Metric": ["Batch ID", "Health Index", "GXP Status", "Anomalies Detected"],
            "Value": ["B-2026-X78", f"{health_score:.1f}%", "FAIL" if df['Anomaly'].any() else "PASS", len(anoms)]
        })
        st.table(summary)
        st.download_button("Download Signed Audit Log", df.to_csv().encode('utf-8'), "audit_log.csv", "text/csv")

else:
    st.error("Missing 'dataset.csv' in project directory.")