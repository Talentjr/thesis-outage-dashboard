import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="Cloud Outage Forecasting Dashboard", layout="wide")
st.title("Predictive Outage Forecasting & Root Cause Analysis")
st.caption("Incident 20240124 — LEMMA-RCA Cloud Computing dataset (simulated replay)")

@st.cache_data
def load_latency():
    with open("dashboard_data.pkl", "rb") as f:
        return pickle.load(f)

latency = load_latency()

st.sidebar.header("Adjust Detection Sensitivity")
normal_pct = st.sidebar.slider("Warning threshold (percentile)", 80, 99, 95)
critical_pct = st.sidebar.slider("Critical threshold (percentile)", normal_pct + 1, 100, 99)

normal_threshold = latency.quantile(normal_pct / 100)
critical_threshold = latency.quantile(critical_pct / 100)

col1, col2, col3 = st.columns(3)
col1.metric("Peak Latency", f"{latency.max():.1f} ms")
col2.metric(f"Warning Threshold ({normal_pct}th pct)", f"{normal_threshold:.1f} ms")
col3.metric(f"Critical Threshold ({critical_pct}th pct)", f"{critical_threshold:.1f} ms")

st.subheader("Latency Timeline")

sample = latency.iloc[::50]
readable_times = [datetime.utcfromtimestamp(t) for t in sample.index]
view_values = sample.values

def build_chart(times, values, up_to=None):
    if up_to is not None:
        times, values = times[:up_to], values[:up_to]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name="Latency (ms)"))
    fig.add_hline(y=normal_threshold, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig.add_hline(y=critical_threshold, line_dash="dash", line_color="red", annotation_text="Critical")
    fig.update_layout(xaxis_title="Time", yaxis_title="Latency (ms)", height=450)
    return fig

chart_placeholder = st.empty()

# ---- SELF-MOVING AUTO-PLAY (no manual clicking needed once enabled) ----
if "frame" not in st.session_state:
    st.session_state.frame = 10

autoplay = st.checkbox("🔴 Live Auto-Play (simulated real-time replay, loops continuously)")

if autoplay:
    st.session_state.frame += 3
    if st.session_state.frame > len(readable_times):
        st.session_state.frame = 10  # loop back to the start automatically

    chart_placeholder.plotly_chart(
        build_chart(readable_times, view_values, up_to=st.session_state.frame),
        width="stretch",
        key=f"auto_{st.session_state.frame}",
    )

    # Decision support reacts to whatever has "arrived" so far in the loop
    visible_values = view_values[:st.session_state.frame]
    visible_times = readable_times[:st.session_state.frame]
    worst_i = int(np.argmax(visible_values))
    worst_val = visible_values[worst_i]
    worst_time = visible_times[worst_i]

    time.sleep(0.15)
    st.rerun()
else:
    st.session_state.frame = len(readable_times)
    chart_placeholder.plotly_chart(build_chart(readable_times, view_values), width="stretch")
    worst_i = int(np.argmax(view_values))
    worst_val = view_values[worst_i]
    worst_time = readable_times[worst_i]

st.subheader("Decision Support")

def classify_severity(value):
    if value >= critical_threshold:
        return "CRITICAL"
    elif value >= normal_threshold:
        return "WARNING"
    return "NORMAL"

sev = classify_severity(worst_val)

if sev == "CRITICAL":
    st.error(f"Worst event so far: {worst_time} — Latency {worst_val:.1f}ms — Severity: {sev}")
elif sev == "WARNING":
    st.warning(f"Worst event so far: {worst_time} — Latency {worst_val:.1f}ms — Severity: {sev}")
else:
    st.info(f"Worst event so far: {worst_time} — Latency {worst_val:.1f}ms — Severity: {sev}")

if sev == "CRITICAL":
    action_text = ("**Recommended action:** Check pod resource limits, scale replica "
                    "count, and inspect recent deployments for the affected service.")
elif sev == "WARNING":
    action_text = "Monitor closely; check for traffic spikes or upstream dependency slowdowns."
else:
    action_text = "No action needed — within normal operating range."
st.write(action_text)

st.subheader("Root Cause — Log-Based TF-IDF Signature")
try:
    X_tfidf, vectorizer, pod_ids = joblib.load("tfidf_rootcause.pkl")
    top_terms = vectorizer.get_feature_names_out()
    st.write(f"Vocabulary built from {len(pod_ids)} pods, {X_tfidf.shape[1]} terms")

    selected_pod = st.selectbox("Inspect a specific pod's log signature", pod_ids)
    pod_idx = pod_ids.index(selected_pod)
    pod_vector = X_tfidf[pod_idx].toarray().flatten()
    top_indices = pod_vector.argsort()[::-1][:10]
    st.write(f"Top terms for **{selected_pod}**:",
              ", ".join(top_terms[i] for i in top_indices if pod_vector[i] > 0))
except FileNotFoundError:
    st.warning("tfidf_rootcause.pkl not found.")