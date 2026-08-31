import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Cloud Outage Forecasting Dashboard", layout="wide")
st.title("Predictive Outage Forecasting & Root Cause Analysis")
st.caption("Incident 20240124 — LEMMA-RCA Cloud Computing dataset")

@st.cache_data
def load_latency():
    with open("dashboard_data.pkl", "rb") as f:
        return pickle.load(f)

latency = load_latency()
normal_threshold = latency.quantile(0.95)
critical_threshold = latency.quantile(0.99)

col1, col2, col3 = st.columns(3)
col1.metric("Peak Latency", f"{latency.max():.1f} ms")
col2.metric("Normal Threshold (95th pct)", f"{normal_threshold:.1f} ms")
col3.metric("Critical Threshold (99th pct)", f"{critical_threshold:.1f} ms")

st.subheader("Latency Timeline")
sample = latency.iloc[::50]
readable_times = [datetime.utcfromtimestamp(t) for t in sample.index]

fig = go.Figure()
fig.add_trace(go.Scatter(x=readable_times, y=sample.values, mode="lines", name="Latency (ms)"))
fig.add_hline(y=normal_threshold, line_dash="dash", line_color="orange", annotation_text="Warning threshold")
fig.add_hline(y=critical_threshold, line_dash="dash", line_color="red", annotation_text="Critical threshold")
fig.update_layout(xaxis_title="Time", yaxis_title="Latency (ms)", height=450)
st.plotly_chart(fig, width="stretch")

st.subheader("Decision Support")

def classify_severity(value):
    if value >= critical_threshold:
        return "CRITICAL"
    elif value >= normal_threshold:
        return "WARNING"
    return "NORMAL"

worst_idx = latency.idxmax()
worst_val = latency.loc[worst_idx]
worst_time = datetime.utcfromtimestamp(worst_idx)
sev = classify_severity(worst_val)

if sev == "CRITICAL":
    st.error(f"Worst event: {worst_time} — Latency {worst_val:.1f}ms — Severity: {sev}")
else:
    st.info(f"Worst event: {worst_time} — Latency {worst_val:.1f}ms — Severity: {sev}")

if sev == "CRITICAL":
    action_text = ("**Recommended action:** Check pod resource limits, scale replica "
                    "count, and inspect recent deployments for the affected service.")
else:
    action_text = "Monitor closely; check for traffic spikes or upstream dependency slowdowns."
st.write(action_text)

st.subheader("Root Cause — Log-Based TF-IDF Signature")
try:
    X_tfidf, vectorizer, pod_ids = joblib.load("tfidf_rootcause.pkl")
    top_terms = vectorizer.get_feature_names_out()
    st.write(f"Vocabulary built from {len(pod_ids)} pods, {X_tfidf.shape[1]} terms")
    st.write("Sample top terms across pods:", ", ".join(top_terms[:15]))
except FileNotFoundError:
    st.warning("tfidf_rootcause.pkl not found.")