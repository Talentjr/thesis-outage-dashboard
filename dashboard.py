import os
import pickle
import time
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cloud Outage Forecasting Dashboard", layout="wide")
st.title("Predictive Outage Forecasting & Root Cause Analysis")
st.caption("LEMMA-RCA Cloud Computing dataset: six recorded incidents (simulated replay, not a live system)")


@st.cache_data
def load_all():
    with open("dashboard_data_all.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_results():
    out = {}
    for name in ("headline_table.csv", "paper_results_summary_all.csv"):
        if os.path.exists(name):
            out[name] = pd.read_csv(name)
    return out


def to_dt(index):
    return pd.to_datetime(np.asarray(index, dtype="int64"), unit="s")


def classify(value, warn, crit):
    if value >= crit:
        return "CRITICAL"
    if value >= warn:
        return "WARNING"
    return "NORMAL"


ACTIONS = {
    "CRITICAL": ("**Recommended action:** Check pod resource limits, scale replica count, "
                 "and inspect recent deployments for the affected service."),
    "WARNING": "Monitor closely; check for traffic spikes or upstream dependency slowdowns.",
    "NORMAL": "No action needed: within normal operating range.",
}

try:
    data = load_all()
except FileNotFoundError:
    st.error("dashboard_data_all.pkl not found. Run prep_dashboard_data.py and add the file to the app.")
    st.stop()

incidents = sorted(data)
st.sidebar.header("Choose data and sensitivity")
incident = st.sidebar.selectbox("Incident", incidents, index=incidents.index("20240124") if "20240124" in incidents else 0)
normal_pct = st.sidebar.slider("Warning threshold (percentile)", 80, 99, 95)
critical_pct = st.sidebar.slider("Critical threshold (percentile)", normal_pct + 1, 100, 99)
st.sidebar.caption("Thresholds are percentiles of the selected incident's own latency, so they differ between incidents.")

tab1, tab2, tab3 = st.tabs(["Incident explorer", "Compare all incidents", "Model results"])

# ------------------------------------------------------------------ explorer
with tab1:
    latency = data[incident]
    warn, crit = float(latency.quantile(normal_pct / 100)), float(latency.quantile(critical_pct / 100))
    st.subheader(f"Incident {incident}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points", f"{len(latency):,}")
    c2.metric("Peak latency", f"{latency.max():.1f} ms")
    c3.metric(f"Warning ({normal_pct}th pct)", f"{warn:.1f} ms")
    c4.metric(f"Critical ({critical_pct}th pct)", f"{crit:.1f} ms")

    # Downsample for the chart by taking the MAXIMUM of each bucket, so short spikes are never skipped.
    n_buckets = min(3000, len(latency))
    edges = np.linspace(0, len(latency), n_buckets + 1).astype(int)
    vals_all = latency.values
    values = np.array([vals_all[a:b].max() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    first_idx = np.array([latency.index[a] for a, b in zip(edges[:-1], edges[1:]) if b > a])
    times = to_dt(first_idx)

    def chart(up_to=None):
        t, v = (times, values) if up_to is None else (times[:up_to], values[:up_to])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=v, mode="lines", name="Latency (ms)"))
        fig.add_hline(y=warn, line_dash="dash", line_color="orange", annotation_text="Warning")
        fig.add_hline(y=crit, line_dash="dash", line_color="red", annotation_text="Critical")
        fig.update_layout(xaxis_title="Time (UTC)", yaxis_title="Latency (ms)", height=450,
                          yaxis_range=[0, float(max(values.max(), crit)) * 1.05],
                          xaxis_range=[times[0], times[-1]])
        return fig

    st.markdown("**Latency timeline**")
    holder = st.empty()
    if st.session_state.get("inc") != incident:        # reset replay when the incident changes
        st.session_state.inc = incident
        st.session_state.frame = 10
    autoplay = st.checkbox("Live auto-play (simulated replay, loops continuously)")
    if autoplay:
        st.session_state.frame += max(1, len(values) // 150)
        if st.session_state.frame > len(values):
            st.session_state.frame = 10
        f = st.session_state.frame
        holder.plotly_chart(chart(f), width="stretch", key=f"auto_{incident}_{f}")
        vis = values[:f]
        wi = int(np.argmax(vis)); wval, wtime = float(vis[wi]), times[wi]
    else:
        st.session_state.frame = len(values)
        holder.plotly_chart(chart(), width="stretch")
        wi = int(np.argmax(values)); wval, wtime = float(values[wi]), times[wi]

    st.markdown("**Decision support**")
    sev = classify(wval, warn, crit)
    msg = f"Worst event{' so far' if autoplay else ''}: {wtime} UTC, latency {wval:.1f} ms, severity {sev}"
    {"CRITICAL": st.error, "WARNING": st.warning, "NORMAL": st.info}[sev](msg)
    st.write(ACTIONS[sev])
    n_crit = int((latency >= crit).sum())
    st.caption(f"{n_crit:,} of {len(latency):,} points ({100 * n_crit / len(latency):.2f}%) are at or above the critical threshold.")

    st.markdown("**Root cause: log-based TF-IDF signature**")
    if incident == "20240124":
        try:
            X_tfidf, vectorizer, pod_ids = joblib.load("tfidf_rootcause.pkl")
            terms = vectorizer.get_feature_names_out()
            st.write(f"Vocabulary built from {len(pod_ids)} pods, {X_tfidf.shape[1]} terms. "
                     "These are term signatures, not a trained root cause classifier.")
            pod = st.selectbox("Inspect a specific pod's log signature", pod_ids)
            vec = X_tfidf[pod_ids.index(pod)].toarray().flatten()
            top = vec.argsort()[::-1][:10]
            st.write(f"Top terms for **{pod}**:", ", ".join(terms[i] for i in top if vec[i] > 0))
        except FileNotFoundError:
            st.warning("tfidf_rootcause.pkl not found.")
    else:
        st.info("Log signatures have only been generated for incident 20240124.")

    if autoplay:
        time.sleep(0.1)
        st.rerun()

# ------------------------------------------------------------------ compare
with tab2:
    st.subheader("All incidents side by side")
    rows = []
    for inc in incidents:
        s = data[inc]
        w, c = s.quantile(normal_pct / 100), s.quantile(critical_pct / 100)
        rows.append({"Incident": inc, "Points": len(s), "Peak latency (ms)": round(float(s.max()), 1),
                     "Median (ms)": round(float(s.median()), 1), f"Warning {normal_pct}th pct (ms)": round(float(w), 1),
                     f"Critical {critical_pct}th pct (ms)": round(float(c), 1),
                     "Start (UTC)": str(to_dt(s.index[:1])[0]), "End (UTC)": str(to_dt(s.index[-1:])[0])})
    comp = pd.DataFrame(rows)
    st.dataframe(comp, hide_index=True, width="stretch")
    fig = go.Figure(go.Bar(x=comp["Incident"], y=comp["Peak latency (ms)"]))
    fig.update_layout(title="Peak latency by incident", xaxis_title="Incident", yaxis_title="ms", height=350)
    st.plotly_chart(fig, width="stretch")
    skipped = [i for i in ["20231207", "20231221", "20240115", "20240124", "20240207", "20240215"] if i not in data]
    if skipped:
        st.caption("Not shown: " + ", ".join(skipped) + ". Incident 20240215 records total cost per hour, not latency, with only 115 samples, so it is not comparable.")

# ------------------------------------------------------------------ results
with tab3:
    st.subheader("How well does the LSTM predict latency anomalies?")
    res = load_results()
    if "headline_table.csv" in res:
        st.dataframe(res["headline_table.csv"], hide_index=True, width="stretch")
        st.markdown(
            "- Only incident 20240124 shows skill (ROC-AUC about 0.67).\n"
            "- On other incidents the LSTM is at or near chance, or unstable across seeds.\n"
            "- Incident 20231221 cannot be scored: latency in its test period is above the training threshold throughout.\n"
            "- Baselines that use past latency beat the LSTM on every scorable incident.")
        st.caption("Five seeds per incident; threshold = 99th percentile of the training split; chronological 70/15/15 split.")
    else:
        st.info("headline_table.csv not found.")
    if "paper_results_summary_all.csv" in res:
        with st.expander("Full results (all models)"):
            st.dataframe(res["paper_results_summary_all.csv"], hide_index=True, width="stretch")
