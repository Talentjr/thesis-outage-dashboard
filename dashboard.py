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


def zoom_axis(crit):
    """Upper limit rounded up to one significant figure, with about ten tick steps (0, 10, 20 ... 100 for a 34 ms critical level)."""
    raw = crit * 2.5
    mag = 10 ** int(np.floor(np.log10(raw)))
    upper = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)   # 1, 2, 5 or 10 times a power of ten
    return float(upper), float(upper) / 10


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
scale_mode = st.sidebar.radio("Vertical scale", ["Zoomed on normal range", "Full range (shows the biggest spike)"])
st.sidebar.caption("Thresholds are percentiles of the selected incident's own latency, so they differ between incidents.")

tab1, tab2, tab3 = st.tabs(["Incident explorer", "Compare all incidents", "Model results"])

# ------------------------------------------------------------------ explorer
@st.cache_data
def times_for(inc):
    return to_dt(data[inc].index)


def find_events(vals, secs, warn, crit):
    """Group consecutive points at or above the warning level into events (a gap of more than ~10 sampling steps starts a new event)."""
    cols = ["Start (UTC)", "End (UTC)", "Duration (s)", "Peak (ms)", "Peak time (UTC)", "Points", "Severity"]
    idx = np.where(vals >= warn)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols)
    step = float(np.median(np.diff(secs))) if len(secs) > 1 else 1.0
    brk = np.where(np.diff(secs[idx]) > max(30.0, 10 * step))[0]
    starts, ends = np.r_[0, brk + 1], np.r_[brk, len(idx) - 1]
    rows = []
    for s_, e_ in zip(starts, ends):
        ii = idx[s_:e_ + 1]
        seg = vals[ii]
        k = int(seg.argmax())
        pk = float(seg[k])
        rows.append({"Start (UTC)": pd.to_datetime(int(secs[ii[0]]), unit="s"), "End (UTC)": pd.to_datetime(int(secs[ii[-1]]), unit="s"),
                     "Duration (s)": int(secs[ii[-1]] - secs[ii[0]]), "Peak (ms)": round(pk, 1),
                     "Peak time (UTC)": pd.to_datetime(int(secs[ii[k]]), unit="s"), "Points": len(ii),
                     "Severity": "CRITICAL" if pk >= crit else "WARNING"})
    return pd.DataFrame(rows, columns=cols)


with tab1:
    latency = data[incident]
    t_all = times_for(incident)
    warn, crit = float(latency.quantile(normal_pct / 100)), float(latency.quantile(critical_pct / 100))
    st.subheader(f"Incident {incident}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points", f"{len(latency):,}")
    c2.metric("Peak latency", f"{latency.max():.1f} ms")
    c3.metric(f"Warning ({normal_pct}th pct)", f"{warn:.1f} ms")
    c4.metric(f"Critical ({critical_pct}th pct)", f"{crit:.1f} ms")

    # ---- period picker (all times UTC)
    t_min, t_max = t_all[0], t_all[-1]
    period_mode = st.radio("Period to look at", ["Whole incident", "Pick a period"], horizontal=True, key=f"pm_{incident}")
    start_ts, end_ts = t_min, t_max
    if period_mode == "Pick a period":
        st.caption(f"This incident runs from {t_min} to {t_max} (UTC). Type or pick a start and an end.")
        p1, p2, p3, p4 = st.columns(4)
        d0 = p1.date_input("Start date", t_min.date(), min_value=t_min.date(), max_value=t_max.date(), key=f"sd_{incident}")
        h0 = p2.time_input("Start time (UTC)", t_min.time(), step=60, key=f"st_{incident}")
        d1 = p3.date_input("End date", t_max.date(), min_value=t_min.date(), max_value=t_max.date(), key=f"ed_{incident}")
        h1 = p4.time_input("End time (UTC)", t_max.time(), step=60, key=f"et_{incident}")
        start_ts, end_ts = pd.Timestamp.combine(d0, h0), pd.Timestamp.combine(d1, h1)
        if end_ts <= start_ts:
            st.warning("The end must be after the start. Showing the whole incident instead.")
            start_ts, end_ts = t_min, t_max
    in_period = (t_all >= start_ts) & (t_all <= end_ts)
    sub = latency[in_period]
    if len(sub) < 2:
        st.warning("No data in that period. Showing the whole incident instead.")
        sub, start_ts, end_ts = latency, t_min, t_max
    sub_secs, sub_vals = sub.index.values.astype("int64"), sub.values.astype("float64")

    # Downsample for the chart by taking the MAXIMUM of each bucket, so short spikes are never skipped.
    n_buckets = min(3000, len(sub))
    edges = np.linspace(0, len(sub), n_buckets + 1).astype(int)
    keep = [(a_, b_) for a_, b_ in zip(edges[:-1], edges[1:]) if b_ > a_]
    values = np.array([sub_vals[a_:b_].max() for a_, b_ in keep])
    times = to_dt(np.array([sub_secs[a_] for a_, _ in keep]))
    events = find_events(sub_vals, sub_secs, warn, crit)

    def chart(up_to=None, now_sev=None):
        t, v = (times, values) if up_to is None else (times[:up_to], values[:up_to])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=v, mode="lines", name="Latency (ms)"))
        if now_sev is not None and len(v):
            col = {"CRITICAL": "red", "WARNING": "orange", "NORMAL": "limegreen"}[now_sev]
            fig.add_trace(go.Scatter(x=[t[-1]], y=[v[-1]], mode="markers", marker=dict(size=12, color=col), name=f"Now: {now_sev}"))
        fig.add_hline(y=warn, line_dash="dash", line_color="orange", annotation_text="Warning", annotation_position="bottom right")
        fig.add_hline(y=crit, line_dash="dash", line_color="red", annotation_text="Critical", annotation_position="top right")
        if scale_mode.startswith("Zoomed"):
            upper, step_ = zoom_axis(crit)
            yr, dt = [0, upper], step_
        else:
            yr, dt = [0, float(max(values.max(), crit)) * 1.05], None
        fig.update_layout(xaxis_title="Time (UTC)", yaxis_title="Latency (ms)", height=450,
                          yaxis_range=yr, xaxis_range=[times[0], times[-1]])
        if dt:
            fig.update_yaxes(dtick=dt)
        return fig

    st.markdown("**Latency timeline**" + ("" if period_mode == "Whole incident" else f" ({start_ts} to {end_ts} UTC)"))
    holder = st.empty()
    ctx = (incident, str(start_ts), str(end_ts))
    if st.session_state.get("ctx") != ctx:              # restart the replay when the incident or period changes
        st.session_state.ctx = ctx
        st.session_state.frame = 10
    autoplay = st.checkbox("Live auto-play (simulated replay of this period, one pass = 50 seconds, loops)")

    if autoplay:
        REPLAY_SECONDS = 50                      # one full pass of the period takes 50 seconds of real time
        if "t0" not in st.session_state or st.session_state.get("t0_ctx") != ctx:
            st.session_state.t0, st.session_state.t0_ctx = time.time(), ctx
        elapsed = (time.time() - st.session_state.t0) % REPLAY_SECONDS      # loops every 50 s
        f = min(len(values), max(10, int(elapsed / REPLAY_SECONDS * len(values))))
        st.session_state.frame = f
        recent = values[max(0, f - 5):f]
        cur, cur_t = float(recent.max()), times[f - 1]
        sev = classify(cur, warn, crit)
        holder.plotly_chart(chart(f, sev), width="stretch", key=f"auto_{incident}_{f}")
        ev_now = events[events["Start (UTC)"] <= cur_t]
    else:
        st.session_state.frame = len(values)
        holder.plotly_chart(chart(), width="stretch")
        cur = float(values.max())
        sev = classify(cur, warn, crit)
        ev_now = events

    if scale_mode.startswith("Zoomed") and float(sub_vals.max()) > zoom_axis(crit)[0]:
        st.caption(f"Zoomed view: spikes above {zoom_axis(crit)[0]:.0f} ms run off the top (the peak here is {sub_vals.max():.1f} ms). "
                   "Choose 'Full range' in the sidebar to see them.")

    st.markdown("**Decision support**" + (" (live: updates as the replay moves)" if autoplay else " (for the whole period shown)"))
    n_w = int((ev_now["Severity"] == "WARNING").sum()); n_c = int((ev_now["Severity"] == "CRITICAL").sum())
    if autoplay:
        if sev == "NORMAL":
            msg = f"NORMAL at {cur_t} UTC: latency {cur:.1f} ms is below the warning level ({warn:.1f} ms)."
        else:
            msg = f"{sev} at {cur_t} UTC: latency {cur:.1f} ms (warning {warn:.1f} ms, critical {crit:.1f} ms)."
    else:
        pk_t = to_dt([int(sub_secs[int(sub_vals.argmax())])])[0]
        if sev == "NORMAL":
            msg = f"NORMAL for this period: no point reached the warning level ({warn:.1f} ms). Peak {sub_vals.max():.1f} ms at {pk_t} UTC."
        else:
            msg = f"{sev} in this period: peak {sub_vals.max():.1f} ms at {pk_t} UTC."
    {"CRITICAL": st.error, "WARNING": st.warning, "NORMAL": st.success}[sev](msg)
    st.write(ACTIONS[sev])
    st.caption(f"{'So far in the replay' if autoplay else 'In this period'}: {n_c} critical and {n_w} warning events "
               f"({int((sub_vals >= crit).sum()):,} of {len(sub_vals):,} points in the period are at or above the critical level).")

    st.markdown("**Events in this period**" + (" (up to the current replay time)" if autoplay else ""))
    if len(ev_now) == 0:
        st.info("No events: nothing in this period reached the warning level.")
    else:
        show = ev_now.sort_values("Start (UTC)").head(300).copy()
        for c_ in ("Start (UTC)", "End (UTC)", "Peak time (UTC)"):
            show[c_] = show[c_].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(show, hide_index=True, width="stretch")
        if len(ev_now) > 300:
            st.caption(f"Showing the first 300 of {len(ev_now):,} events. Choose a shorter period to see the rest.")

    st.markdown("**Root cause: log-based TF-IDF signature**")
    if incident == "20240124":
        try:
            X_tfidf, vectorizer, pod_ids = joblib.load("tfidf_rootcause.pkl")
            terms = vectorizer.get_feature_names_out()
            st.write(f"Vocabulary built from {len(pod_ids)} pods, {X_tfidf.shape[1]} terms. "
                     "These are term signatures for the whole incident, not a trained root cause classifier.")
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
