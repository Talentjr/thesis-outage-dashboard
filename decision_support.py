import glob, os
import pandas as pd
import numpy as np
from datetime import datetime

LOCAL_DIR = "./lemma_rca_cloud"
INCIDENT = "20240124"

# --- Reload the same feature/target data used for the LSTM ---
SCENARIO = "Latency"
metric_dir = f"{LOCAL_DIR}/Metrics Data/{INCIDENT}/{INCIDENT}/{SCENARIO}"
metric_files = glob.glob(f"{metric_dir}/*.npy")

kpi_series = None
for mf in metric_files:
    d = np.load(mf, allow_pickle=True).item()
    for label, entry in d.items():
        if kpi_series is None:
            times = np.array(entry["time"])
            kpi_series = pd.Series(entry["Sequence"][:, -1], index=times, name="Latency")

latency = kpi_series
normal_threshold = latency.quantile(0.95)
critical_threshold = latency.quantile(0.99)

def classify_severity(value):
    if value >= critical_threshold:
        return "CRITICAL"
    elif value >= normal_threshold:
        return "WARNING"
    return "NORMAL"

def recommend_action(severity, value):
    if severity == "CRITICAL":
        return (f"Latency at {value:.1f}ms exceeds critical threshold "
                f"({critical_threshold:.1f}ms). Recommend: check pod resource limits, "
                f"scale replica count, and inspect recent deployments for the affected service.")
    elif severity == "WARNING":
        return (f"Latency at {value:.1f}ms is elevated above normal "
                f"({normal_threshold:.1f}ms). Recommend: monitor closely; check for "
                f"traffic spikes or upstream dependency slowdowns.")
    return f"Latency at {value:.1f}ms is within normal range."

# Demonstrate on a few real timestamps, including the worst spike
worst_idx = latency.idxmax()
sample_points = [latency.index[100], latency.index[50000], worst_idx]

print("=== DECISION SUPPORT DEMO (real data, incident 20240124) ===")
print(f"Normal threshold (95th pct): {normal_threshold:.2f}ms")
print(f"Critical threshold (99th pct): {critical_threshold:.2f}ms\n")

for ts in sample_points:
    val = latency.loc[ts]
    sev = classify_severity(val)
    action = recommend_action(sev, val)
    readable_time = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{readable_time}] Severity: {sev}")
    print(f"  {action}\n")

print("Decision support logic ready — will be wired into the dashboard next.")