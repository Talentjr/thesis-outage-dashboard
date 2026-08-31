import glob, os
import numpy as np
import pandas as pd
import pickle

LOCAL_DIR = "./lemma_rca_cloud"
INCIDENT = "20240124"
SCENARIO = "Latency"

metric_dir = f"{LOCAL_DIR}/Metrics Data/{INCIDENT}/{INCIDENT}/{SCENARIO}"
metric_files = glob.glob(f"{metric_dir}/*.npy")

for mf in metric_files:
    d = np.load(mf, allow_pickle=True).item()
    for label, entry in d.items():
        times = np.array(entry["time"])
        latency = pd.Series(entry["Sequence"][:, -1], index=times, name="Latency")
        break
    break

with open("dashboard_data.pkl", "wb") as f:
    pickle.dump(latency, f)

print(f"Saved dashboard_data.pkl — {len(latency)} points, "
      f"{os.path.getsize('dashboard_data.pkl') / 1024:.1f} KB")