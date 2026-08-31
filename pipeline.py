import glob, os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOCAL_DIR = "./lemma_rca_cloud"
print("Skipping unzip — already extracted manually", flush=True)

INCIDENT = "20240124"
SCENARIO = "Latency"
metric_dir = f"{LOCAL_DIR}/Metrics Data/{INCIDENT}/{INCIDENT}/{SCENARIO}"
metric_files = glob.glob(f"{metric_dir}/*.npy")
print(f"Found {len(metric_files)} metric files", flush=True)

feature_frames = []
kpi_series = None
for mf in metric_files:
    d = np.load(mf, allow_pickle=True).item()
    for label, entry in d.items():
        seq = entry["Sequence"]
        n_kpi = len(entry["KPI_Feature"])
        pod_cols = seq[:, :-n_kpi] if n_kpi > 0 else seq
        kpi_cols = seq[:, -n_kpi:] if n_kpi > 0 else None
        times = np.array(entry["time"])
        metric_name = os.path.basename(mf).replace(".npy", "")
        pod_names = entry["Pod_Name"]

        df = pd.DataFrame(pod_cols, columns=[f"{metric_name}__{p}" for p in pod_names])
        df["time"] = times
        df = df.set_index("time")
        feature_frames.append(df)

        if kpi_series is None and kpi_cols is not None:
            kpi_series = pd.Series(kpi_cols[:, 0], index=times, name=entry["KPI_Feature"][0])

features = pd.concat(feature_frames, axis=1, join="inner")
features = features.join(kpi_series, how="inner")
print("Feature matrix shape:", features.shape, flush=True)

TARGET_COL = "Latency"
ANOMALY_THRESHOLD = features[TARGET_COL].quantile(0.99)
print(f"\nAnomaly threshold (99th percentile): {ANOMALY_THRESHOLD:.2f}ms")

features["is_anomaly"] = (features[TARGET_COL] > ANOMALY_THRESHOLD).astype(int)
anomaly_rate = features["is_anomaly"].mean()
print(f"Anomaly rate in this incident: {anomaly_rate*100:.2f}% of timesteps")

feature_cols = [c for c in features.columns if c not in (TARGET_COL, "is_anomaly")]

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features[feature_cols].values).astype("float32")
y_binary = features["is_anomaly"].values.astype("float32")

N = len(X_scaled)
train_end = int(N * 0.7)
val_end = train_end + int(N * 0.15)

WINDOW = 60
HORIZON = 1

def make_dataset(X, y, window=WINDOW, horizon=HORIZON, batch_size=256, shuffle=False):
    delay = window + horizon - 1
    return tf.keras.utils.timeseries_dataset_from_array(
        data=X[:-delay],
        targets=y[delay:],
        sequence_length=window,
        batch_size=batch_size,
        shuffle=shuffle,
    )

train_ds = make_dataset(X_scaled[:train_end], y_binary[:train_end], shuffle=True)
val_ds = make_dataset(X_scaled[train_end:val_end], y_binary[train_end:val_end])
test_ds = make_dataset(X_scaled[val_end:], y_binary[val_end:])

print("Train/Val/Test raw lengths:", train_end, val_end - train_end, N - val_end, flush=True)

train_y = y_binary[:train_end]
n_pos = train_y.sum()
n_neg = len(train_y) - n_pos
class_weight = {0: 1.0, 1: (n_neg / max(n_pos, 1))}
print(f"Class weights: {class_weight}")

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

n_features = X_scaled.shape[1]

model = models.Sequential([
    layers.Input(shape=(WINDOW, n_features)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(32),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc"),
              tf.keras.metrics.Precision(name="precision"),
              tf.keras.metrics.Recall(name="recall")],
)
model.summary()

early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1,
)

model.save("lstm_anomaly_classifier.keras")
print("Model saved as lstm_anomaly_classifier.keras", flush=True)

all_probs, all_true = [], []
for batch_X, batch_y in test_ds:
    probs = model.predict(batch_X, verbose=0).flatten()
    all_probs.extend(probs)
    all_true.extend(batch_y.numpy())

all_probs = np.array(all_probs)
all_true = np.array(all_true)
all_preds = (all_probs >= 0.5).astype(int)

print(f"\n=== CLASSIFICATION RESULTS ({len(all_true)} predictions) ===")
print(f"Actual anomalies in test set: {int(all_true.sum())} out of {len(all_true)} "
      f"({all_true.mean()*100:.2f}%)")
print("\nConfusion Matrix (default threshold 0.5):")
print(confusion_matrix(all_true, all_preds))
print("\nClassification report (default threshold 0.5):")
print(classification_report(all_true, all_preds, target_names=["Normal", "Anomaly"], zero_division=0))
try:
    auc = roc_auc_score(all_true, all_probs)
    print(f"ROC-AUC: {auc:.4f}")
except ValueError as e:
    print(f"Could not compute ROC-AUC: {e}")

fpr, tpr, roc_thresholds = roc_curve(all_true, all_probs)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(all_true, all_probs):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve — Outage Anomaly Classifier")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()
print("\nSaved roc_curve.png", flush=True)

precisions, recalls, pr_thresholds = precision_recall_curve(all_true, all_probs)

plt.figure(figsize=(7, 6))
plt.plot(recalls, precisions, label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Outage Anomaly Classifier")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
plt.close()
print("Saved precision_recall_curve.png", flush=True)

f1_scores = []
for p, r in zip(precisions[:-1], recalls[:-1]):
    if p + r == 0:
        f1_scores.append(0)
    else:
        f1_scores.append(2 * p * r / (p + r))

f1_scores = np.array(f1_scores)
best_idx = np.argmax(f1_scores)
best_threshold = pr_thresholds[best_idx]

print(f"\n=== THRESHOLD TUNING (F1) ===")
print(f"Best threshold by F1: {best_threshold:.4f}")
print(f"  Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")

tuned_preds = (all_probs >= best_threshold).astype(int)
print("\nConfusion Matrix at F1-tuned threshold:")
print(confusion_matrix(all_true, tuned_preds))
print("\nClassification report at F1-tuned threshold:")
print(classification_report(all_true, tuned_preds, target_names=["Normal", "Anomaly"], zero_division=0))

# ---- RECALL-PRIORITIZED THRESHOLD ----
print("\n=== RECALL-PRIORITIZED THRESHOLD SEARCH ===")
target_recall = 0.80

best_recall_threshold = None
best_p, best_r = None, None
for p, r, t in zip(precisions[:-1], recalls[:-1], pr_thresholds):
    if r >= target_recall:
        best_recall_threshold = t
        best_p, best_r = p, r

if best_recall_threshold is not None:
    print(f"Threshold achieving >= {target_recall*100:.0f}% recall: {best_recall_threshold:.4f}")
    print(f"  Precision: {best_p:.4f}, Recall: {best_r:.4f}")
    recall_preds = (all_probs >= best_recall_threshold).astype(int)
    print("\nConfusion Matrix at recall-prioritized threshold:")
    print(confusion_matrix(all_true, recall_preds))
    print("\nClassification report at recall-prioritized threshold:")
    print(classification_report(all_true, recall_preds, target_names=["Normal", "Anomaly"], zero_division=0))
else:
    print("No threshold reaches 80% recall in this run — will adjust target after seeing this.")

print("\n=== DONE — check roc_curve.png and precision_recall_curve.png ===")