import glob, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

LOCAL_DIR = "./lemma_rca_cloud"
INCIDENT = "20240124"

log_dir = f"{LOCAL_DIR}/Log Data/{INCIDENT}"
structured_files = glob.glob(f"{log_dir}/**/*_messages_structured.csv", recursive=True)
print(f"Found {len(structured_files)} structured log files")

docs, pod_ids = [], []
for sf in structured_files:
    df = pd.read_csv(sf)
    if "EventTemplate" not in df.columns:
        continue
    text = " ".join(df["EventTemplate"].astype(str).tolist())
    docs.append(text)
    pod_ids.append(os.path.basename(sf).replace("_messages_structured.csv", ""))

vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2), token_pattern=r"[a-zA-Z]{3,}")
X_tfidf = vectorizer.fit_transform(docs)

print("TF-IDF matrix shape:", X_tfidf.shape)
print("Top vocabulary terms:", vectorizer.get_feature_names_out()[:20])

import joblib
joblib.dump((X_tfidf, vectorizer, pod_ids), "tfidf_rootcause.pkl")
print("Saved tfidf_rootcause.pkl")