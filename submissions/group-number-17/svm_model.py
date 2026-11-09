"""
Model B: word TF-IDF + linear SVM.
Step 2: baseline with one C value.

Run (from team folder): python svm_model.py
"""
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# Load data and the shared validation split
train = pd.read_csv("data/train.csv")
val_ids = set(pd.read_csv("val_ids.csv")["id"])
is_val = train["id"].isin(val_ids)
tr, va = train[~is_val], train[is_val]
print(f"Train rows: {len(tr)}, Val rows: {len(va)}")

# Word TF-IDF (single words + word pairs) -> linear SVM
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LinearSVC(C=1.0),
)

start = time.time()
model.fit(tr["text"], tr["label"])
secs = time.time() - start

# AUROC only needs a ranking score, so we use the SVM's decision_function
train_auc = roc_auc_score(tr["label"], model.decision_function(tr["text"]))
val_auc = roc_auc_score(va["label"], model.decision_function(va["text"]))

print(f"C=1.0  train AUROC={train_auc:.4f}  val AUROC={val_auc:.4f}  ({secs:.1f}s)")