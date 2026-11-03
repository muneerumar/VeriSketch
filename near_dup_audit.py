"""Near-duplicate audit between the test split and the training/validation splits.

TF-IDF (word 1-2 grams) cosine of each test post to its nearest training or validation post;
counts above thresholds; the per-post maximum is saved for the sensitivity analysis."""
import os, json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from common import ROOT, RES, save_json
df = pd.read_csv(os.path.join(ROOT, "data", "kaggle_clean_v3.csv"))
te = df[df.split == "test"].reset_index(drop=True); tr = df[df.split != "test"].reset_index(drop=True)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(df.key.fillna(""))
A = vec.transform(te.key.fillna("")); B = vec.transform(tr.key.fillna("")).T.tocsc()
best = np.zeros(len(te)); arg = np.zeros(len(te), int)
for i in range(0, len(te), 500):
    S = (A[i:i + 500] @ B).toarray(); best[i:i + 500] = S.max(1); arg[i:i + 500] = S.argmax(1)
lab_same = te.label.values == tr.label.values[arg]
out = dict(n_test=len(te), thresholds={})
for t in (0.8, 0.9, 0.95):
    m = best >= t
    out["thresholds"][str(t)] = dict(n=int(m.sum()), frac=float(m.mean()), same_label_frac=float(lab_same[m].mean()) if m.any() else None)
np.save(os.path.join(RES, "near_dup_test_maxcos.npy"), best)
save_json(out, os.path.join(RES, "near_dup_audit.json")); print(out)
