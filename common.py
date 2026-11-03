import os, json, numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 123, 2025, 7, 1234]
RES = os.path.join(ROOT, "results")
CONFIG_FILE = os.path.join(ROOT, "config", "defaults.json")


def public_stats(corpus, feat="cls"):
    """Standardisation statistics from the TRAINING split of an unrelated public corpus.
    Kaggle models use HateXplain-train statistics, HateXplain models use Kaggle-train statistics,
    so no validation or test features of either corpus enter the normaliser."""
    X = np.load(os.path.join(ROOT, "features", f"{corpus}_{feat}.npy"))
    if corpus == "hx":
        hx = pd.read_json(os.path.join(ROOT, "data", "hatexplain_clean.jsonl"), lines=True)
        X = X[hx.split.values == "train"]
    else:
        df = pd.read_csv(os.path.join(ROOT, "data", "kaggle_clean_v3.csv"))
        X = X[df.v1_index.values[df.split.values == "train"]]
    return X.mean(0), X.std(0) + 1e-6


def load_kaggle(feat="cls", standardize=True, version=3):
    if version == 3:
        df = pd.read_csv(os.path.join(ROOT, "data", "kaggle_clean_v3.csv"))
        X = np.load(os.path.join(ROOT, "features", f"kaggle_{feat}.npy"))[df.v1_index.values]
        mu, sd = public_stats("hx", feat)                     # HateXplain training split (public, disjoint)
        if standardize: X = (X - mu) / sd
        y = df.label.values; out = {}
        for s in ("train", "val", "test"):
            m = df.split.values == s
            out[s] = (torch.tensor(X[m], dtype=torch.float32), torch.tensor(y[m], dtype=torch.long))
        out["norm"] = (mu, sd); out["df"] = df
        return out
    return load_kaggle_v1(feat, standardize)


def load_kaggle_v1(feat="cls", standardize=True):
    df = pd.read_csv(os.path.join(ROOT, "data", "kaggle_clean.csv"))
    if feat == "concat":
        X = np.concatenate([np.load(os.path.join(ROOT, "features", "kaggle_cls.npy")),
                            np.load(os.path.join(ROOT, "features", "kaggle_mean.npy"))], 1)
    else:
        X = np.load(os.path.join(ROOT, "features", f"kaggle_{feat}.npy"))
    y = df.label.values
    out = {}
    tr = df.split.values == "train"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    if standardize: X = (X - mu) / sd
    for s in ("train", "val", "test"):
        m = df.split.values == s
        out[s] = (torch.tensor(X[m], dtype=torch.float32), torch.tensor(y[m], dtype=torch.long))
    out["norm"] = (mu, sd)
    return out


def load_hx(feat="cls", standardize=True):
    hx = pd.read_json(os.path.join(ROOT, "data", "hatexplain_clean.jsonl"), lines=True)
    X = np.load(os.path.join(ROOT, "features", f"hx_{feat}.npy")); y = hx.label.values
    mu, sd = public_stats("kaggle", feat)                      # Kaggle training split (public, disjoint)
    if standardize: X = (X - mu) / sd
    out = {s: (torch.tensor(X[hx.split.values == s], dtype=torch.float32),
               torch.tensor(y[hx.split.values == s], dtype=torch.long)) for s in ("train", "val", "test")}
    out["norm"] = (mu, sd); out["df"] = hx
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)
