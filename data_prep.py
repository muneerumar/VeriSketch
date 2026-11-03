"""Data preparation for VeriSketch-FL.

Reproduces the first-stage preprocessing and frozen split exactly
(null/empty removal -> exact-duplicate removal -> drop <3 tokens -> text
normalisation; stratified 80/20 split, seed 42), then carves a stratified
10% validation set out of the TRAINING split so that no design decision is
taken on the test set.  Also prepares HateXplain with its official split.
"""
import json, re, html, hashlib, os
from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")
CLASS_NAMES = ["not_cyberbullying", "age", "ethnicity", "gender", "religion", "other_cyberbullying"]
LABEL_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def _ws(t):
    t = html.unescape(t); t = re.sub(r"\s+", " ", t); return t.strip()

def _emo(t):
    for p, r in {r"<3": " love ", r":\)": " smile ", r":\(": " sad ", r":D": " laugh ", r"xD": " laugh "}.items():
        t = re.sub(p, r, t)
    return t

def _urls(t):
    t = re.sub(r"http\S+|www\.\S+", " URL ", t); t = re.sub(r"@\w+", " USER ", t)
    return re.sub(r"#(\w+)", r"\1", t)


def prepare_kaggle():
    raw = pd.read_csv(os.path.join(D, "cyberbullying_tweets.csv"))
    log = {"raw": len(raw)}
    df = raw.dropna(subset=["tweet_text"]).copy(); df = df[df.tweet_text.str.strip() != ""]
    log["nulls_empty_removed"] = log["raw"] - len(df)
    n = len(df); df = df.drop_duplicates(subset=["tweet_text"], keep="first").copy()
    log["duplicates_removed"] = n - len(df)
    n = len(df); df = df[df.tweet_text.str.split().str.len() >= 3].copy()
    log["short_removed"] = n - len(df)
    df["tweet_text"] = [_urls(_emo(_ws(t))).lower() for t in df.tweet_text.astype(str)]
    df = df.reset_index(drop=True)
    df["label"] = df.cyberbullying_type.str.lower().map(LABEL_TO_ID)
    assert df.label.notna().all()
    tr, te = train_test_split(df.index.values, test_size=0.2, random_state=42, stratify=df.label)
    tr2, va = train_test_split(tr, test_size=0.1, random_state=42, stratify=df.label.values[tr])
    split = np.full(len(df), "", dtype=object); split[tr2] = "train"; split[va] = "val"; split[te] = "test"
    df["split"] = split
    # hash of the original frozen split (train_idx/test_idx lists) for provenance
    h = hashlib.sha256(json.dumps({"train_idx": [int(i) for i in tr], "test_idx": [int(i) for i in te]}, indent=2).encode()).hexdigest()
    log.update(final=len(df), train=len(tr2), val=len(va), test=len(te), orig_train=len(tr), split_sha256_recomputed=h)
    df[["tweet_text", "label", "split"]].to_csv(os.path.join(D, "kaggle_clean.csv"), index=False)
    json.dump(log, open(os.path.join(D, "kaggle_prep_log.json"), "w"), indent=2)
    return df, log


HX_CLASSES = ["normal", "offensive", "hatespeech"]

def prepare_hatexplain():
    js = json.load(open(os.path.join(D, "dataset.json")))
    div = json.load(open(os.path.join(D, "post_id_divisions.json")))
    which = {pid: s for s in ("train", "val", "test") for pid in div[s]}
    rows, dropped = [], 0
    for pid, p in js.items():
        labs = [a["label"] for a in p["annotators"]]
        c = Counter(labs).most_common()
        if c[0][1] < 2: dropped += 1; continue          # no majority -> excluded (standard practice)
        lab = c[0][0]
        toks = p["post_tokens"]
        rat = None
        if lab != "normal" and len(p["rationales"]) > 0:
            rr = [r for r in p["rationales"] if len(r) == len(toks)]
            m = np.array(rr, dtype=float) if rr else None
            if m is not None:
                rat = (m.mean(0) >= 0.5).astype(int).tolist()  # majority-of-rationale-annotators
        tg = Counter(t for a in p["annotators"] for t in set(a["target"]) if t not in ("None", "Other"))
        targets = [t for t, k in tg.items() if k >= 2]
        rows.append(dict(post_id=pid, tokens=toks, text=" ".join(toks), label=HX_CLASSES.index(lab),
                         rationale=rat, targets=targets, split=which.get(pid, "none")))
    hx = pd.DataFrame(rows)
    hx.to_json(os.path.join(D, "hatexplain_clean.jsonl"), orient="records", lines=True)
    log = dict(total=len(js), no_majority_dropped=dropped, kept=len(hx), split=hx.split.value_counts().to_dict(),
               labels=hx.label.value_counts().sort_index().tolist(),
               with_rationale=int(hx.rationale.notna().sum()))
    json.dump(log, open(os.path.join(D, "hatexplain_prep_log.json"), "w"), indent=2)
    return hx, log




# ============================================================================ duplicate-safe split used in the paper
def norm_key(t):
    t = re.sub(r"\b(url|user)\b", " ", t.lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def prepare_kaggle_v2():
    """Normalise first, then group (near-)duplicates on a normalised key, drop groups with
    conflicting labels, keep one representative per group, and only then split.
    Keeps the index of the first-stage cleaned corpus so the frozen features can be reused."""
    raw = pd.read_csv(os.path.join(D, "cyberbullying_tweets.csv"))
    log = {"raw": len(raw)}
    df = raw.dropna(subset=["tweet_text"]).copy(); df = df[df.tweet_text.str.strip() != ""]
    df = df.drop_duplicates(subset=["tweet_text"], keep="first")            # first stage (exact raw duplicates)
    df = df[df.tweet_text.str.split().str.len() >= 3].copy()
    df["tweet_text"] = [_urls(_emo(_ws(t))).lower() for t in df.tweet_text.astype(str)]
    df = df.reset_index(drop=True); df["v1_index"] = np.arange(len(df))    # == row index of the first-stage corpus and features
    df["label"] = df.cyberbullying_type.str.lower().map(LABEL_TO_ID)
    log["after_v1_cleaning"] = len(df)
    df["key"] = df.tweet_text.map(norm_key)
    df = df[df.key.str.split().str.len() >= 1]
    g = df.groupby("key").label.agg(["nunique", "size"])
    conflict_keys = set(g.index[g["nunique"] > 1]); dup_keys = set(g.index[(g["nunique"] == 1) & (g["size"] > 1)])
    log["normalised_duplicate_groups"] = int(len(dup_keys) + len(conflict_keys))
    log["conflicting_label_groups"] = len(conflict_keys)
    log["rows_in_conflicting_groups_dropped"] = int(df.key.isin(conflict_keys).sum())
    df = df[~df.key.isin(conflict_keys)]
    n = len(df); df = df.drop_duplicates(subset=["key"], keep="first")
    log["duplicate_rows_dropped"] = n - len(df)
    df = df.reset_index(drop=True)
    tr, te = train_test_split(df.index.values, test_size=0.2, random_state=42, stratify=df.label)
    tr2, va = train_test_split(tr, test_size=0.1, random_state=42, stratify=df.label.values[tr])
    split = np.full(len(df), "", dtype=object); split[tr2] = "train"; split[va] = "val"; split[te] = "test"
    df["split"] = split
    assert df.key.duplicated().sum() == 0
    log.update(final=len(df), train=len(tr2), val=len(va), test=len(te),
               class_counts=df.label.value_counts().sort_index().tolist(),
               cross_split_duplicate_keys=0)
    df[["tweet_text", "label", "split", "v1_index", "key"]].to_csv(os.path.join(D, "kaggle_clean_v2.csv"), index=False)
    json.dump(log, open(os.path.join(D, "kaggle_prep_log_v2.json"), "w"), indent=2)
    return df, log


def prepare_kaggle_v3():
    """Label-conflict policy used in the paper.

    1. Eligible rows: non-empty raw text with at least three whitespace tokens.
    2. Every eligible raw row is normalised (no earlier duplicate removal) and keyed by norm_key.
    3. Every key whose rows carry more than one label is removed completely (all its rows).
    4. Each remaining key keeps its first raw row.  The representative's row in the first-stage
       corpus (v1_index) indexes the frozen features.
    5. Stratified 80/20 split (seed 42), then a stratified 10% validation split of the training part.
    Writes the corpus, a row-level audit trail (one line per eligible raw row) and the split manifest."""
    raw = pd.read_csv(os.path.join(D, "cyberbullying_tweets.csv"))
    raw["raw_index"] = np.arange(len(raw))
    log = {"raw": len(raw)}
    ok = raw.tweet_text.notna() & (raw.tweet_text.astype(str).str.strip() != "")
    ok &= raw.tweet_text.astype(str).str.split().str.len() >= 3
    df = raw[ok].copy()
    log["eligible"] = len(df)
    # first-stage corpus (index of the frozen features): exact raw duplicates removed keep-first, then >= 3 tokens
    v1 = raw.dropna(subset=["tweet_text"]); v1 = v1[v1.tweet_text.str.strip() != ""]
    v1 = v1.drop_duplicates(subset=["tweet_text"], keep="first")
    v1 = v1[v1.tweet_text.str.split().str.len() >= 3]
    v1_of_raw = dict(zip(v1.raw_index.values, np.arange(len(v1))))
    df["tweet_text"] = [_urls(_emo(_ws(t))).lower() for t in df.tweet_text.astype(str)]
    df["label"] = df.cyberbullying_type.str.lower().map(LABEL_TO_ID)
    assert df.label.notna().all()
    df["key"] = df.tweet_text.map(norm_key)
    empty = df.key.str.len() == 0
    log["empty_key_removed"] = int(empty.sum()); df = df[~empty]
    g = df.groupby("key").label.agg(["nunique", "size"])
    conflict = set(g.index[g["nunique"] > 1])
    exact = raw[ok].groupby("tweet_text").cyberbullying_type.nunique()
    log["exact_text_conflict_groups"] = int((exact > 1).sum())
    log["exact_text_conflict_rows"] = int(raw[ok].tweet_text.isin(exact.index[exact > 1]).sum())
    log["normalised_groups_with_conflict"] = len(conflict)
    log["rows_in_conflicting_groups_removed"] = int(df.key.isin(conflict).sum())
    first = df.groupby("key").raw_index.transform("min")
    df["decision"] = np.where(df.key.isin(conflict), "conflict_removed",
                              np.where(df.raw_index == first, "kept", "duplicate_removed"))
    df["representative"] = first
    log["duplicate_rows_removed"] = int((df.decision == "duplicate_removed").sum())
    log["normalised_groups_size_gt1_consistent"] = int(((g["nunique"] == 1) & (g["size"] > 1)).sum())
    audit = df[["raw_index", "label", "decision", "representative"]].copy()
    audit["key_sha256"] = [hashlib.sha256(k.encode()).hexdigest() for k in df.key]
    keep = df[df.decision == "kept"].copy().sort_values("raw_index").reset_index(drop=True)
    keep["v1_index"] = [v1_of_raw[r] for r in keep.raw_index]        # representatives are first raw occurrences
    tr, te = train_test_split(keep.index.values, test_size=0.2, random_state=42, stratify=keep.label)
    tr2, va = train_test_split(tr, test_size=0.1, random_state=42, stratify=keep.label.values[tr])
    split = np.full(len(keep), "", dtype=object); split[tr2] = "train"; split[va] = "val"; split[te] = "test"
    keep["split"] = split
    assert keep.key.duplicated().sum() == 0
    log.update(final=len(keep), train=len(tr2), val=len(va), test=len(te),
               class_counts=keep.label.value_counts().sort_index().tolist(), cross_split_duplicate_keys=0)
    keep[["tweet_text", "label", "split", "v1_index", "raw_index", "key"]].to_csv(os.path.join(D, "kaggle_clean_v3.csv"), index=False)
    audit.sort_values("raw_index").to_csv(os.path.join(D, "kaggle_group_audit_v3.csv"), index=False)
    man = pd.DataFrame(dict(raw_index=keep.raw_index, label=keep.label, split=keep.split,
                            key_sha256=[hashlib.sha256(k.encode()).hexdigest() for k in keep.key]))
    man.to_csv(os.path.join(D, "split_manifest.csv"), index=False, lineterminator="\n")
    log["manifest_sha256"] = file_sha256(os.path.join(D, "split_manifest.csv"))
    log["audit_trail_sha256"] = file_sha256(os.path.join(D, "kaggle_group_audit_v3.csv"))
    json.dump(log, open(os.path.join(D, "kaggle_prep_log_v3.json"), "w"), indent=2)
    return keep, log


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


EXPECTED = os.path.join(ROOT, "MANIFEST_SHA256.txt")


if __name__ == "__main__":
    import sys
    _, l1 = prepare_kaggle(); print(json.dumps(l1, indent=1))          # first-stage corpus (feature index)
    _, l2 = prepare_hatexplain(); print(json.dumps(l2, indent=1))
    _, l3 = prepare_kaggle_v3(); print(json.dumps(l3, indent=1))      # corpus and split used in the paper
    if os.path.exists(EXPECTED):
        exp = open(EXPECTED).read().split()[0]
        ok = exp == l3["manifest_sha256"]
        print("manifest SHA-256", l3["manifest_sha256"], "matches" if ok else "DOES NOT MATCH", "the published value")
        if not ok: sys.exit(1)
    elif "--write-expected" in sys.argv:
        open(EXPECTED, "w").write(l3["manifest_sha256"] + "  data/split_manifest.csv\n")
