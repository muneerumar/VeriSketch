"""Experiment grid runner (resumable).  Usage: python exp.py <group> [shard i/n]"""
import sys, os, json, itertools, time, traceback
import numpy as np, torch
from common import *
import fl

BASE = dict(K=10, T=30, lr=1e-3, batch=64, local_epochs=1, dropout=0.1, d_hidden=768,
            clip_update=None, f_assumed=3, attack="none", n_malicious=0, dp=None, partition="iid", eval_every=5)
CFG_FILE = CONFIG_FILE


def base():
    b = dict(BASE)
    if os.path.exists(CFG_FILE): b.update(json.load(open(CFG_FILE)))
    return b


ROBUST_AGGS = ["fedavg", "mkrum", "median", "trimmed", "adaptive_plain", "verisketch", "verisketch_mk"]
ATTACKS_V2 = ["flip_evasion", "flip_rotate", "signflip", "signflip_nm", "gauss", "alie", "ipm", "adaptive"]


def grid(group):
    B = base(); runs = []
    def add(name, **kw):
        c = dict(B); c.update(kw); runs.append((name, c))
    dpcfg = lambda eps: dict(epsilon=eps, delta=1e-5, clip=B["dp_clip"], q=B["dp_q"])
    if group == "baseline":
        for s in SEEDS:
            add(f"central_s{s}", K=1, T=40, local_epochs=1, seed=s, agg="fedavg", clip_update=1e9, select_best_val=True, eval_every=1)
            for part in ("iid", 0.5, 0.1):
                for agg in ("fedavg", "secagg", "krum") + tuple(a for a in ROBUST_AGGS if a != "fedavg"):
                    add(f"{agg}_{part}_s{s}", partition=part, seed=s, agg=agg)
    elif group == "robust":
        for s in SEEDS:
            for part in ("iid", 0.5):
                for atk in ATTACKS_V2:
                    for agg in ROBUST_AGGS:
                        add(f"{atk}_{agg}_{part}_s{s}", partition=part, seed=s, agg=agg, attack=atk, n_malicious=3)
    elif group == "joint":      # DP and poisoning together (no-DP cells come from the robust/baseline groups)
        for s in SEEDS:
            for part in ("iid", 0.5):
                for atk in ("none", "flip_evasion", "signflip_nm", "adaptive"):
                    for agg in ("secagg", "mkrum", "adaptive_plain", "verisketch", "verisketch_mk"):
                        add(f"eps2_{atk}_{agg}_{part}_s{s}", partition=part, seed=s, agg=agg, lr=B["dp_lr"], dp=dpcfg(2.0),
                            attack=atk, n_malicious=(3 if atk != "none" else 0))
                    if True:   # aggregate-only SecAgg under attack without DP (robust grid has plaintext FedAvg)
                        if atk != "none": add(f"nodp_{atk}_secagg_{part}_s{s}", partition=part, seed=s, agg="secagg", attack=atk, n_malicious=3)
    elif group == "dpctl":     # noise controls for the joint grid: same DP-SGD pipeline with sigma = 0, and eps = 8
        for s in SEEDS:
            for part in ("iid", 0.5):
                for atk in ("none", "flip_evasion", "signflip_nm", "adaptive"):
                    for agg in ("secagg", "mkrum", "adaptive_plain", "verisketch", "verisketch_mk"):
                        for tag, eps in (("clip", None), ("eps8", 8.0)):
                            add(f"{tag}_{atk}_{agg}_{part}_s{s}", partition=part, seed=s, agg=agg, lr=B["dp_lr"],
                                dp=dict(dpcfg(2.0), epsilon=eps), attack=atk, n_malicious=(3 if atk != "none" else 0))
    elif group == "fsweep":
        for s in SEEDS[:3]:
            for f in (1, 2, 4):
                for atk in ("alie", "flip_evasion", "signflip", "adaptive"):
                    for agg in ("fedavg", "mkrum", "median", "verisketch", "verisketch_mk"):
                        add(f"f{f}_{atk}_{agg}_s{s}", seed=s, agg=agg, attack=atk, n_malicious=f, f_assumed=max(f, 1))
    elif group == "sketchdim":
        for s in SEEDS[:3]:
            for rows, width in ((1, 16), (2, 32), (4, 16), (4, 64), (4, 256), (8, 128)):
                for atk in ("none", "adaptive", "signflip_nm", "flip_evasion"):
                    add(f"r{rows}w{width}_{atk}_s{s}", seed=s, agg="verisketch", attack=atk,
                        n_malicious=(3 if atk != "none" else 0), rows=rows, width=width)
    elif group == "nullspace":
        for s in SEEDS[:3]:
            for f in (1, 3):
                for sm in ("predictable", "post_commit"):
                    add(f"vs_{sm}_f{f}_s{s}", seed=s, agg="verisketch", attack="nullspace", n_malicious=f, sigma_mode=sm)
                for agg in ("fedavg", "mkrum", "median"):
                    add(f"{agg}_f{f}_s{s}", seed=s, agg=agg, attack="nullspace", n_malicious=f)
    elif group == "elsa":       # ELSA-style norm-bound baseline (aggregate-only), same grid as robust + clean
        for s in SEEDS:
            for part in ("iid", 0.5):
                for atk in ["none"] + ATTACKS_V2:
                    add(f"{atk}_normbound_{part}_s{s}", partition=part, seed=s, agg="normbound", attack=atk,
                        n_malicious=(3 if atk != "none" else 0))
    elif group == "dp":
        for s in SEEDS:
            for part in ("iid", 0.5):
                for eps in (0.5, 1.0, 2.0, 4.0, 8.0):
                    for agg in ("secagg", "verisketch"):
                        add(f"eps{eps}_{agg}_{part}_s{s}", partition=part, seed=s, agg=agg, lr=B["dp_lr"], dp=dpcfg(eps))
    else:
        raise ValueError(group)
    return runs


def mia(head, d, n=5000):
    """Loss-threshold membership inference (Yeom et al., 2018) on the final global model."""
    from sklearn.metrics import roc_curve, roc_auc_score
    g = np.random.default_rng(0)
    Xtr, ytr = d["train"]; Xte, yte = d["test"]
    im = g.choice(len(ytr), n, replace=False); inn = g.choice(len(yte), n, replace=False)
    with torch.no_grad():
        lm = torch.nn.functional.cross_entropy(head.logits(Xtr[im]), ytr[im], reduction="none").numpy()
        ln = torch.nn.functional.cross_entropy(head.logits(Xte[inn]), yte[inn], reduction="none").numpy()
    s = np.concatenate([-lm, -ln]); lab = np.concatenate([np.ones(n), np.zeros(n)])
    fpr, tpr, _ = roc_curve(lab, s)
    return dict(auc=float(roc_auc_score(lab, s)), tpr_at_1fpr=float(np.interp(0.01, fpr, tpr)),
                gap_train_test_loss=float(ln.mean() - lm.mean()))


def file_sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def provenance(cfg):
    """Hashes that tie a result file to its inputs: split manifest, features, normaliser source,
    protocol/training code and the exact run configuration."""
    import hashlib
    feat = cfg.get("feat", "cls")
    return dict(manifest_sha256=file_sha256(os.path.join(ROOT, "data", "split_manifest.csv")),
                features_sha256=file_sha256(os.path.join(ROOT, "features", f"kaggle_{feat}.npy")),
                code_sha256={f: file_sha256(os.path.join(ROOT, f)) for f in ("secagg.py", "fl.py", "common.py")},
                config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest())


def main():
    group = sys.argv[1]
    shard, nsh = (map(int, sys.argv[2].split("/")) if len(sys.argv) > 2 else (0, 1))
    runs = grid(group)[shard::nsh]
    data_cache = {}; failed, stale, done = [], [], 0
    for name, cfg in runs:
        path = os.path.join(RES, group, name + ".json")
        prov = provenance(cfg)
        if os.path.exists(path):
            old = json.load(open(path)).get("provenance")
            if old != prov: stale.append(name)                     # incompatible output: reported, never reused silently
            else: done += 1
            continue
        feat = cfg.get("feat", base().get("feat", "cls"))
        if feat not in data_cache: data_cache[feat] = load_kaggle(feat)
        d = data_cache[feat]
        if cfg["clip_update"] is None: cfg["clip_update"] = 1e9
        t = time.time()
        try:
            out, head, pred, prob = fl.run_fl(cfg, *d["train"], *d["val"], *d["test"], log=lambda *a: None)
        except Exception:
            traceback.print_exc(); failed.append(name); continue
        out["provenance"] = prov
        out["test_pred"] = pred.astype(np.int8).tolist()
        os.makedirs(os.path.join(RES, "probs", group), exist_ok=True)
        np.savez_compressed(os.path.join(RES, "probs", group, name + ".npz"), prob=prob.astype(np.float16))
        out["mia"] = mia(head, d)
        save_json(out, path); done += 1
        print(f"[{group}] {name}: f1={out['test']['macro_f1']:.4f} tmr={out['test']['tmr']:.4f} "
              f"far={out['test']['far']:.4f} aborted={out['n_aborted']} ({time.time()-t:.0f}s)", flush=True)
    print(f"[{group} {shard}/{nsh}] complete={done} failed={len(failed)} stale={len(stale)} of {len(runs)}", flush=True)
    for n_ in failed: print("FAILED", n_, flush=True)
    for n_ in stale: print("STALE (provenance mismatch, delete to rerun)", n_, flush=True)
    if failed or stale: sys.exit(1)


if __name__ == "__main__":
    main()
