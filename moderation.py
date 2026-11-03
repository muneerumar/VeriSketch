"""Moderation operating points.

Harm score = 1 - p(not cyberbullying).  Operating points: "argmax" flags a post when the most
probable of the six classes is harmful (true multiclass argmax); "harm05" flags it when the harm
score exceeds 0.5; "far10" and "maxf1" use thresholds chosen on the validation split.  Reports recall and
FAR, precision at realistic prevalence (re-weighting the measured recall and FAR), moderator
workload, ROC-AUC/AP, calibration, per-class recall and a second corpus (HateXplain)."""
import os, json, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
from common import *
import fl
from exp import BASE


def ece(p, y, bins=15):
    e = np.linspace(0, 1, bins + 1); out = 0.0
    for a, b in zip(e[:-1], e[1:]):
        m = (p > a) & (p <= b)
        if m.any(): out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def op_point(s_val, y_val, s_te, y_te, rule, argmax_te=None):
    if rule == "argmax":
        tau = float("nan"); ph = argmax_te != 0
    elif rule == "far10":
        neg = np.sort(s_val[y_val == 0]); tau = neg[int(np.ceil(0.9 * len(neg))) - 1]
    elif rule == "maxf1":
        pr, rc, th = precision_recall_curve(y_val, s_val); f = 2 * pr * rc / (pr + rc + 1e-12); tau = th[np.argmax(f[:-1])]
    else:
        tau = 0.5
    if rule != "argmax": ph = s_te > tau
    tpr = (ph & (y_te == 1)).sum() / (y_te == 1).sum(); far = (ph & (y_te == 0)).sum() / (y_te == 0).sum()
    r = dict(rule=rule, tau=float(tau), tpr=float(tpr), far=float(far))
    for pi in (0.05, 0.10, 0.20):
        r[f"precision@{pi}"] = float(pi * tpr / (pi * tpr + (1 - pi) * far + 1e-12)); r[f"flagged@{pi}"] = float(pi * tpr + (1 - pi) * far)
    return r


def main():
    from exp import require_inputs; require_inputs()
    cfgb = json.load(open(CONFIG_FILE))
    d = load_kaggle("cls"); out = []
    # second source: HateXplain test posts (Gab/Twitter, different annotation scheme), mapped to
    # harmful = offensive or hate speech.  Features use the Kaggle model's normaliser, which is computed
    # from HateXplain TRAINING features only, so no HateXplain test feature enters preprocessing.
    hx = pd.read_json(os.path.join(ROOT, "data", "hatexplain_clean.jsonl"), lines=True)
    mu, sd = public_stats("hx", "cls")
    Xhx = torch.tensor(((np.load(os.path.join(ROOT, "features", "hx_cls.npy")) - mu) / sd)[hx.split.values == "test"], dtype=torch.float32)
    yhx = (hx.label.values[hx.split.values == "test"] > 0).astype(int)
    for s in SEEDS:
        for agg in ("secagg", "verisketch"):
            for eps in (None, 2.0):
                c = dict(BASE); c.update(cfgb); c.update(seed=s, agg=agg, partition="iid")
                if eps: c.update(lr=cfgb["dp_lr"], dp=dict(epsilon=eps, delta=1e-5, clip=cfgb["dp_clip"], q=cfgb["dp_q"]))
                o, head, pred, prob = fl.run_fl(c, *d["train"], *d["val"], *d["test"], log=lambda *a: None)
                with torch.no_grad():
                    pv = torch.softmax(head.logits(d["val"][0]), 1).numpy()
                yv = (d["val"][1].numpy() != 0).astype(int); yt = (d["test"][1].numpy() != 0).astype(int)
                sv, st = 1 - pv[:, 0], 1 - prob[:, 0]
                rec = dict(seed=s, agg=agg, eps=eps, roc_auc=float(roc_auc_score(yt, st)), ap=float(average_precision_score(yt, st)),
                           ece_harm=ece(st, yt), ece_top=ece(prob.max(1), (prob.argmax(1) == d["test"][1].numpy()).astype(float)),
                           confusion=confusion_matrix(d["test"][1].numpy(), prob.argmax(1), labels=list(range(6))).tolist(),
                           points=[op_point(sv, yv, st, yt, r, prob.argmax(1)) for r in ("argmax", "harm05", "far10", "maxf1")])
                with torch.no_grad():
                    phx = torch.softmax(head.logits(Xhx), 1).numpy(); shx = 1 - phx[:, 0]
                rec["second_source"] = dict(corpus="HateXplain test", n=int(len(yhx)), prevalence=float(yhx.mean()),
                                            roc_auc=float(roc_auc_score(yhx, shx)), ap=float(average_precision_score(yhx, shx)),
                                            points=[op_point(sv, yv, shx, yhx, r, phx.argmax(1)) for r in ("argmax", "harm05", "far10", "maxf1")])
                cm = np.array(rec["confusion"]); rec["per_class_recall"] = (cm.diagonal() / cm.sum(1)).tolist()
                pr_, rc_, _ = precision_recall_curve(yt, st); k = max(1, len(pr_) // 200)
                rec["pr"] = [pr_[::k].tolist(), rc_[::k].tolist()]
                fpr, tpr, _ = roc_curve(yt, st); rec["roc"] = [fpr[::max(1, len(fpr) // 200)].tolist(), tpr[::max(1, len(tpr) // 200)].tolist()]
                out.append(rec); print(s, agg, eps, round(rec["roc_auc"], 4), rec["points"][1], flush=True)
                from exp import save_with_provenance
                save_with_provenance(out, os.path.join(RES, "moderation", "moderation.json"), ["moderation.py"])


if __name__ == "__main__":
    main()
