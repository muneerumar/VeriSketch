"""What the aggregation server can infer about a client's label mix.

The model follows the actual VeriSketch trajectory (masked aggregation with sketch-space
filtering).  Attacked views: the 4x64 and 4x256 sketches, the plaintext update (all
parameters and output layer only), the released aggregate, and K-1 colluding clients (who
isolate the target's update exactly).  The population prior is a baseline.  Shadow clients
span the real client-size range and are observed in several rounds (multi-round attack).
Results are an empirical evaluation of these attacks, not a bound."""
import os, json, numpy as np, torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from common import *
import fl, secagg as sa

N_SHADOW = int(os.environ.get("N_SHADOW", 300))
ROUNDS_OBS = (1, 10, 20)
SKETCHES = ((4, 64), (4, 256))


def last_layer(u, head):
    n1 = head.shapes[0][0] * head.shapes[0][1] + head.shapes[1][0]
    return u[n1:].numpy()


def fit_predict(Xs, Ps, Xr):
    sc = StandardScaler().fit(Xs)
    m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(sc.transform(Xs), Ps)
    pr = np.clip(m.predict(sc.transform(Xr)), 0, None) + 1e-9
    return pr / pr.sum(1, keepdims=True)


def run(seed, dp_eps, cfgb):
    from exp import BASE
    c = dict(BASE); c.update(cfgb); c.update(seed=seed, partition=0.5, agg="verisketch")
    d = load_kaggle(cfgb.get("feat", "cls")); Xtr, ytr = d["train"]; Xva, yva = d["val"]
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    parts = fl.dirichlet_partition(ytr.numpy(), c["K"], 0.5, rng); sizes = [len(p) for p in parts]
    c["class_weights"] = torch.ones(6)
    head = fl.Head(Xtr.shape[1], c["d_hidden"], 6, seed)
    dp = None
    if dp_eps:
        from opacus.accountants.utils import get_noise_multiplier
        q = cfgb["dp_q"]
        sig = get_noise_multiplier(target_epsilon=dp_eps, target_delta=1e-5, sample_rate=q,
                                   steps=int(round(c["local_epochs"] / q)) * c["T"], accountant="rdp")
        dp = dict(q=q, sigma=sig, clip=cfgb["dp_clip"]); c["lr"] = cfgb["dp_lr"]
    sec = fl.SecureRound(c["K"])
    prior = np.bincount(yva.numpy(), minlength=6) / len(yva)
    truth = np.array([np.bincount(ytr.numpy()[p], minlength=6) / len(p) for p in parts])
    lo, hi = int(0.8 * min(sizes)), int(1.2 * max(sizes))
    out = []
    clip = lambda u: u * min(1.0, c["clip_update"] / float(u.norm()))
    # shadow datasets are drawn ONCE so that the same shadow client can be observed over several rounds
    # (multi-round attack); sizes span the real client-size range
    shadows = []
    for m in range(N_SHADOW):
        p = rng.dirichlet([0.5] * 6); nk = int(rng.integers(lo, hi + 1)); lab = rng.choice(6, nk, p=p)
        idx = np.concatenate([rng.choice(np.where(yva.numpy() == k)[0], (lab == k).sum()) for k in range(6)])
        shadows.append(idx)
    Ps = np.array([np.bincount(yva.numpy()[idx], minlength=6) / len(idx) for idx in shadows])
    ssz = np.array([len(i) for i in shadows], float)
    groups = rng.integers(0, N_SHADOW, (300, c["K"]))
    agg_P = np.stack([(Ps[g] * ssz[g, None]).sum(0) / ssz[g].sum() for g in groups])
    hist = {}                                       # per-round features for the multi-round attack
    for r in range(1, max(ROUNDS_OBS) + 1):
        ups = [clip(fl.local_train(head, Xtr[p], ytr[p], c, g, dp)) for p in parts]
        agg, st = sec.run(ups, [float(s) for s in sizes], r, "verisketch", c["f_assumed"], c["clip_update"])
        if r in ROUNDS_OBS:
            css = {sk: sa.CountSketch(os.urandom(32), head.dim, *sk) for sk in SKETCHES}
            def views(u):
                v = dict(plain_full=u.numpy(), plain_out=last_layer(u, head))
                for sk, cs in css.items():
                    v[f"sketch_{sk[0]}x{sk[1]}"] = cs.to_float(cs(sa.encode(u.numpy(), 1e3))).reshape(-1)
                return v
            Vs = {}
            for idx in shadows:
                u = clip(fl.local_train(head, Xva[idx], yva[idx], c, g, dp))
                for k, v in views(u).items(): Vs.setdefault(k, []).append(v)
            Vs = {k: np.array(v) for k, v in Vs.items()}
            rv = [views(u) for u in ups]; real = {k: np.array([v[k] for v in rv]) for k in Vs}
            rec = dict(seed=seed, eps=dp_eps, round=r, shadow_size_range=[lo, hi], real_sizes=sizes,
                       accepted=(sorted(st.get("accepted", [])) if isinstance(st, dict) else None),
                       prior_mae=float(np.abs(truth - prior).mean()), prior_dom_acc=float(np.mean(truth.argmax(1) == prior.argmax())))
            for k in Vs:
                pr = fit_predict(Vs[k], Ps, real[k])
                rec[f"{k}_mae"] = float(np.abs(truth - pr).mean()); rec[f"{k}_dom_acc"] = float(np.mean(truth.argmax(1) == pr.argmax(1)))
                rec[f"{k}_per_client_mae"] = np.abs(truth - pr).mean(1).tolist()
            # aggregate-only server: regress the federation's class mix from the released (size-weighted)
            # aggregate and attribute it to every client
            W = Vs["plain_out"]; agg_X = np.stack([(W[g] * ssz[g, None]).sum(0) / ssz[g].sum() for g in groups])
            real_agg = last_layer(agg, head)[None] if agg is not None else (real["plain_out"] * np.array(sizes)[:, None]).sum(0, keepdims=True) / sum(sizes)
            pr = np.repeat(fit_predict(agg_X, agg_P, real_agg), c["K"], 0)
            rec["aggregate_mae"] = float(np.abs(truth - pr).mean()); rec["aggregate_dom_acc"] = float(np.mean(truth.argmax(1) == pr.argmax(1)))
            # K-1 colluding clients subtract their own (known) updates from the aggregate: the target's update is
            # isolated exactly, so the attack reduces to the plaintext attack on that client
            rec["collude_k-1_mae"] = rec["plain_full_mae"]; rec["collude_k-1_dom_acc"] = rec["plain_full_dom_acc"]
            for k in ("sketch_4x64", "sketch_4x256", "plain_out"):
                hist.setdefault(k, []).append((Vs[k], real[k]))
            hist.setdefault("aggregate", []).append((agg_X, real_agg))
            out.append(rec); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if (k.endswith("_mae") and "per_client" not in k) or k in ("round", "eps")}, flush=True)
        if agg is not None: head.set_flat(head.flat() + agg)
    # multi-round attack: the same shadow/real clients observed in all observation rounds, features concatenated
    rec = dict(seed=seed, eps=dp_eps, round="multi", prior_mae=float(np.abs(truth - prior).mean()),
               prior_dom_acc=float(np.mean(truth.argmax(1) == prior.argmax())))
    for k, L in hist.items():
        Xs = np.concatenate([a for a, _ in L], 1); Xr = np.concatenate([b for _, b in L], 1)
        if k == "aggregate":
            pr = np.repeat(fit_predict(Xs, agg_P, Xr), c["K"], 0)
        else:
            pr = fit_predict(Xs, Ps, Xr)
        rec[f"{k}_mae"] = float(np.abs(truth - pr).mean()); rec[f"{k}_dom_acc"] = float(np.mean(truth.argmax(1) == pr.argmax(1)))
    out.append(rec); print("multi", {k: round(v, 3) for k, v in rec.items() if k.endswith("_mae")}, flush=True)
    return out


if __name__ == "__main__":
    cfgb = json.load(open(CONFIG_FILE))
    allr = []
    for eps in (None, cfgb.get("eps_star", 2.0)):
        for s in SEEDS[:3]:
            allr += run(s, eps, cfgb)
            save_json(allr, os.path.join(RES, "leak", "leak_results.json"))
