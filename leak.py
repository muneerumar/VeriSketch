"""What the aggregation server can infer about each client's label mix, measured on the
server's ACTUAL protocol view.

For every deployment configuration (sketch size, privacy level) the model follows its own
VeriSketch trajectory.  In the observed rounds the code captures what the AS really holds: the
weighted field sketches under that round's sketch seed, the public weights, the accepted set and
the released aggregate.  The attacker trains shadow clients from the current global model on
public data (the validation split), passes their updates through the same sketch (same seed,
same weighting) and fits a regressor from views to label mixes.

Attacked views (per client):
  sketch       the actual protocol sketch S_sigma(w_k Delta_k) / w_k, plus the public weight;
  aggregate    the released aggregate, the client's weight and its membership in the accepted
               set; shadow federations are filtered with the protocol rule on their sketches under the
               captured sketch map and aggregated with the same quantised weighting; the aggregate
               feature set (output layer or full update) is chosen on held-out shadow federations;
  collusion    the AS with the K-1 other clients: for an accepted target the colluders subtract
               their own contributions from the aggregate and attack the isolated update with the
               plaintext attack; an excluded target is not in the aggregate, so the attack falls back
               to the sketch view.  Coverage = share of accepted targets;
  plaintext    the update a plaintext FedAvg server would see (reference);
  weight_only  the public weight alone; prior = population label mix.
Every choice of the attacker (ridge penalty, plaintext feature set) is made on held-out shadow
clients, never on the targets.  Results are an empirical evaluation of these attacks, not a bound."""
import os, sys, json, numpy as np, torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from common import *
import fl, secagg as sa

N_SHADOW = int(os.environ.get("N_SHADOW", 300))
N_FED = int(os.environ.get("N_FED", 300))
ROUNDS_OBS = tuple(int(x) for x in os.environ.get("LEAK_ROUNDS", "1,10,20").split(","))
CONFIGS = [dict(name="4x64", rows=4, width=64, eps=None), dict(name="4x64_eps2", rows=4, width=64, eps=2.0),
           dict(name="4x256", rows=4, width=256, eps=None)]

_last_cs = {}
class _RecCS(sa.CountSketch):                 # records the round's CountSketch (the AS knows sigma after the coin)
    def __init__(self, *a, **k):
        super().__init__(*a, **k); _last_cs["cs"] = self
sa.CountSketch = _RecCS


def out_layer(U, head):
    n1 = head.shapes[0][0] * head.shapes[0][1] + head.shapes[1][0]
    return U[..., n1:]


class Reg:
    def __init__(self, Xs, Ps):
        self.sc = StandardScaler().fit(Xs)
        self.m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(self.sc.transform(Xs), Ps)
    def __call__(self, X):
        pr = np.clip(self.m.predict(self.sc.transform(X)), 0, None) + 1e-9
        return pr / pr.sum(1, keepdims=True)


def holdout_mae(Xs, Ps, frac=0.2, seed=0):
    r = np.random.default_rng(seed); idx = r.permutation(len(Xs)); k = int(len(Xs) * (1 - frac))
    m = Reg(Xs[idx[:k]], Ps[idx[:k]]); return float(np.abs(m(Xs[idx[k:]]) - Ps[idx[k:]]).mean())


def scores(truth, pr):
    return float(np.abs(truth - pr).mean()), float(np.mean(truth.argmax(1) == pr.argmax(1)))


def run(seed, conf, cfgb):
    from exp import BASE
    c = dict(BASE); c.update(cfgb); c.update(seed=seed, partition=0.5, agg="verisketch")
    d = load_kaggle(cfgb.get("feat", "cls")); Xtr, ytr = d["train"]; Xva, yva = d["val"]
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    K = c["K"]; parts = fl.dirichlet_partition(ytr.numpy(), K, 0.5, rng); sizes = np.array([len(p) for p in parts], float)
    c["class_weights"] = torch.ones(6)
    head = fl.Head(Xtr.shape[1], c["d_hidden"], 6, seed)
    dp = None
    if conf["eps"]:
        from opacus.accountants.utils import get_noise_multiplier
        q = cfgb["dp_q"]
        sig = get_noise_multiplier(target_epsilon=conf["eps"], target_delta=1e-5, sample_rate=q,
                                   steps=int(round(c["local_epochs"] / q)) * c["T"], accountant="rdp")
        dp = dict(q=q, sigma=sig, clip=cfgb["dp_clip"]); c["lr"] = cfgb["dp_lr"]
    sec = fl.SecureRound(K, rows=conf["rows"], width=conf["width"])
    Cu = c["clip_update"]; clip = lambda u: u * min(1.0, Cu / float(u.norm()))
    prior = np.bincount(yva.numpy(), minlength=6) / len(yva)
    truth = np.array([np.bincount(ytr.numpy()[p], minlength=6) / len(p) for p in parts])
    nmax = sizes.max(); w_real = sizes / nmax
    lo, hi = int(0.8 * sizes.min()), int(1.2 * sizes.max())
    # shadow clients are drawn once and observed in every observation round (multi-round attack)
    shadows = []
    for m in range(N_SHADOW):
        p = rng.dirichlet([0.5] * 6); nk = int(rng.integers(lo, hi + 1)); lab = rng.choice(6, nk, p=p)
        shadows.append(np.concatenate([rng.choice(np.where(yva.numpy() == k)[0], (lab == k).sum()) for k in range(6)]))
    Ps = np.array([np.bincount(yva.numpy()[i], minlength=6) / len(i) for i in shadows])
    w_sh = np.array([len(i) for i in shadows], float) / nmax          # the shadow weights use the real public max
    feds = rng.integers(0, N_SHADOW, (N_FED, K))
    out, hist = [], {}
    base = dict(seed=seed, config=conf["name"], eps=conf["eps"], prior_mae=scores(truth, np.repeat(prior[None], K, 0))[0],
                prior_dom_acc=scores(truth, np.repeat(prior[None], K, 0))[1])
    r_w = Reg(w_sh[:, None], Ps)(w_real[:, None]); base["weight_only_mae"], base["weight_only_dom_acc"] = scores(truth, r_w)
    for r in range(1, max(ROUNDS_OBS) + 1):
        ups = [clip(fl.local_train(head, Xtr[p], ytr[p], c, g, dp)) for p in parts]
        agg, st = sec.run(ups, [float(s) for s in sizes], r, "verisketch", c["f_assumed"], Cu, capture=True)
        if r in ROUNDS_OBS and agg is not None:
            cs = _last_cs["cs"]; view = st["view"]; A = view["accepted"]
            real_sk = view["sketch"] / np.array(view["weights"])[:, None]
            # shadow updates from the current global model, passed through the round's sketch
            sh = [clip(fl.local_train(head, Xva[i], yva[i], c, g, dp)) for i in shadows]
            SH = torch.stack(sh).numpy().astype(np.float64)
            B = [sa.int_bound(wi * Cu * (1 + 1e-6)) / sa.SCALE for wi in w_sh]
            sh_sk = np.stack([cs.to_float(cs(sa.encode(w_sh[j] * SH[j], B[j]))).reshape(-1) / w_sh[j] for j in range(N_SHADOW)])
            UP = torch.stack(ups).numpy().astype(np.float64)
            feats_sh = dict(sketch=np.c_[sh_sk, w_sh], plain_full=np.c_[SH, w_sh], plain_out=np.c_[out_layer(SH, head), w_sh])
            feats_re = dict(sketch=np.c_[real_sk, w_real], plain_full=np.c_[UP, w_real], plain_out=np.c_[out_layer(UP, head), w_real])
            rec = dict(base, round=r, accepted=sorted(A), real_sizes=sizes.tolist())
            for k in feats_sh:
                rec[f"{k}_mae"], rec[f"{k}_dom_acc"] = scores(truth, Reg(feats_sh[k], Ps)(feats_re[k]))
                rec[f"{k}_shadow_holdout_mae"] = holdout_mae(feats_sh[k], Ps)
            # aggregate view: shadow federations are filtered with the protocol rule on their sketches under the
            # captured sketch map (same seed, same quantised weighting), and their aggregate is the quantised
            # weighted mean of the accepted shadows.  The aggregate feature set (output layer or full update) is
            # chosen on held-out shadow federations.
            SKs = sh_sk.reshape(N_SHADOW, conf["rows"], conf["width"])
            nrm = np.sqrt(np.median((SKs ** 2).sum(2), 1))
            Qsh = np.stack([sa.to_signed(sa.encode(w_sh[j] * SH[j], B[j])) for j in range(N_SHADOW)]).astype(np.float64) / sa.SCALE
            fe = {"out": [], "full": []}; Pa = []; fid = []
            for q_, fed in enumerate(feds):
                F = SKs[fed]; Dm = np.median(((F[:, None] - F[None]) ** 2).sum(3), 2)
                sel = list(fl.adaptive_select(nrm[fed], Dm, c["f_assumed"], Cu, 1.5, 3.0, fixed=False))
                wf = w_sh[fed]
                ag = Qsh[fed][sel].sum(0) / wf[sel].sum() if len(sel) else np.zeros(SH.shape[1])
                for jj in range(K):
                    tail = [wf[jj], float(jj in sel)]
                    fe["out"].append(np.r_[out_layer(ag, head), tail]); fe["full"].append(np.r_[ag, tail])
                    Pa.append(Ps[fed[jj]]); fid.append(q_)
            Pa, fid = np.array(Pa), np.array(fid)
            hold = np.isin(fid, np.random.default_rng(seed).permutation(N_FED)[: N_FED // 5])
            def fed_holdout(X):
                m = Reg(X[~hold], Pa[~hold]); return float(np.abs(m(X[hold]) - Pa[hold]).mean())
            agg_np = view["U"] / w_real[A].sum()                           # released weighted mean of the accepted set
            Xr_all = {"out": np.array([np.r_[out_layer(agg_np, head), w_real[k], float(k in A)] for k in range(K)]),
                      "full": np.array([np.r_[agg_np, w_real[k], float(k in A)] for k in range(K)])}
            hmae = {k: fed_holdout(np.array(v)) for k, v in fe.items()}
            best_a = min(hmae, key=hmae.get); Xa = np.array(fe[best_a]); Xr = Xr_all[best_a]
            ra = Reg(Xa, Pa); rec["aggregate_mae"], rec["aggregate_dom_acc"] = scores(truth, ra(Xr))
            rec["aggregate_view"] = best_a; rec["aggregate_shadow_holdout_mae"] = hmae
            hist.setdefault("aggregate", []).append((Xa, Pa, Xr))
            # K-1 collusion: subtract the colluders' own contributions for an accepted target
            best_plain = min(("plain_full", "plain_out"), key=lambda k: rec[f"{k}_shadow_holdout_mae"])
            reg_plain, reg_sk = Reg(feats_sh[best_plain], Ps), Reg(feats_sh["sketch"], Ps)
            pr = np.zeros((K, 6))
            for t in range(K):
                if t in A:
                    iso = (view["U"] - sum(w_real[j] * UP[j] for j in A if j != t)) / w_real[t]
                    f = np.r_[iso, w_real[t]] if best_plain == "plain_full" else np.r_[out_layer(iso, head), w_real[t]]
                    pr[t] = reg_plain(f[None])[0]
                    rec.setdefault("collude_isolation_err", []).append(float(np.abs(iso - UP[t]).max()))
                else:
                    pr[t] = reg_sk(feats_re["sketch"][t][None])[0]
            rec["collude_k-1_mae"], rec["collude_k-1_dom_acc"] = scores(truth, pr)
            rec["collude_coverage"] = len(A) / K; rec["collude_view"] = best_plain
            rec["collude_isolation_err"] = max(rec.get("collude_isolation_err", [0.0]))
            for k in ("sketch", "plain_out", "plain_full"): hist.setdefault(k, []).append((feats_sh[k], Ps, feats_re[k]))
            out.append(rec)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()
                   if k.endswith("_mae") and "holdout" not in k or k in ("round", "config", "collude_coverage")}, flush=True)
        if agg is not None: head.set_flat(head.flat() + agg)
    rec = dict(base, round="multi")
    for k, L in hist.items():
        if len(L) < 2: continue
        Xs = np.concatenate([a for a, _, _ in L], 1); Xr = np.concatenate([b for _, _, b in L], 1)
        rec[f"{k}_mae"], rec[f"{k}_dom_acc"] = scores(truth, Reg(Xs, L[0][1])(Xr))
    out.append(rec); print("multi", {k: round(v, 3) for k, v in rec.items() if k.endswith("_mae")}, flush=True)
    return out


if __name__ == "__main__":
    cfgb = json.load(open(CONFIG_FILE))
    shard, nsh = (map(int, sys.argv[1].split("/")) if len(sys.argv) > 1 else (0, 1))
    path = os.path.join(RES, "leak", os.environ.get("LEAK_OUT", f"leak_part{shard}of{nsh}.json"))
    allr = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["config"], r["seed"]) for r in allr}
    seeds = SEEDS[: int(os.environ.get("LEAK_SEEDS", 5))]
    jobs = [(conf, s) for conf in CONFIGS for s in seeds][shard::nsh]
    for conf, s in jobs:
        if (conf["name"], s) in done: continue
        allr += run(s, conf, cfgb)
        save_json(allr, path)
    import glob as _g                                   # merge the parts once all jobs of all shards are done
    parts = sorted(_g.glob(os.path.join(RES, "leak", f"leak_part*of{nsh}.json")))
    merged = sum((json.load(open(p)) for p in parts), [])
    if len({(r["config"], r["seed"]) for r in merged}) == len(CONFIGS) * len(seeds):
        save_json(merged, os.path.join(RES, "leak", "leak_results.json"))
        from exp import provenance_side
        save_json(provenance_side(["leak.py"]), os.path.join(RES, "leak", "leak_provenance.json"))
