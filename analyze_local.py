"""Builds every analysis table from the raw result files of locally regenerated runs.

The experimental unit is the training seed.  Methods are compared paired by seed (same
partition, same attackers, same data order); the mean seed-level difference is reported with
a t-based 95% interval; families of comparisons are Holm-corrected; "matches FedAvg" is a
TOST equivalence test with a margin of +-0.01 macro-F1.  Attacks are never pooled."""
import os, json, glob
import numpy as np, pandas as pd
from scipy import stats
from common import RES, SEEDS

OUT = os.path.join(RES, "analysis")
MARGIN = 0.01
AGG_LABEL = {"fedavg": "FedAvg (plaintext)", "secagg": "SecAgg-FedAvg", "verisketch": "VeriSketch",
             "verisketch_mk": "VeriSketch-MK", "krum": "Krum (plaintext)", "mkrum": "Multi-Krum (plaintext)",
             "median": "Coord. median (plaintext)", "trimmed": "Trimmed mean (plaintext)",
             "adaptive_plain": "Adaptive filter (plaintext)", "normbound": "Norm bound (ELSA-style)", "central": "Centralised"}
ATTACKS = ["none", "flip_evasion", "flip_rotate", "signflip", "signflip_nm", "gauss", "alie", "ipm", "adaptive"]
ATK_LABEL = {"none": "No attack", "flip_evasion": "Label flip (evasion)", "flip_rotate": "Label flip (rotation)",
             "signflip": "Sign flip (×4)", "signflip_nm": "Sign flip (norm-matched)", "gauss": "Gaussian (10×)",
             "alie": "ALIE", "ipm": "IPM (0.5)", "adaptive": "Filter-aware adaptive", "nullspace": "Null-space"}


# ------------------------------------------------------------------------- loading
_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "split_manifest.csv"))   # same row order as kaggle_clean_v3.csv
Y_TEST = _df[_df.split == "test"].label.values
CLASSES = ["not_cb", "age", "ethnicity", "gender", "religion", "other_cb"]


def load_group(g):
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, g, "*.json"))):
        o = json.load(open(p)); c = o["cfg"]
        r = dict(group=g, name=os.path.basename(p)[:-5], method=c["agg"] if c["K"] > 1 else "central",
                 partition=str(c["partition"]), seed=c["seed"], attack=c.get("attack", "none"),
                 n_mal=c.get("n_malicious", 0), f_assumed=c.get("f_assumed"),
                 eps=((c["dp"]["epsilon"] if c["dp"].get("epsilon") is not None else 0.0) if c.get("dp") else np.inf),
                 **{k: o["test"][k] for k in ("accuracy", "macro_f1", "tmr", "far")})
        if "mia" in o: r.update({"mia_" + k: v for k, v in o["mia"].items()})
        h = o["history"]; mal = set(o.get("malicious", []))
        r["mal_incl_rate"] = float(np.mean([len(x["mal_selected"]) / len(mal) for x in h])) if mal else np.nan
        r["honest_excl_rate"] = float(np.mean([1 - len([i for i in x["selected"] if i not in mal]) / (c["K"] - len(mal))
                                               for x in h])) if c["K"] > 1 else 0.0
        sz = np.array(o.get("partition_sizes", [1] * c["K"]), float)
        r["accepted_size"] = float(np.mean([len(x["selected"]) for x in h]))
        r["mal_weight_frac"] = float(np.mean([sz[[i for i in x["selected"] if i in mal]].sum() / max(sz[x["selected"]].sum(), 1)
                                              for x in h])) if mal else np.nan
        if o.get("test_pred") is not None and len(o["test_pred"]) == len(Y_TEST):
            pr = np.asarray(o["test_pred"])
            for ci, cn in enumerate(CLASSES): r["recall_" + cn] = float((pr[Y_TEST == ci] == ci).mean())
        r["t_round"] = float(np.mean([x["t_round"] for x in h])); r["wall"] = o["wall"]
        r["n_aborted"] = o.get("n_aborted", 0)
        r["aborts_filter"] = sum(1 for x in h if x.get("aborted") and x.get("abort_kind") == "filter")
        r["aborts_protocol"] = sum(1 for x in h if x.get("aborted") and x.get("abort_kind") != "filter")
        r["abort_reasons"] = "; ".join(sorted({x.get("reason", "") for x in h if x.get("aborted")}))
        r["verified_all"] = all(x.get("verified", True) for x in h)
        r["sigma"] = o.get("sigma"); r["sigma_mode"] = c.get("sigma_mode")
        hv = [x for x in h if "val_macro_f1" in x]
        if hv: r.update(val_f1=hv[-1]["val_macro_f1"], val_tmr=hv[-1]["val_tmr"], val_far=hv[-1]["val_far"])
        for k in ("rows", "width"): r[k] = c.get(k)
        rows.append(r)
    return pd.DataFrame(rows)


_cache = {}
def G(g):
    if g not in _cache:
        df = load_group(g) if os.path.isdir(os.path.join(RES, g)) else pd.DataFrame()
        if not len(df):
            df = pd.DataFrame(columns=["name", "method", "partition", "seed", "attack", "n_mal", "eps", "macro_f1", "tmr", "far",
                                       "accuracy", "mal_incl_rate", "honest_excl_rate", "n_aborted", "sigma", "sigma_mode",
                                       "rows", "width", "val_f1", "val_tmr", "val_far", "mia_auc", "mia_tpr_at_1fpr"])
        _cache[g] = df
    return _cache[g]


# ------------------------------------------------------------------------- statistics
def ms(x, d=4, pct=False):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return "–"
    m, s = x.mean(), (x.std(ddof=1) if len(x) > 1 else 0.0)
    return f"{100*m:.2f} ± {100*s:.2f}" if pct else f"{m:.{d}f} ± {s:.{d}f}"


def paired_diff(a, b):
    """a, b: Series indexed by seed.  Returns mean diff (a-b), 95% CI, paired-t p, n."""
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna(); d = (j.a - j.b).values; n = len(d)
    if n < 2: return dict(diff=float(d.mean()) if n else np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=n)
    se = d.std(ddof=1) / np.sqrt(n); tq = stats.t.ppf(0.975, n - 1)
    p = stats.ttest_1samp(d, 0).pvalue if se > 0 else (0.0 if d.mean() != 0 else 1.0)
    return dict(diff=float(d.mean()), lo=float(d.mean() - tq * se), hi=float(d.mean() + tq * se), p=float(p), n=n)


def tost(a, b, margin=MARGIN):
    """Paired TOST for |mean(a-b)| < margin; p = max of the two one-sided p-values."""
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna(); d = (j.a - j.b).values; n = len(d)
    if n < 2: return np.nan
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0: return 0.0 if abs(d.mean()) < margin else 1.0
    p1 = 1 - stats.t.cdf((d.mean() + margin) / se, n - 1); p2 = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return float(max(p1, p2))


def holm(ps):
    ps = np.asarray(ps, float); m = np.sum(~np.isnan(ps)); order = np.argsort(np.where(np.isnan(ps), np.inf, ps))
    adj = np.full(len(ps), np.nan); run = 0.0
    for rank, i in enumerate(order):
        if np.isnan(ps[i]): continue
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def by_seed(df, metric="macro_f1", **filt):
    s = df
    for k, v in filt.items(): s = s[s[k] == v]
    return s.groupby("seed")[metric].mean()


# ------------------------------------------------------------------------- clean runs
CLEAN_AGGS = ["fedavg", "secagg", "krum", "mkrum", "median", "trimmed", "adaptive_plain", "verisketch", "verisketch_mk"]


def t_baseline():
    df = G("baseline"); rows = []
    c = df[df.method == "central"]
    rows.append(dict(partition="central", method=AGG_LABEL["central"], n=len(c), accuracy=ms(c.accuracy, pct=True),
                     macro_f1=ms(c.macro_f1), tmr=ms(c.tmr, pct=True), far=ms(c.far, pct=True)))
    eq = []
    for part in ("iid", "0.5", "0.1"):
        fa = by_seed(df, method="fedavg", partition=part)
        for agg in CLEAN_AGGS:
            sub = df[(df.method == agg) & (df.partition == part)]
            if not len(sub): continue
            r = dict(partition=part, method=AGG_LABEL[agg], n=len(sub), accuracy=ms(sub.accuracy, pct=True),
                     macro_f1=ms(sub.macro_f1), tmr=ms(sub.tmr, pct=True), far=ms(sub.far, pct=True),
                     honest_excl=f"{100*sub.honest_excl_rate.mean():.1f}", aborts=int(sub.n_aborted.sum()))
            if agg != "fedavg":
                pdf = paired_diff(by_seed(df, method=agg, partition=part), fa)
                r.update(d_vs_fedavg=f"{pdf['diff']:+.4f} [{pdf['lo']:+.4f}, {pdf['hi']:+.4f}]", p_tost=tost(by_seed(df, method=agg, partition=part), fa))
                eq.append(dict(partition=part, method=agg, **pdf, p_tost=r["p_tost"]))
            rows.append(r)
    eq = pd.DataFrame(eq)
    if len(eq): eq["p_tost_holm"] = holm(eq.p_tost.values); eq["equivalent"] = eq.p_tost_holm < 0.05
    return pd.DataFrame(rows), eq


# ------------------------------------------------------------------------- poisoning
def cell(aggs_df_robust, base, method, part, atk, metric="macro_f1"):
    if atk == "none": return base[(base.method == method) & (base.partition == part)]
    return aggs_df_robust[(aggs_df_robust.method == method) & (aggs_df_robust.partition == part) & (aggs_df_robust.attack == atk)]


ROB_AGGS = ["fedavg", "normbound", "mkrum", "median", "trimmed", "adaptive_plain", "verisketch", "verisketch_mk"]


def robust_frames():
    """Robust grid plus the ELSA-style norm-bound runs (its clean runs are added to the baseline frame)."""
    rob, base, el = G("robust"), G("baseline"), G("elsa")
    if len(el):
        rob = pd.concat([rob, el[el.attack != "none"]], ignore_index=True)
        base = pd.concat([base, el[el.attack == "none"]], ignore_index=True)
    return rob, base


def t_robust(metric="macro_f1"):
    rob, base = robust_frames(); rows = []
    for part in ("iid", "0.5"):
        for atk in ATTACKS:
            r = dict(partition=part, attack=ATK_LABEL[atk])
            for agg in ROB_AGGS:
                sub = cell(rob, base, agg, part, atk)
                r[agg] = sub[metric].mean() if len(sub) else np.nan
                r[agg + "_sd"] = sub[metric].std(ddof=1) if len(sub) > 1 else np.nan
                r[agg + "_n"] = len(sub)
                r[agg + "_malincl"] = sub.mal_incl_rate.mean() if len(sub) and atk != "none" else np.nan
                r[agg + "_tmr"] = sub.tmr.mean() if len(sub) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


EFFECT_REFS = ["fedavg", "mkrum", "median", "trimmed", "adaptive_plain", "normbound"]


def t_attack_effects():
    """Attack-specific paired effects of VeriSketch against a pre-specified set of references: plaintext FedAvg,
    each plaintext robust rule (Multi-Krum, median, trimmed mean), the plaintext version of the same adaptive
    filter (cost of sketching) and the aggregate-only norm bound.  Holm correction within each reference family
    (18 settings).  No reference is chosen by its test result."""
    rob, base = robust_frames(); rows = []
    for part in ("iid", "0.5"):
        for atk in ATTACKS:
            get = lambda m: by_seed(cell(rob, base, m, part, atk))
            vs = get("verisketch")
            if not len(vs): continue
            for ref in EFFECT_REFS:
                if not len(get(ref)): continue
                rows.append(dict(partition=part, attack=atk, ref=ref, **paired_diff(vs, get(ref))))
    out = pd.DataFrame(rows)
    if len(out):
        for ref in EFFECT_REFS:
            m = out.ref == ref
            if m.any(): out.loc[m, "p_holm"] = holm(out.loc[m, "p"].values)
    return out


def t_ablation_summary():
    """Sketch cost: VeriSketch vs the exact-distance plaintext version of the same rule, over all cells."""
    e = t_attack_effects()
    if not len(e): return {}
    a = e[e.ref == "adaptive_plain"]
    return dict(n_cells=len(a), median_diff=float(a["diff"].median()), min_diff=float(a["diff"].min()),
                max_diff=float(a["diff"].max()), n_sig_holm=int((a.p_holm < 0.05).sum()),
                n_equiv=int(sum(1 for _, r in a.iterrows() if r.lo > -MARGIN and r.hi < MARGIN)))


JOINT_AGGS = ["secagg", "mkrum", "adaptive_plain", "verisketch", "verisketch_mk"]


DP_LEVELS = {"none": "no DP", "clip": "DP-SGD, σ = 0", "eps8": "ε = 8", "eps2": "ε = 2"}


def joint_cell(dp, agg, part, atk):
    """Runs of one joint-grid cell.  'none' = the non-private configuration (lr 1e-3, minibatch SGD);
    'clip' = the DP-SGD pipeline (Poisson sampling, per-example clipping, lr 3e-3) with sigma = 0;
    'eps8', 'eps2' = the same pipeline with noise."""
    j, rob, base, ctl = G("joint"), G("robust"), G("baseline"), G("dpctl")
    if dp == "eps2": return j[(j.eps == 2.0) & (j.method == agg) & (j.partition == part) & (j.attack == atk)]
    if dp in ("clip", "eps8"):
        e = 0.0 if dp == "clip" else 8.0
        return ctl[(ctl.eps == e) & (ctl.method == agg) & (ctl.partition == part) & (ctl.attack == atk)] if len(ctl) else ctl
    if atk == "none": return base[(base.method == agg) & (base.partition == part)]
    if agg == "secagg": return j[(j.eps == np.inf) & (j.method == "secagg") & (j.partition == part) & (j.attack == atk)]
    return rob[(rob.method == agg) & (rob.partition == part) & (rob.attack == atk)]


def t_joint():
    """{no DP, DP-SGD sigma=0, eps=8, eps=2} x {IID, Dir(0.5)} x {none, evasion flip, nm sign flip, adaptive} x 5 methods x 5 seeds."""
    rows = []; LONG = []; os.makedirs(OUT, exist_ok=True)
    for dp in DP_LEVELS:
        for part in ("iid", "0.5"):
            for atk in ("none", "flip_evasion", "signflip_nm", "adaptive"):
                r = dict(dp=DP_LEVELS[dp], dp_key=dp, partition=part, attack=ATK_LABEL[atk])
                for agg in JOINT_AGGS:
                    sub = joint_cell(dp, agg, part, atk)
                    r[agg] = sub.macro_f1.mean() if len(sub) else np.nan
                    r[agg + "_sd"] = sub.macro_f1.std(ddof=1) if len(sub) > 1 else np.nan
                    r[agg + "_tmr"] = sub.tmr.mean() if len(sub) else np.nan
                    r[agg + "_n"] = len(sub)
                    r[agg + "_malincl"] = sub.mal_incl_rate.mean() if len(sub) and atk != "none" else np.nan
                    if len(sub):
                        LONG.append(dict(dp=r["dp"], partition=part, attack=atk, method=agg, n=len(sub),
                                         macro_f1=sub.macro_f1.mean(), macro_f1_sd=sub.macro_f1.std(ddof=1), tmr=sub.tmr.mean(), far=sub.far.mean(),
                                         **{c: sub[c].mean() for c in sub.columns if c.startswith("recall_")},
                                         mal_incl=sub.mal_incl_rate.mean(), honest_excl=sub.honest_excl_rate.mean(),
                                         accepted_size=sub.accepted_size.mean(), mal_weight_frac=sub.mal_weight_frac.mean(),
                                         aborted_rounds=int(sub.n_aborted.sum()), all_verified=bool(sub.verified_all.all())))
                rows.append(r)
    pd.DataFrame(LONG).to_csv(os.path.join(OUT, "joint_long.csv"), index=False)
    return pd.DataFrame(rows)


def t_joint_effects():
    """Exploratory paired effects of each filter against SecAgg-FedAvg in every joint cell.
    Family (fixed before looking at these effects): per privacy level, 4 filters x 4 attacks x 2 partitions = 32
    comparisons, Holm-corrected within the family.  These are not pre-registered superiority tests."""
    rows = []
    for dp in DP_LEVELS:
        for part in ("iid", "0.5"):
            for atk in ("none", "flip_evasion", "signflip_nm", "adaptive"):
                ref = by_seed(joint_cell(dp, "secagg", part, atk))
                for agg in ("mkrum", "adaptive_plain", "verisketch", "verisketch_mk"):
                    a = by_seed(joint_cell(dp, agg, part, atk))
                    if len(a) and len(ref): rows.append(dict(dp=dp, partition=part, attack=atk, method=agg, **paired_diff(a, ref)))
    out = pd.DataFrame(rows)
    for dp in DP_LEVELS:
        m = out.dp == dp
        if m.any(): out.loc[m, "p_holm"] = holm(out.loc[m, "p"].values); out.loc[m, "family_size"] = int(m.sum())
    return out


def t_aborts():
    """Every aborted round in every group, split into filter-induced aborts (accepted set below the release
    threshold) and protocol-check aborts (receipt, coin, sketch self-check, verification, range, consistency)."""
    rows = []
    for g in ("baseline", "robust", "joint", "dpctl", "dp", "elsa", "fsweep", "sketchdim", "nullspace"):
        df = G(g)
        if not len(df): continue
        for _, r in df[df.n_aborted > 0].iterrows():
            rows.append(dict(group=g, run=r["name"], aborted=int(r.n_aborted), filter=int(r.aborts_filter),
                             protocol=int(r.aborts_protocol), reasons=r.abort_reasons))
    return pd.DataFrame(rows, columns=["group", "run", "aborted", "filter", "protocol", "reasons"])


def t_fsweep():
    fs, rob, base = G("fsweep"), G("robust"), G("baseline"); rows = []
    if not len(fs): return pd.DataFrame()
    for atk in ("alie", "flip_evasion", "signflip", "adaptive"):
        for agg in ("fedavg", "mkrum", "median", "verisketch", "verisketch_mk"):
            for f in (0, 1, 2, 3, 4):
                if f == 0: sub = base[(base.method == agg) & (base.partition == "iid") & base.seed.isin(SEEDS[:3])]
                elif f == 3: sub = rob[(rob.method == agg) & (rob.partition == "iid") & (rob.attack == atk) & rob.seed.isin(SEEDS[:3])]
                else: sub = fs[(fs.method == agg) & (fs.attack == atk) & (fs.n_mal == f)]
                if len(sub): rows.append(dict(attack=atk, agg=agg, f=f, f1=sub.macro_f1.mean(), sd=sub.macro_f1.std(ddof=1),
                                              tmr=sub.tmr.mean(), malincl=sub.mal_incl_rate.mean(), n=len(sub)))
    return pd.DataFrame(rows)


def t_sketchdim():
    df = G("sketchdim")
    if not len(df): return pd.DataFrame()
    df = df.copy(); df["rw"] = df.rows.astype(int).astype(str) + "×" + df.width.astype(int).astype(str)
    g = df.groupby(["rw", "attack"]).agg(f1=("macro_f1", "mean"), sd=("macro_f1", "std"), n=("macro_f1", "size"),
                                          tmr=("tmr", "mean"), honest_excl=("honest_excl_rate", "mean"),
                                          malincl=("mal_incl_rate", "mean")).reset_index()
    return g


def t_nullspace():
    df = G("nullspace")
    if not len(df): return pd.DataFrame()
    df = df.copy(); df["variant"] = df.apply(lambda r: f"verisketch ({r.sigma_mode} seed)" if r.method == "verisketch" else AGG_LABEL[r.method], axis=1)
    return df.groupby(["variant", "n_mal"]).agg(f1=("macro_f1", "mean"), sd=("macro_f1", "std"), tmr=("tmr", "mean"),
                                                 malincl=("mal_incl_rate", "mean"), n=("macro_f1", "size")).reset_index()


# ------------------------------------------------------------------------- DP
def t_dp():
    df, base = G("dp"), G("baseline"); rows = []
    for part in ("iid", "0.5"):
        for agg in ("secagg", "verisketch"):
            for eps in (0.5, 1.0, 2.0, 4.0, 8.0, np.inf):
                sub = base[(base.method == agg) & (base.partition == part)] if np.isinf(eps) else \
                    df[(df.method == agg) & (df.partition == part) & (df.eps == eps)] if len(df) else pd.DataFrame()
                if not len(sub): continue
                rows.append(dict(partition=part, agg=agg, eps=eps, n=len(sub), acc=sub.accuracy.mean(),
                                 f1=sub.macro_f1.mean(), f1_sd=sub.macro_f1.std(ddof=1), tmr=sub.tmr.mean(), tmr_sd=sub.tmr.std(ddof=1),
                                 far=sub.far.mean(), far_sd=sub.far.std(ddof=1), mia_auc=sub.mia_auc.mean(), mia_auc_sd=sub.mia_auc.std(ddof=1),
                                 mia_tpr1=sub.mia_tpr_at_1fpr.mean(),
                                 sigma=float(pd.to_numeric(sub.sigma, errors="coerce").mean()) if not np.isinf(eps) else np.nan))
    return pd.DataFrame(rows)


def eps_star(part="iid", agg="secagg"):
    """Smallest epsilon whose VALIDATION TMR is within +5 points and FAR within +10 points of the
    non-private validation values (rule fixed before the test set was evaluated)."""
    df, base = G("dp"), G("baseline")
    if not len(df): return None, pd.DataFrame(), None
    b = base[(base.method == agg) & (base.partition == part)][["val_tmr", "val_far", "val_f1"]].mean()
    rows = []
    for eps in (0.5, 1.0, 2.0, 4.0, 8.0):
        v = df[(df.method == agg) & (df.partition == part) & (df.eps == eps)][["val_tmr", "val_far", "val_f1"]].mean()
        rows.append(dict(eps=eps, val_tmr=v.val_tmr, val_far=v.val_far, val_f1=v.val_f1,
                         ok=bool(v.val_tmr <= b.val_tmr + 0.05 and v.val_far <= b.val_far + 0.10)))
    r = pd.DataFrame(rows); feas = r[r.ok]
    return (float(feas.eps.min()) if len(feas) else None), r, b


# ------------------------------------------------------------------------- side studies
def rj(*p):
    f = os.path.join(RES, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def t_fidelity():
    x = rj("fidelity", "sketch_fidelity.json")
    if not x: return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(x)
    a = df.groupby("sketch").agg(dist_err=("dist_rel_err_median", "median"), norm_err=("norm_rel_err_median", "median"),
                                 disagree=("decision_disagreement", "mean"), mal_exact=("mal_accepted_exact", "mean"),
                                 mal_sketch=("mal_accepted_sketch", "mean"), n=("seed", "size")).reset_index()
    b = df[df.sketch == "4x64"].groupby(["attack", "partition"]).agg(dist_err=("dist_rel_err_median", "median"),
                                                                      disagree=("decision_disagreement", "mean"),
                                                                      mal_exact=("mal_accepted_exact", "mean"),
                                                                      mal_sketch=("mal_accepted_sketch", "mean")).reset_index()
    return a, b


LEAK_VIEWS = ["prior", "weight_only", "aggregate", "sketch", "collude_k-1", "plain_out", "plain_full"]


def t_leak():
    """Label-mix inference on the AS's captured view.  The unit is the federation (one seed): per-client errors
    are averaged within a federation, and the table gives the mean over federations with a t-based 95% interval."""
    x = rj("leak", "leak_results.json")
    if not x: return pd.DataFrame(), pd.DataFrame()
    r = pd.DataFrame(x); rows = []
    for (cfg, rnd), g in r.groupby(["config", r["round"].astype(str)]):
        rr = dict(config=cfg, round=rnd, n_federations=g.seed.nunique())
        for v in LEAK_VIEWS:
            if f"{v}_mae" not in g or g[f"{v}_mae"].isna().all(): continue
            m = g[f"{v}_mae"].dropna().values; n = len(m)
            half = stats.t.ppf(0.975, n - 1) * m.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            rr[f"{v}_mae"] = m.mean(); rr[f"{v}_lo"] = m.mean() - half; rr[f"{v}_hi"] = m.mean() + half
            rr[f"{v}_dom"] = g[f"{v}_dom_acc"].mean()
        if "collude_coverage" in g: rr["collude_coverage"] = g["collude_coverage"].mean()
        rows.append(rr)
    return pd.DataFrame(rows), r


def t_moderation():
    x = rj("moderation", "moderation.json")
    if not x: return pd.DataFrame(), pd.DataFrame()
    rows, summ = [], []
    for rec in x:
        for p in rec["points"]:
            rows.append(dict(seed=rec["seed"], agg=rec["agg"], eps="∞" if rec["eps"] is None else f"{rec['eps']:g}", **p))
        summ.append(dict(seed=rec["seed"], agg=rec["agg"], eps="∞" if rec["eps"] is None else f"{rec['eps']:g}",
                         roc_auc=rec["roc_auc"], ap=rec["ap"], ece_harm=rec["ece_harm"], ece_top=rec["ece_top"]))
    d = pd.DataFrame(rows); s = pd.DataFrame(summ)
    num = [c for c in d.columns if c not in ("seed", "agg", "eps", "rule")]
    pts = d.groupby(["agg", "eps", "rule"])[num].agg(["mean", "std"]).reset_index()
    pts.columns = ["_".join(c).strip("_") for c in pts.columns]
    sm = s.groupby(["agg", "eps"])[["roc_auc", "ap", "ece_harm", "ece_top"]].agg(["mean", "std"]).reset_index()
    sm.columns = ["_".join(c).strip("_") for c in sm.columns]
    return pts, sm


def _xai_frames():
    fr = {}
    for eps in ("inf", "8.0", "2.0", "0.5"):
        for s in SEEDS[:3]:
            p = os.path.join(RES, "xai", f"eps{eps}_s{s}_xai.csv")
            if os.path.exists(p): fr[(eps, s)] = pd.read_csv(p)
    return fr


def t_xai():
    """XAI analysis.  For each eps against eps=inf: the mean difference, a two-stage bootstrap interval over
    trained models and posts, and a post-level Wilcoxon test that is conditional on the trained models
    (Holm over eps x metric within a subset).  Common-correct posts are those that ALL three non-private
    heads and ALL three private heads classify correctly (the mean of the binary correctness indicators is 1
    at both levels); they form a sensitivity analysis on a prediction-selected subset."""
    fr = _xai_frames()
    if not fr: return {}
    METRICS = ["pred_ig_tf1_oracle", "pred_ig_tf1_budget", "pred_ig_auprc", "comprehensiveness", "sufficiency"]
    BASEL = ["pred_gxi_tf1_budget", "pred_random_tf1_budget", "pred_gxi_auprc", "pred_random_auprc"]
    levels = [e for e in ("inf", "8.0", "2.0", "0.5") if any(k[0] == e for k in fr)]
    out = dict(levels=[], seed_means=[], tests=[], baselines=[], residual=[], common_correct=[])
    def stack(eps):
        L = [fr[(eps, s)].assign(seed=s) for s in SEEDS[:3] if (eps, s) in fr]
        return pd.concat(L) if L else None
    ref = stack("inf")
    for eps in levels:
        x = stack(eps)
        if x is None: continue
        for s, g in x.groupby("seed"):
            out["seed_means"].append(dict(eps=eps, seed=int(s), acc=g.correct.mean(), **{m: g[m].mean() for m in METRICS + BASEL}))
        out["residual"].append(dict(eps=eps, median_abs_residual=float(x.ig_residual.abs().median()),
                                    median_rel_residual=float((x.ig_residual.abs() / (x.ig_delta.abs() + 1e-9)).median())))
        for b in ("ig", "gxi", "random"):
            out["baselines"].append(dict(eps=eps, attribution=b, tf1_budget=x[f"pred_{b}_tf1_budget"].mean(),
                                         tf1_oracle=x[f"pred_{b}_tf1_oracle"].mean(), auprc=x[f"pred_{b}_auprc"].mean()))
        if eps != "inf" and ref is not None:
            # (1) post-level paired Wilcoxon test on model-averaged values: CONDITIONAL on the trained models
            # (2) two-stage bootstrap over models (within each level) and posts (paired): interval that also
            #     reflects training-model variation; with 3 models per level the interval is approximate
            # (3) common-correct posts: a sensitivity analysis on a prediction-selected subset, not a decomposition
            a = ref.groupby("post_id")[METRICS + ["correct"]].mean(); b2 = x.groupby("post_id")[METRICS + ["correct"]].mean()
            j = a.join(b2, lsuffix="_ref", rsuffix="_dp", how="inner")
            cc_ids = set(j[(j.correct_ref == 1) & (j.correct_dp == 1)].index)
            posts = list(j.index); pidx = {p: i for i, p in enumerate(posts)}
            def mat(df, m):     # models x posts
                return np.stack([g.set_index("post_id").reindex(posts)[m].values for _, g in df.groupby("seed")])
            for subset in ("all", "common-correct"):
                keep = np.array([p in cc_ids for p in posts]) if subset == "common-correct" else np.ones(len(posts), bool)
                J = j[keep]
                for m in METRICS:
                    d = (J[m + "_dp"] - J[m + "_ref"]).dropna()
                    p = stats.wilcoxon(d).pvalue if len(d) > 10 and (d != 0).any() else np.nan
                    R, D = mat(ref, m)[:, keep], mat(x, m)[:, keep]
                    rng = np.random.default_rng(12345); boots = []
                    for _ in range(2000):
                        pi = rng.integers(0, R.shape[1], R.shape[1])
                        mr = R[rng.integers(0, R.shape[0], R.shape[0])][:, pi]; md = D[rng.integers(0, D.shape[0], D.shape[0])][:, pi]
                        boots.append(np.nanmean(md) - np.nanmean(mr))
                    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                    out["tests"].append(dict(eps=eps, subset=subset, metric=m, n_posts=len(d), ref=float(J[m + "_ref"].mean()),
                                             dp=float(J[m + "_dp"].mean()), diff=float(d.mean()), boot_lo=float(lo), boot_hi=float(hi),
                                             p=float(p)))
    t = pd.DataFrame(out["tests"])
    if len(t):
        for subset in t.subset.unique():
            msk = t.subset == subset; t.loc[msk, "p_holm"] = holm(t.loc[msk, "p"].values)
    out["tests"] = t
    for k in ("seed_means", "baselines", "residual"): out[k] = pd.DataFrame(out[k])
    sm = out["seed_means"]
    if len(sm): out["levels"] = sm.groupby("eps").agg(["mean", "std"]).drop(columns="seed", level=0)
    out["step"] = {}
    for tag in ("epsinf_s42", "eps2.0_s42"):
        st = os.path.join(RES, "xai", f"ig_step_stability_{tag}.csv")
        if os.path.exists(st): out["step"][tag] = pd.read_csv(st).median(numeric_only=True).to_dict()
    out["meta"] = rj("xai", "xai_meta.json")
    cls = []
    for p in glob.glob(os.path.join(RES, "xai", "eps*_cls.json")):
        o = json.load(open(p)); tag = os.path.basename(p).split("_")
        cls.append(dict(eps=tag[0][3:], seed=int(tag[1][1:]), acc=o["test"]["accuracy"], macro_f1=o["test"]["macro_f1"]))
    out["cls"] = pd.DataFrame(cls)
    gold = []
    for (eps, s), x in fr.items():
        if "gold_ig_tf1_budget" in x:
            gold.append(dict(eps=eps, seed=s, gold_tf1_budget=x.gold_ig_tf1_budget.mean(), pred_tf1_budget=x.pred_ig_tf1_budget.mean()))
    out["gold"] = pd.DataFrame(gold)
    return out


def t_tamper():
    x = rj("tamper", "tamper_results.json")
    if not x: return pd.DataFrame()
    return pd.DataFrame([dict(scenario=k, trials=v["trials"], aborted=v["aborted"], rate=v["rate"],
                              reason="; ".join(f"{a} ({b})" for a, b in v.get("reasons", {}).items()))
                         for k, v in x.items() if isinstance(v, dict) and "trials" in v and "aborted" in v])


def t_bench():
    return rj("bench", "bench.json")


# ------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    tb, eq = t_baseline(); tb.to_csv(os.path.join(OUT, "baseline.csv"), index=False); eq.to_csv(os.path.join(OUT, "equivalence.csv"), index=False)
    print(tb.to_string()); print(eq.round(4).to_string())
    for name, fn in (("robust", t_robust), ("attack_effects", t_attack_effects), ("joint", t_joint), ("joint_effects", t_joint_effects),
                     ("aborts", t_aborts), ("fsweep", t_fsweep),
                     ("sketchdim", t_sketchdim), ("nullspace", t_nullspace), ("dp", t_dp), ("tamper", t_tamper)):
        t = fn()
        if len(t): t.to_csv(os.path.join(OUT, name + ".csv"), index=False); print("\n==", name); print(t.round(4).to_string()[:6000])
    print("\n== ablation", t_ablation_summary())
    e, r, b = eps_star(); print("\n== eps*", e); print(r)
    a, b = t_fidelity(); print("\n== fidelity"); print(a.round(4).to_string()); print(b.round(4).to_string())
    l, _ = t_leak(); print("\n== leak"); print(l.round(4).to_string())
    p, s = t_moderation(); print("\n== moderation"); print(s.round(4).to_string())
    x = t_xai()
    if x:
        print("\n== xai"); print(x["tests"].round(4).to_string()); print(x["baselines"].round(4).to_string())
        x["tests"].to_csv(os.path.join(OUT, "xai_tests.csv"), index=False)        # aggregate statistics only
        x["baselines"].to_csv(os.path.join(OUT, "xai_baselines.csv"), index=False)


if __name__ == "__main__":
    main()


def t_neardup(th=0.9):
    """Sensitivity of clean macro-F1 to removing near-duplicate test posts."""
    from sklearn.metrics import f1_score
    f = os.path.join(RES, "near_dup_test_maxcos.npy")
    if not os.path.exists(f): return pd.DataFrame()
    keep = np.load(f) < th; rows = []
    for p in sorted(glob.glob(os.path.join(RES, "baseline", "*_s*.json"))):
        nm = os.path.basename(p)[:-5]
        if not (nm.startswith("fedavg_") or nm.startswith("verisketch_") or nm.startswith("secagg_")) or "_mk" in nm: continue
        o = json.load(open(p)); pr = np.asarray(o["test_pred"])
        rows.append(dict(name=nm, method=o["cfg"]["agg"], partition=str(o["cfg"]["partition"]), seed=o["cfg"]["seed"],
                         f1_all=f1_score(Y_TEST, pr, average="macro"), f1_dedup=f1_score(Y_TEST[keep], pr[keep], average="macro")))
    d = pd.DataFrame(rows); d["delta"] = d.f1_dedup - d.f1_all
    return d
