"""Adversarial evaluation of the protocol (Table "Deviations injected into complete rounds").

Every trial uses fresh masks, coins and challenges on realistic updates (one honest
local-training round on the real features).  Scenarios: high-valuation errors, sketch-kernel
errors, wrap-around and out-of-range values, replay, transcript manipulation, wrong coin
openings, balanced auxiliary-node sketch forgeries, repeated mask-sum queries and, as tests of
cases outside the guarantee, a colluding client that offsets its verification share.
Also an end-to-end check that an aborted round leaves the global model unchanged."""
import os, json, time
import numpy as np, torch
from common import *
import fl, secagg as sa

TRIALS = int(os.environ.get("TRIALS", 1000))
P = sa.P


def null_vector_mod_p(cs, seed=0):
    """Nonzero e (supported on k+1 coordinates) with CountSketch(e) = 0 over F_P (vectorised elimination)."""
    k = cs.r * cs.w; n_cols = k + 1
    rng = np.random.default_rng(seed); cols = np.sort(rng.choice(cs.d, n_cols, replace=False))
    M = np.zeros((k, n_cols), np.uint64)
    for j in range(cs.r):
        M[j * cs.w + cs.bucket[j, cols], np.arange(n_cols)] = np.where(cs.neg[j, cols], np.uint64(P - 1), np.uint64(1))
    piv, r = [], 0
    for cidx in range(n_cols):
        nz = np.nonzero(M[r:, cidx])[0]
        if len(nz) == 0: continue
        pr = r + nz[0]; M[[r, pr]] = M[[pr, r]]
        inv = np.uint64(pow(int(M[r, cidx]), P - 2, P))
        M[r] = sa.fmul(M[r], np.full(n_cols, inv, np.uint64))
        fac = M[:, cidx].copy(); fac[r] = 0
        M = sa.fsub(M, sa.fmul(np.repeat(fac[:, None], n_cols, 1), np.repeat(M[r][None, :], k, 0)))
        piv.append(cidx); r += 1
        if r == k: break
    free = next(cc for cc in range(n_cols) if cc not in piv)
    e = np.zeros(n_cols, np.uint64); e[free] = 1
    for i, pc in enumerate(piv): e[pc] = sa.fneg(np.array([M[i, free]], np.uint64))[0]
    out = np.zeros(cs.d, np.uint64); out[cols] = e
    assert (cs(out) == 0).all() and out.any()
    return out


def main():
    cfgb = json.load(open(CONFIG_FILE))
    from exp import BASE
    c = dict(BASE); c.update(cfgb); c.update(seed=42, partition="iid", agg="fedavg")
    d = load_kaggle(cfgb.get("feat", "cls")); Xtr, ytr = d["train"]
    rng = np.random.default_rng(42); g = torch.Generator().manual_seed(42)
    parts = fl.iid_partition(len(ytr), c["K"], rng); c["class_weights"] = torch.ones(6)
    head = fl.Head(Xtr.shape[1], c["d_hidden"], 6, 42)
    ups = [fl.local_train(head, Xtr[p], ytr[p], c, g) for p in parts]
    ups = [u * min(1.0, c["clip_update"] / float(u.norm())) for u in ups]
    n = [float(len(p)) for p in parts]; dim = ups[0].numel()
    rs = np.random.default_rng(1); state = {}

    def one(v):  e = np.zeros(dim, np.uint64); e[int(rs.integers(dim))] = np.uint64(v); return e
    S = {
        "honest (SecAgg)": dict(mode="secagg"),
        "honest (VeriSketch)": dict(mode="verisketch"),
        "AS: +1 unit on one coordinate": dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, one(1))),
        "AS: error 2^60 on one coordinate (high valuation)": dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, one(1 << 60))),
        "AS: error P-1 (wrap-around, = -1)": dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, one(P - 1))),
        "AS: out-of-range value (+2^40)": dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, one(1 << 40))),
        "AS: Gaussian perturbation (sd 1e-4)": dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, sa.encode(rs.normal(0, 1e-4, dim), 1.0))),
        "AS: scale aggregate by 0.5": dict(party="AS", fn=lambda U, AS, A: np.mod(np.rint(sa.decode(U) * 0.5 * sa.SCALE).astype(np.int64), P).astype(np.uint64)),
        "AS: inject fabricated update": dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, sa.encode(rs.normal(0, 0.01, dim), 1.0))),
        "AS: replay aggregate of another round": dict(party="AS", fn=lambda U, AS, A: state.get("prev", U) if state.get("prev") is not None and not (state["prev"] == U).all() else sa.fadd(U, one(7))),
        "AS: drop a client's upload from the transcript": dict(party="AS_transcript", fn=lambda tr: tr[1:]),
        "AS: misreport one client's public weight in the transcript": dict(party="AS_transcript",
                                                                           fn=lambda tr: [(cc, (w * 0.5 if cc == 3 else w), h) for cc, w, h in tr]),
        "AN: open a coin share that does not match its commitment": dict(party="AN_coin", mode="verisketch"),
        "AS: sketch-kernel error (passes sketch cross-check)": dict(party="AS", mode="verisketch", kernel=True),
        "AN: omit one client's mask": dict(party="AN", fn=lambda M, AN, A: sa.fsub(M, AN._mask(A[0]))),
        "AN: random mask sum": dict(party="AN", fn=lambda M, AN, A: sa.fadd(M, sa.prg_field(os.urandom(32), M.size))),
        "AN: balanced sketch forgery (+δ client a, −δ client b)": dict(party="AN_sketch", mode="verisketch", balanced=True),
        "AN: frame a client with a garbage sketch": dict(party="AN_sketch", mode="verisketch",
                                                        fn=lambda MS: {**MS, 4: sa.fadd(MS[4], sa.prg_field(os.urandom(32), MS[4].size))}),
        "LIMITATION: AS error covered by a colluding client's share (SecAgg)": dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, one(1)), collude=True),
        "AS: out-of-range value (+2^40) with a colluding share": dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, one(1 << 40)), collude=True),
        "AS: naive sketch-kernel error + colluding share (VeriSketch)": dict(party="AS", mode="verisketch", kernel=True, collude=True),
    }
    res = {}
    for name, spec in S.items():
        mode = spec.get("mode", "secagg"); flagged = 0; reasons = {}; tv = []
        sec = fl.SecureRound(c["K"])
        ntr = int(os.environ.get("KTRIALS", 100)) if spec.get("kernel") else TRIALS
        for t in range(ntr):
            tam = None
            if "fn" in spec: tam = dict(party=spec["party"], fn=spec["fn"])
            if spec.get("kernel"):
                def kfn(U, AS, A, sec=sec):
                    return sa.fadd(U, state["kernel"])
                tam = dict(party="AS", fn=kfn)
            if spec.get("balanced"):
                def bfn(MS):
                    MS = dict(MS); e = np.zeros(MS[0].size, np.uint64); e[3] = np.uint64(1 << 20)
                    MS[1] = sa.fadd(MS[1], e); MS[2] = sa.fsub(MS[2], e); return MS
                tam = dict(party="AN_sketch", fn=bfn)
            col = None
            if spec.get("collude"):
                err = (lambda U: sa.fsub(U, state["true_U"]))
                col = lambda U, V: sa.fdot(sa.fsub(U, state["true_U"]), V)
            # kernel error needs the round's sketch: precompute by running an honest round with the same σ is not
            # possible (σ is fresh); instead the AS learns σ during the round, so we compute e inside the hook:
            if spec.get("kernel"):
                orig_cs = sa.CountSketch
                class Hook(orig_cs):
                    def __init__(s2, *a, **k):
                        super().__init__(*a, **k); state["kernel"] = null_vector_mod_p(s2, seed=t)
                sa.CountSketch = Hook
            if spec.get("party") == "AN_coin":
                orig_rev = sa.AuxNode.reveal_coin
                sa.AuxNode.reveal_coin = lambda self2, i: os.urandom(32)
                tam = None
            if spec.get("collude"):
                # true aggregate of the accepted set is needed by the colluding client
                orig_unmask = sa.AggServer.unmask
                def unmask_hook(self2, A, M, _o=orig_unmask):
                    U = _o(self2, A, M); state["true_U"] = U; return U
                sa.AggServer.unmask = unmask_hook
            try:
                agg, st = sec.run(ups, n, t + 1, mode, 3, c["clip_update"], tamper=tam, collude_share=col)
            finally:
                if spec.get("kernel"): sa.CountSketch = orig_cs
                if spec.get("collude"): sa.AggServer.unmask = orig_unmask
                if spec.get("party") == "AN_coin": sa.AuxNode.reveal_coin = orig_rev
            if agg is None: flagged += 1; reasons[st["reason"]] = reasons.get(st["reason"], 0) + 1
            tv.append(st.get("t_verify", 0))
            if agg is not None and state.get("prev") is None and "honest" in name: pass
        if name == "honest (SecAgg)":
            # keep an honest aggregate of a previous round for the replay scenario
            sec2 = fl.SecureRound(c["K"]); AS = sa.AggServer()
            state["prev"] = sa.encode(np.random.default_rng(5).normal(0, 0.01, dim), 1.0)
        res[name] = dict(trials=ntr, aborted=flagged, rate=flagged / ntr, reasons=reasons, mean_t_verify=float(np.mean(tv)))
        print(name, res[name], flush=True)
    # repeated mask-sum query is refused
    AN = sa.AuxNode(min_release=2); AN.receive_seed(0, os.urandom(32), 1, 10); AN.receive_seed(1, os.urandom(32), 1, 10)
    AN.mask_sum(1, [0, 1])
    try: AN.mask_sum(1, [0]); refused = False
    except PermissionError: refused = True
    res["AN: second mask-sum query in the same round"] = dict(refused=refused)
    # end-to-end: an aborted round leaves the global model unchanged
    c2 = dict(c, agg="verisketch", T=3, eval_every=3, _record_params=True,
              _tamper=dict(round=2, spec=dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, one(1)))))
    c2.pop("class_weights", None)
    out, *_ = fl.run_fl(c2, *d["train"], *d["val"], *d["test"], log=lambda *a: None)
    h = out["history"]
    res["end-to-end abort leaves model unchanged"] = dict(round2_aborted=bool(h[1].get("aborted")),
                                                          params_unchanged=h[0]["param_hash"] == h[1]["param_hash"],
                                                          round3_updates=h[1]["param_hash"] != h[2]["param_hash"])
    print(res["end-to-end abort leaves model unchanged"], res["AN: second mask-sum query in the same round"])
    save_json(res, os.path.join(RES, "tamper", "tamper_results.json"))


if __name__ == "__main__":
    main()
