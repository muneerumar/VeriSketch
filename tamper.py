"""Adversarial evaluation of the protocol (Table "Deviations injected into complete rounds").

Every trial uses fresh masks, coins and challenges on realistic updates (one honest
local-training round on the real features).  Scenarios: high-valuation errors, sketch-kernel
errors, wrap-around and out-of-range values, replay, transcript manipulation, wrong coin
openings, balanced auxiliary-node sketch forgeries, malformed requests to the AN and to clients and,
as tests of cases outside the guarantee, a colluding client that offsets its verification share.
A deviating server also reports acceptance in its own verdict; detection must come from an honest party.
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
    from exp import require_inputs; require_inputs()
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
    rs = np.random.default_rng(1); state = {}; g2 = torch.Generator().manual_seed(7)

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
        "AS: replay the accepted aggregate of the previous round": dict(replay="aggregate"),
        "AS: replay the transcript of the previous round": dict(replay="transcript"),
        "AN: open a coin share of another session": dict(party="AN_coin_other", mode="verisketch"),
        "AS: forged aggregate with a forged AN acceptance": dict(party="AS_forge_AN_verdict", fn=lambda U, AS, A: sa.fadd(U, one(1))),
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
            elif spec.get("party") == "AS_forge_AN_verdict": tam = dict(party=spec["party"])
            if spec.get("replay"):
                # an honest round with fresh updates first; its accepted aggregate / transcript is then replayed
                # in the next round of the same session, whose clients send new updates
                rr = 2 * t + 1; rec = {}
                ups_a = [u + 0.01 * torch.randn(u.shape, generator=g2) for u in ups]
                ups_a = [u * min(1.0, c["clip_update"] / float(u.norm())) for u in ups_a]
                o_unmask = sa.AggServer.unmask
                def rec_unmask(self2, A, M, _o=o_unmask):
                    U = _o(self2, A, M); rec["U"] = U; return U
                sa.AggServer.unmask = rec_unmask
                o_tr = sa.AggServer.transcript
                def rec_tr(self2, w, _o=o_tr):
                    tr = _o(self2, w); rec.setdefault("tr", tr); return tr
                sa.AggServer.transcript = rec_tr
                try: a0, s0 = sec.run(ups_a, n, rr, mode, 3, c["clip_update"])
                finally: sa.AggServer.unmask = o_unmask; sa.AggServer.transcript = o_tr
                assert a0 is not None
                ups_b = [u + 0.01 * torch.randn(u.shape, generator=g2) for u in ups]
                ups_b = [u * min(1.0, c["clip_update"] / float(u.norm())) for u in ups_b]
                if spec["replay"] == "aggregate": tam = dict(party="AS", fn=lambda U, AS, A, R=rec: R["U"])
                else: tam = dict(party="AS_transcript", fn=lambda tr, R=rec: R["tr"])
                agg, st = sec.run(ups_b, n, rr + 1, mode, 3, c["clip_update"], tamper=tam)
                if agg is None:
                    flagged += 1; kk = "; ".join(st.get("detected_by") or [st["reason"]]); reasons[kk] = reasons.get(kk, 0) + 1
                tv.append(st.get("t_verify", 0)); continue
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
                        super().__init__(*a, **k)
                        if state.get("ksig") != (a[0], t): state["kernel"] = null_vector_mod_p(s2, seed=t); state["ksig"] = (a[0], t)
                sa.CountSketch = Hook
            if spec.get("party") in ("AN_coin", "AN_coin_other"):
                orig_rev = sa.AuxNode.reveal_coin
                if spec["party"] == "AN_coin":
                    sa.AuxNode.reveal_coin = lambda self2, i: os.urandom(32)
                else:   # a valid opening, but of the coin committed in another session
                    other = sa.AuxNode(sa.SessionConfig(os.urandom(16), range(c["K"]), dim, 4, 64, mode="verisketch"))
                    other.open_attempt(1, 0)
                    sa.AuxNode.reveal_coin = lambda self2, i, o=other: o._coins[i]
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
                if spec.get("party") in ("AN_coin", "AN_coin_other"): sa.AuxNode.reveal_coin = orig_rev
            if agg is None:
                flagged += 1; kk = "; ".join(st.get("detected_by") or [st["reason"]]); reasons[kk] = reasons.get(kk, 0) + 1
            tv.append(st.get("t_verify", 0))
            if agg is not None and state.get("prev") is None and "honest" in name: pass
        res[name] = dict(trials=ntr, aborted=flagged, rate=flagged / ntr, reasons=reasons, mean_t_verify=float(np.mean(tv)))
        print(name, res[name], flush=True)
    # malformed requests to the AN and to clients (each must be refused)
    def refused(f, *a):
        try: return f(*a) is False
        except PermissionError: return True
    sid = os.urandom(16)
    def fresh(mode="secagg"):
        cfgq = sa.SessionConfig(sid, range(10), 16, 4, 16, None, 1, 5, 1, 2.0, mode)
        AN = sa.AuxNode(cfgq, {k: os.urandom(32) for k in range(10)}); AN.open_attempt(1, 0)
        for k in range(10): AN.receive_seed(sid, 1, 0, k, os.urandom(32))
        return AN
    q = {}
    AN = fresh()
    q["mask sum before the transcript digest"] = refused(AN.mask_sum, sid, 1, 0, list(range(10)))
    AN.receive_digest(sid, 1, 0, b"D0", os.urandom(32)); AN.mask_sum(sid, 1, 0, list(range(10)))
    q["second mask-sum query in the same attempt"] = refused(AN.mask_sum, sid, 1, 0, list(range(9)))
    q["query under a fabricated round with stored seeds"] = refused(AN.mask_sum, sid, 2, 0, list(range(9)))
    q["reuse of an attempt id"] = refused(AN.open_attempt, 1, 0)
    q["retry beyond the attempt limit"] = refused(AN.open_attempt, 1, 1)
    AN.open_attempt(3, 0)
    q["stale round"] = refused(AN.open_attempt, 2, 0)
    AN = fresh(); AN.receive_digest(sid, 1, 0, b"D0", os.urandom(32))
    q["duplicate identities (five copies of client 0)"] = refused(AN.mask_sum, sid, 1, 0, [0] * 5)
    q["unknown identity"] = refused(AN.mask_sum, sid, 1, 0, [0, 1, 2, 3, 77])
    q["set below ceil(K/2)"] = refused(AN.mask_sum, sid, 1, 0, [0, 1, 2, 3])
    q["singleton set"] = refused(AN.mask_sum, sid, 1, 0, [0])
    q["query from another session"] = refused(AN.mask_sum, os.urandom(16), 1, 0, list(range(5)))
    AN = fresh("verisketch")
    q["coin opening before the transcript digest"] = refused(AN.reveal_coin, 0)
    AN.receive_digest(sid, 1, 0, b"D0", os.urandom(32))
    q["challenge coin opening before the commitment"] = refused(AN.reveal_coin, 1)
    q["mask sketches before the sketch seed is fixed"] = refused(AN.mask_sketches, sid, 1, 0)
    AN.reveal_coin(0)
    q["enlarged mask sketch (8 x 512)"] = refused(AN.mask_sketches, sid, 1, 0, 8, 512)
    q["mask sum before the filtering step"] = refused(AN.mask_sum, sid, 1, 0, list(range(10)))
    AN.mask_sketches(sid, 1, 0)
    q["second mask-sketch request"] = refused(AN.mask_sketches, sid, 1, 0)
    # clients: the overlapping-set extraction of the review (fabricated rounds, sets of 10 and 9 clients)
    secq = fl.SecureRound(10); a0, _ = secq.run(ups, n, 1, "secagg", 3, c["clip_update"]); cl = secq.clients[0]; cq = secq.cfg
    q["client asked to open a round that the AN did not open"] = refused(cl.open_attempt, 2, 0, b"h1", b"h2",
                                                                          sa.mac(os.urandom(32), b"open", cq.hash, sa.ints(2, 0), b"h1", b"h2"))
    q["client asked for shares outside an open attempt (overlapping sets)"] = refused(cl.shares)
    ANq = secq.AN; h1, h2 = ANq.open_attempt(2, 0)
    for cc in secq.clients: cc.open_attempt(2, 0, h1, h2, ANq.open_tags[cc.cid])
    q["client asked to accept a set that the AN did not authorise"] = refused(cl.receive_accepted, list(range(1, 10)),
                                                                               sa.mac(os.urandom(32), b"accepted", cq.hash, sa.ints(2, 0), sa.ints(*range(1, 10))))
    q["client asked for shares before the commitment and the authorised set"] = refused(cl.shares)
    res["AN and client query enforcement"] = dict(refused=q, all_refused=all(q.values()))
    # end-to-end: an aborted round leaves the global model unchanged
    c2 = dict(c, agg="verisketch", T=3, eval_every=3, _record_params=True,
              _tamper=dict(round=2, spec=dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, one(1)))))
    c2.pop("class_weights", None)
    out, *_ = fl.run_fl(c2, *d["train"], *d["val"], *d["test"], log=lambda *a: None)
    h = out["history"]
    res["end-to-end abort leaves model unchanged"] = dict(round2_aborted=bool(h[1].get("aborted")),
                                                          params_unchanged=h[0]["param_hash"] == h[1]["param_hash"],
                                                          round3_updates=h[1]["param_hash"] != h[2]["param_hash"])
    # scope of the share check: two accepted clients add +1 and -1 to their shares on an honest aggregate
    passed = 0; same = 0
    sec = fl.SecureRound(c["K"]); ref = fl.SecureRound(c["K"])
    for t in range(100):
        a1, _ = sec.run(ups, n, t + 1, "secagg", 3, c["clip_update"], collude_share={0: lambda U, V: 1, 1: lambda U, V: P - 1})
        a0, _ = ref.run(ups, n, t + 1, "secagg", 3, c["clip_update"])
        passed += a1 is not None; same += bool(a1 is not None and torch.equal(a0, a1))
    res["SCOPE: cancelling share offsets (+1, -1) of two clients on an honest aggregate"] = dict(trials=100, passed=passed, aggregate_unchanged=same)
    print(res["end-to-end abort leaves model unchanged"], res["AN and client query enforcement"],
          res["SCOPE: cancelling share offsets (+1, -1) of two clients on an honest aggregate"])
    from exp import provenance_side
    res["provenance"] = provenance_side(["tamper.py"])
    save_json(res, os.path.join(RES, "tamper", "tamper_results.json"))


if __name__ == "__main__":
    main()
