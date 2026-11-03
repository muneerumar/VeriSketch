"""Checks that the hardened protocol code (secagg.py, fl.py) computes the same aggregate, accepted set
and trained model as the pre-hardening code in legacy/ that produced the main grids.

Both versions run in separate processes on identical inputs.  The sketch seed is fixed to a public
function of the round in both, because it is otherwise drawn from fresh coins; masks, mask seeds and
challenges stay random (they cancel or only affect the checks).  Compared: (1) 600 single rounds with
honest and poisoned updates under SecAgg, VeriSketch and VeriSketch-MK; (2) complete non-DP training
runs taken from the main grids.  Usage: python equivalence_check.py   (writes results/equivalence.json)"""
import os, sys, json, hashlib, subprocess
ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = [("robust", "signflip_verisketch_iid_s42"), ("robust", "flip_evasion_verisketch_mk_0.5_s42"),
        ("robust", "adaptive_verisketch_iid_s123"), ("baseline", "secagg_iid_s42")]


def impl_main(out):
    if os.environ.get("VS_IMPL") == "legacy": sys.path.insert(0, os.path.join(ROOT, "legacy"))
    import numpy as np, torch
    import secagg as sa, fl
    sa.sketch_seed = lambda sid, rnd, D, rho, att=0: sa.H(b"equivalence-fixed-seed", sa.ints(rnd, att))
    rec = {"code": {m: hashlib.sha256(open(sys.modules[m].__file__, "rb").read()).hexdigest() for m in ("secagg", "fl")}}
    rng = np.random.default_rng(0); rounds = []
    for mode in ("secagg", "verisketch", "verisketch_mk"):
        sec = fl.SecureRound(10)
        for r in range(1, 201):
            base = rng.normal(0, 0.01, 12406)
            ups = [torch.tensor(base + rng.normal(0, 0.004, 12406), dtype=torch.float32) for _ in range(10)]
            kind = r % 4
            for k in range(3):
                if kind == 1: ups[k] = -ups[k]                                  # sign flip
                elif kind == 2: ups[k] = ups[k] + torch.tensor(rng.normal(0, 0.02, 12406), dtype=torch.float32)
                elif kind == 3: ups[k] = ups[k] * 1.4
            ups = [u * min(1.0, 2.0 / float(u.norm())) for u in ups]
            n = [float(rng.integers(500, 4000)) for _ in range(10)]
            agg, st = sec.run(ups, n, r, mode, 3, 2.0)
            rounds.append(dict(mode=mode, r=r, accepted=list(map(int, st.get("accepted", []))), aborted=bool(st["aborted"]),
                               agg=None if agg is None else hashlib.sha256(agg.numpy().tobytes()).hexdigest()))
    rec["rounds"] = rounds
    import exp
    from common import load_kaggle
    d = load_kaggle("cls"); grids = {}; runs = []
    for g, name in RUNS:
        if g not in grids: grids[g] = dict(exp.grid(g))
        cfg = grids[g][name]
        o, head, pred, prob = fl.run_fl(cfg, *d["train"], *d["val"], *d["test"], log=lambda *a: None)
        runs.append(dict(name=name, test=o["test"], pred=hashlib.sha256(np.asarray(pred).tobytes()).hexdigest(),
                         selected=[h.get("selected") for h in o["history"]]))
    rec["runs"] = runs
    json.dump(rec, open(out, "w"))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--impl":
        impl_main(sys.argv[2]); sys.exit(0)
    tmp = os.path.join(ROOT, "results", "equivalence"); os.makedirs(tmp, exist_ok=True)
    outs = {}
    for tag in ("legacy", "current"):
        o = os.path.join(tmp, f"{tag}.json")
        env = dict(os.environ, VS_IMPL=tag)
        subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "--impl", o], env=env, check=True, cwd=tmp)
        outs[tag] = json.load(open(o))
    L, C = outs["legacy"], outs["current"]
    same_rounds = sum(a == b for a, b in zip(L["rounds"], C["rounds"]))
    same_runs = [a == b for a, b in zip(L["runs"], C["runs"])]
    res = dict(legacy_code=L["code"], current_code=C["code"], rounds=len(L["rounds"]), rounds_identical=same_rounds,
               rounds_aborted=sum(x["aborted"] for x in L["rounds"]),
               runs=[dict(name=r["name"], identical=s, test_macro_f1=r["test"].get("macro_f1")) for r, s in zip(L["runs"], same_runs)],
               all_identical=bool(same_rounds == len(L["rounds"]) and all(same_runs)))
    json.dump(res, open(os.path.join(ROOT, "results", "equivalence.json"), "w"), indent=1)
    if res["all_identical"]:            # record the code lineage used by exp.legacy_ok
        sys.path.insert(0, ROOT); import exp
        lin = dict(legacy_code_sha256={f: exp.file_sha256(os.path.join(ROOT, "legacy", f)) for f in ("secagg.py", "fl.py")},
                   current_code_sha256={f: exp.file_sha256(os.path.join(ROOT, f)) for f in ("secagg.py", "fl.py", "common.py")},
                   normaliser_stats_sha256_retrospective=exp.normaliser_record()["stats_sha256"],
                   note=("Main-grid records (schema 1) store manifest, feature, configuration and code hashes written at run time. "
                         "The normaliser hash was added later; its value here was computed retrospectively from this snapshot. "
                         "equivalence_check.py found identical rounds and runs for the legacy and current code."),
                   equivalence=dict(rounds=res["rounds"], rounds_identical=res["rounds_identical"],
                                    runs_identical=sum(r["identical"] for r in res["runs"]), runs=len(res["runs"])))
        lin["legacy_code_sha256"]["common.py"] = lin["current_code_sha256"]["common.py"]
        json.dump(lin, open(os.path.join(ROOT, "config", "code_lineage.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_identical"] else 1)
