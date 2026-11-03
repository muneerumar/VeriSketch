"""Design check on the final split.  Test-set evaluation is disabled
(eval_test=False) and every record is time-stamped.  Usage: python tune.py [shard i/n]"""
import json, os, sys, glob, datetime, numpy as np
from common import *
import fl
from exp import BASE, save_with_provenance, cached_ok
from exp import require_inputs; require_inputs()      # stop before any run or cache reuse
shard, nsh = (map(int, sys.argv[1].split("/")) if len(sys.argv) > 1 else (0, 1))
d = load_kaggle("cls"); base = dict(BASE); base.update(json.load(open(CONFIG_FILE)))
grid = [(dh, dp, lr) for dh in (16, 32, 128) for dp in (None, 2.0) for lr in ((1e-3, 3e-4) if dp is None else (3e-3, 1e-3))]
part = os.path.join(RES, "tune", f"design_check_part{shard}of{nsh}.json")
SET = dict(grid=grid, shard=[shard, nsh])
out = json.load(open(part)) if cached_ok(part, ["tune.py"], SET) else []     # resume only a part with matching provenance
done = {(r["d_hidden"], r["eps"], r["lr"]) for r in out}
for i, (dh, dp, lr) in enumerate(grid):
    if i % nsh != shard or (dh, dp, lr) in done: continue
    c = dict(base, d_hidden=dh, lr=lr, seed=42, agg="fedavg", partition="iid", eval_every=30, eval_test=False,
             dp=(dict(epsilon=dp, delta=1e-5, clip=base["dp_clip"], q=base["dp_q"]) if dp else None))
    o, *_ = fl.run_fl(c, *d["train"], *d["val"], *d["test"], log=lambda *a: None)
    assert o["test"] == {}
    r = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), d_hidden=dh, eps=dp, lr=lr,
             val_macro_f1=o["history"][-1]["val_macro_f1"], val_tmr=o["history"][-1]["val_tmr"], val_far=o["history"][-1]["val_far"])
    out.append(r); print(r, flush=True)
    save_with_provenance(out, part, ["tune.py"], SET)
parts = glob.glob(os.path.join(RES, "tune", f"design_check_part*of{nsh}.json"))
if len(parts) == nsh:
    allr = sum((json.load(open(p)) for p in sorted(parts) if cached_ok(p, ["tune.py"], dict(grid=grid, shard=[int(p.split("part")[1].split("of")[0]), nsh]))), [])
    if len(allr) == len(grid):
        key = {(r["d_hidden"], r["eps"], r["lr"]): r for r in allr}
        save_with_provenance([key[g] for g in grid], os.path.join(RES, "tune", "design_check.json"), ["tune.py"], dict(grid=grid))
