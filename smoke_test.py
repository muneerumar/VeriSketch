"""Fresh-machine smoke test (< 5 min on 2 CPU cores): field arithmetic, DP clipping vs autograd,
a 3-round VeriSketch run on 2,000 training tweets, and an injected forgery that must abort
the round without changing the model."""
import subprocess, sys, json, os, numpy as np, torch
from common import *
import fl, secagg as sa
from exp import BASE
assert subprocess.run([sys.executable, "test_secagg.py"]).returncode == 0
assert subprocess.run([sys.executable, "test_dp.py"]).returncode == 0
assert subprocess.run([sys.executable, "test_protocol.py"]).returncode == 0
d = load_kaggle("cls"); Xtr, ytr = d["train"]
cfg = dict(BASE); cfg.update(json.load(open(CONFIG_FILE)))
cfg.update(seed=0, agg="verisketch", T=3, eval_every=3, eval_test=False, _record_params=True,
           _tamper=dict(round=2, spec=dict(party="AS", fn=lambda U, AS, A: sa.fadd(U, np.eye(1, U.size, 7, dtype=np.uint64)[0]))))
sub = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(0))[:2000]
out, *_ = fl.run_fl(cfg, Xtr[sub], ytr[sub], *d["val"], *d["test"], log=lambda *a: None)
h = out["history"]
assert h[1]["aborted"] and h[0]["param_hash"] == h[1]["param_hash"] and h[1]["param_hash"] != h[2]["param_hash"]
print("validation macro-F1 after 3 rounds:", round(h[-1]["val_macro_f1"], 3))
print("SMOKE TEST PASSED")
