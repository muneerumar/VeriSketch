"""Design check on the final split.  Test-set evaluation is disabled
(eval_test=False) and every record is time-stamped."""
import json, os, time, datetime, numpy as np
from common import *
import fl
from exp import BASE
d = load_kaggle("cls"); base = dict(BASE); base.update(json.load(open(CONFIG_FILE)))
out = []
for dh in (16, 32, 128):
    for dp in (None, 2.0):
        for lr in ((1e-3, 3e-4) if dp is None else (3e-3, 1e-3)):
            c = dict(base, d_hidden=dh, lr=lr, seed=42, agg="fedavg", partition="iid", eval_every=30, eval_test=False,
                     dp=(dict(epsilon=dp, delta=1e-5, clip=base["dp_clip"], q=base["dp_q"]) if dp else None))
            o, *_ = fl.run_fl(c, *d["train"], *d["val"], *d["test"], log=lambda *a: None)
            assert o["test"] == {}
            r = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), d_hidden=dh, eps=dp, lr=lr,
                     val_macro_f1=o["history"][-1]["val_macro_f1"], val_tmr=o["history"][-1]["val_tmr"], val_far=o["history"][-1]["val_far"])
            out.append(r); print(r, flush=True)
            save_json(out, os.path.join(RES, "tune", "design_check.json"))
