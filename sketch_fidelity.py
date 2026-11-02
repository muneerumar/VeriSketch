"""Isolates the effect of sketching from the selection rule.

For real client updates (IID and Dirichlet(0.5); several attacks; rounds 1/10/20) the exact
pairwise distances and norms are compared with their CountSketch estimates, and the adaptive
rule's accept/reject decisions on sketches are compared with those on exact values."""
import os, json, numpy as np, torch
from common import *
import fl, secagg as sa
from exp import BASE

SKETCHES = ((1, 16), (2, 32), (4, 16), (4, 64), (4, 256), (8, 128))


def main():
    cfgb = json.load(open(CONFIG_FILE))
    d = load_kaggle("cls"); Xtr, ytr = d["train"]; rows = []
    for seed in SEEDS[:3]:
        for part in ("iid", 0.5):
            for atk in ("none", "flip_evasion", "signflip_nm", "adaptive"):
                c = dict(BASE); c.update(cfgb); c.update(seed=seed, partition=part, class_weights=torch.ones(6))
                rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
                parts = fl.iid_partition(len(ytr), 10, rng) if part == "iid" else fl.dirichlet_partition(ytr.numpy(), 10, 0.5, rng)
                mal = [0, 1, 2] if atk != "none" else []
                head = fl.Head(768, c["d_hidden"], 6, seed)
                for r in range(1, 21):
                    ups = []
                    for k, p in enumerate(parts):
                        yk = fl.poison_labels(ytr[p], atk) if (k in mal and atk == "flip_evasion") else ytr[p]
                        u = fl.local_train(head, Xtr[p], yk, c, g); ups.append(u * min(1.0, 2.0 / float(u.norm())))
                    hon = [u for k, u in enumerate(ups) if k not in mal]
                    if atk == "signflip_nm":
                        for k in mal: ups[k] = fl.malicious_update("signflip_nm", ups[k], hon, 3, 10, g)
                    if atk == "adaptive":
                        mv, _ = fl.filter_aware_attack(hon, 3, 10, 2.0)
                        for k in mal: ups[k] = mv
                    if r in (1, 10, 20):
                        U = torch.stack(ups); en = U.norm(dim=1).numpy(); ED = fl._pdist(U).numpy()
                        sel_exact = set(fl.adaptive_select(en, ED, 3, 2.0))
                        for rw in SKETCHES:
                            cs = sa.CountSketch(os.urandom(32), head.dim, *rw)
                            F = [cs.to_float(cs(sa.encode(u.numpy(), 1e3))) for u in ups]
                            sn = np.array([sa.sketch_sqnorm(f) ** 0.5 for f in F])
                            SD = np.array([[sa.sketch_sqdist(F[a], F[b]) if a != b else 0 for b in range(10)] for a in range(10)])
                            sel_sk = set(fl.adaptive_select(sn, SD, 3, 2.0))
                            iu = np.triu_indices(10, 1)
                            rows.append(dict(seed=seed, partition=str(part), attack=atk, round=r, sketch=f"{rw[0]}x{rw[1]}",
                                             dist_rel_err_median=float(np.median(np.abs(SD[iu] - ED[iu]) / (ED[iu] + 1e-12))),
                                             norm_rel_err_median=float(np.median(np.abs(sn - en) / (en + 1e-12))),
                                             decision_disagreement=len(sel_exact ^ sel_sk) / 10,
                                             mal_accepted_exact=len(sel_exact & set(mal)), mal_accepted_sketch=len(sel_sk & set(mal))))
                    agg = (torch.tensor([len(p) for p in parts], dtype=torch.float32)[:, None] * torch.stack(ups)).sum(0) / sum(len(p) for p in parts)
                    head.set_flat(head.flat() + agg)
                print(seed, part, atk, flush=True)
                save_json(rows, os.path.join(RES, "fidelity", "sketch_fidelity.json"))


if __name__ == "__main__":
    main()
