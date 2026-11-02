"""Cost of one complete protocol round, timed end to end.

For the 12,406-parameter head and synthetic dimensions, a complete VeriSketch round (masking,
transcript and receipts, sketch release, owner checks, filtering, single mask-sum release,
unmasking, commitment, challenge, shares, verification, range and sketch checks) is timed
with SecureRound.run.  Key setup (X25519 key generation and agreement) is timed separately.
Communication is counted message by message; peak RSS is recorded."""
import os, time, json, resource, numpy as np, torch
from common import *
import fl, secagg as sa


def comm(d, K, r, w, kappa=1):
    """Bytes per round (field elements are sent as 8-byte words; one-time setup listed separately)."""
    tr = K * (8 + 8 + 32)                                   # transcript: id, public weight, upload hash per client
    c = dict(client_to_AS_upload_B=d * 8, client_to_AN_seed_B=32, client_to_AS_shares_B=8 * kappa,
             AS_to_client_B=tr + r * w * 8 + d * 8 + 32 * 4,  # transcript, own sketch, aggregate, commitment + coin shares
             AN_to_AS_mask_sketches_B=K * r * w * 8, AN_to_AS_mask_sum_B=d * 8, AN_coins_B=4 * 32,
             model_download_per_client_B=d * 4, setup_public_keys_per_client_B=32 * (K - 1))
    c["total_round_B"] = (K * (c["client_to_AS_upload_B"] + 32 + c["client_to_AS_shares_B"] + c["AS_to_client_B"])
                          + c["AN_to_AS_mask_sketches_B"] + c["AN_to_AS_mask_sum_B"] + c["AN_coins_B"])
    c["plaintext_fedavg_round_B"] = K * d * 4 * 2
    return c


def bench(d, K=10, reps=3, rows=4, width=64, kappa=1):
    rng = np.random.default_rng(0)
    ups = [torch.tensor(rng.normal(0, 1, d) / np.sqrt(d) * 1.5, dtype=torch.float32) for _ in range(K)]
    n = [float(rng.integers(2000, 4000)) for _ in range(K)]
    t = time.time(); sec = fl.SecureRound(K, rows, width); t_setup = time.time() - t
    times, parts = [], []
    for rep in range(reps):
        t = time.time(); agg, st = sec.run(ups, n, rep + 1, "verisketch", 3, 2.0, kappa=kappa); times.append(time.time() - t)
        assert agg is not None
        parts.append({k: st[k] for k in ("t_client_mask", "t_sketch", "t_unmask", "t_verify")})
    t = time.time(); _ = torch.stack(ups).mean(0); t_plain = time.time() - t
    return dict(d=d, K=K, rows=rows, width=width, setup_s=t_setup, round_s_median=float(np.median(times)),
                round_s_all=times, components_median={k: float(np.median([p[k] for p in parts])) for k in parts[0]},
                plaintext_mean_s=t_plain, peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                kappa=kappa, **comm(d, K, rows, width, kappa))


def confidentiality(d=12406, n_trials=20):
    rng = np.random.default_rng(1); r_field, r_float = [], []
    for _ in range(n_trials):
        u = rng.normal(0, 1, d) / np.sqrt(d) * 1.5
        x = sa.fadd(sa.encode(u, 2.0), sa.prg_field(os.urandom(32), d))
        r_field.append(abs(np.corrcoef(sa.to_signed(x).astype(float), u)[0, 1]))
        uw = u * 3267                                          # earlier implementation: weighted update + N(0,1)*1e-3
        xf = uw + rng.normal(0, 1.0, d) * 0.001
        r_float.append(np.corrcoef(xf, uw)[0, 1]); rel = np.linalg.norm(xf - uw) / np.linalg.norm(uw)
    return dict(field_abs_corr_mean=float(np.mean(r_field)), field_abs_corr_max=float(np.max(r_field)),
                float_corr_mean=float(np.mean(r_float)), float_relative_mask_norm=float(rel))


if __name__ == "__main__":
    res = dict(confidentiality=confidentiality(), scaling=[bench(d) for d in (12406, 595206, 2_000_000)],
               scaling_K=[bench(12406, K=k) for k in (5, 10, 20)], scaling_sketch=[bench(12406, rows=4, width=256)], scaling_kappa=[bench(12406, kappa=2)])
    print(json.dumps(res, indent=1)[:3000])
    save_json(res, os.path.join(RES, "bench", "bench.json"))
