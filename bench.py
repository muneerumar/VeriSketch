"""Cost of complete simulated protocol rounds (single process, all parties in one address space).

For each configuration a fresh subprocess times REPS complete rounds of VeriSketch and, as the matched
comparator, of SecAgg-FedAvg with the same masking, transcript, release, commitment and challenge
(SecureRound.run with mode "secagg"); the difference is the incremental cost of sketching and filtering.
Key setup (X25519 key generation and agreement) is timed separately.  Peak RSS is measured per
subprocess, so configurations do not share a high-water mark.  Communication is an explicit message
ledger (8-byte field elements).  Usage: python bench.py  (or python bench.py --one d K rows width kappa)."""
import os, sys, time, json, platform, resource, subprocess, numpy as np, torch
from common import *
import fl, secagg as sa

REPS = int(os.environ.get("BENCH_REPS", 20))


def ledger(d, K, r, w, kappa=1, T=30, verisketch=True):
    """Bytes per round per message type (A = all K clients accepted; 32-byte hashes, coins and MAC tags;
    8-byte field elements and identifiers)."""
    tr = K * (8 + 8 + 32)                                         # transcript entry: id, public weight, upload hash
    vs = 1 if verisketch else 0
    L = dict(
        AN_to_client_attempt_opening=K * (2 * 32 + 32),           # coin commitments + MAC
        AN_to_AS_coin_commitments=2 * 32,
        client_to_AS_masked_upload=K * d * 8,
        client_to_AN_mask_seed=K * 32,
        AS_to_client_transcript_and_digest=K * (tr + 32),
        client_to_AS_receipt_ack=K * 32,
        AS_to_AN_digest_and_coin_share=32 + 32,
        AN_to_AS_coin_opening_1=vs * 32,
        AN_to_client_coin_opening_1=vs * K * (32 + 32 + 32),      # AN share, AS share, MAC
        AN_to_AS_mask_sketches=vs * K * r * w * 8,
        AS_to_client_own_sketch=vs * K * r * w * 8,
        AS_to_AN_accepted_set=8 * K,
        AN_to_AS_mask_sum=d * 8,
        AN_to_client_authorised_set=K * (8 * K + 32),
        AS_to_AN_commitment_and_coin_share=32 + 32,
        AN_to_AS_coin_opening_2=32,
        AN_to_client_commitment_and_coin_opening_2=K * (32 + 32 + 32 + 32),
        client_to_AN_shares=K * (8 * kappa + 32),
        client_to_AS_shares=K * (8 * kappa + 32),
        AS_to_AN_aggregate=d * 8,
        AS_to_client_aggregate_broadcast=K * d * 8,               # clients update the model from this broadcast;
        AN_to_client_verdict=K * (1 + 32),                        # no separate model download is sent
        AS_to_client_verdict=K * (1 + 32),
    )
    L["total_round_B"] = sum(L.values())
    L["setup_public_keys_once_B"] = K * 32 * (K - 1) + K * 2 * 32  # pairwise X25519 keys and the two server channels
    L["setup_amortised_per_round_B"] = L["setup_public_keys_once_B"] / T
    L["plaintext_fedavg_round_B"] = K * d * 4 * 2                 # float32 upload + model download
    return L


def one(d, K, rows, width, kappa):
    rng = np.random.default_rng(0)
    ups = [torch.tensor(rng.normal(0, 1, d) / np.sqrt(d) * 1.5, dtype=torch.float32) for _ in range(K)]
    n = [float(rng.integers(2000, 4000)) for _ in range(K)]
    t = time.perf_counter(); sec_v = fl.SecureRound(K, rows, width); t_setup = time.perf_counter() - t
    sec_s = fl.SecureRound(K, rows, width)
    tv, ts, parts = [], [], []
    for rep in range(REPS + 1):                                    # first repetition is warm-up
        t = time.perf_counter(); a, st = sec_v.run(ups, n, rep + 1, "verisketch", 3, 2.0, kappa=kappa); dt_v = time.perf_counter() - t
        t = time.perf_counter(); b, _ = sec_s.run(ups, n, rep + 1, "secagg", 3, 2.0, kappa=kappa); dt_s = time.perf_counter() - t
        assert a is not None and b is not None
        if rep: tv.append(dt_v); ts.append(dt_s); parts.append({k: st[k] for k in ("t_client_mask", "t_sketch", "t_unmask", "t_verify")})
    U = torch.stack(ups); t = time.perf_counter()
    for _ in range(REPS): (U * torch.tensor(n)[:, None]).sum(0) / sum(n)
    t_plain = (time.perf_counter() - t) / REPS
    q = lambda x: dict(median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                       min=float(np.min(x)), max=float(np.max(x)))
    inc = np.array(tv) - np.array(ts)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / 1024 if sys.platform.startswith("linux") else rss / 2 ** 20      # Linux: KiB; macOS: bytes
    return dict(d=d, K=K, rows=rows, width=width, kappa=kappa, reps=REPS, setup_s=t_setup,
                verisketch_round_s=q(tv), secagg_round_s=q(ts), incremental_s=q(inc),
                components_median={k: float(np.median([p[k] for p in parts])) for k in parts[0]},
                plaintext_mean_s=t_plain, peak_rss_MiB=rss_mib, ledger_verisketch=ledger(d, K, rows, width, kappa),
                ledger_secagg=ledger(d, K, rows, width, kappa, verisketch=False))


def confidentiality(d=12406, n_trials=20):
    rng = np.random.default_rng(1); r_field = []
    for _ in range(n_trials):
        u = rng.normal(0, 1, d) / np.sqrt(d) * 1.5
        x = sa.fadd(sa.encode(u, 2.0), sa.prg_field(os.urandom(32), d))
        r_field.append(abs(np.corrcoef(sa.to_signed(x).astype(float), u)[0, 1]))
    return dict(field_abs_corr_mean=float(np.mean(r_field)), field_abs_corr_max=float(np.max(r_field)))


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "--one"):
        from exp import require_inputs; require_inputs()
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print(json.dumps(one(*[int(x) for x in sys.argv[2:7]]))); sys.exit(0)
    cfgs = [(12406, 10, 4, 64, 1), (595206, 10, 4, 64, 1), (2_000_000, 10, 4, 64, 1),
            (12406, 5, 4, 64, 1), (12406, 20, 4, 64, 1), (12406, 10, 4, 256, 1), (12406, 10, 4, 64, 2)]
    runs = []
    for c in cfgs:
        r = subprocess.run([sys.executable, __file__, "--one", *map(str, c)], capture_output=True, text=True, check=True)
        runs.append(json.loads(r.stdout.strip().splitlines()[-1])); print(c, runs[-1]["verisketch_round_s"]["median"],
                                                                       runs[-1]["secagg_round_s"]["median"], flush=True)
    cpu = ""
    try: cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception: pass
    res = dict(platform=dict(system=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                             cpu=cpu, cores=os.cpu_count(), torch_threads=torch.get_num_threads()),
               note="single-process simulation of all parties; no network", confidentiality=confidentiality(), runs=runs)
    from exp import provenance_side
    res["provenance"] = provenance_side(["bench.py"])
    save_json(res, os.path.join(RES, "bench", "bench.json"))
