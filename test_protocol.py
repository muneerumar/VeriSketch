"""Negative and boundary tests for the protocol implementation.
Run: python test_protocol.py"""
import os, numpy as np, torch
import secagg as sa, fl

P = sa.P


def refused(f, *a):
    """Raises a protocol error or returns False (a failed check)."""
    try: return f(*a) is False
    except (PermissionError, ValueError): return True


# --- 1. encoding: non-finite values and invalid bounds are rejected; exact integer bound
assert refused(sa.encode, np.array([np.nan]), 1.0), "NaN accepted"
assert refused(sa.encode, np.array([np.inf]), 1.0), "inf accepted"
assert refused(sa.encode, np.array([0.1]), float("nan")), "NaN bound accepted"
assert refused(sa.encode, np.array([0.1]), 0.0), "zero bound accepted"
assert refused(sa.encode, np.array([0.1]), -1.0), "negative bound accepted"
assert refused(sa.encode, np.array([1.0 + 2 ** -20]), 1.0), "out-of-range value accepted"
assert sa.to_signed(sa.encode(np.array([1.0, -1.0]), 1.0)).tolist() == [1 << 24, -(1 << 24)]

# --- 2. rounding boundary: ten honest one-coordinate updates equal to the bound must pass
class _Hd:  # minimal stand-in for the head dimension
    pass
for sizes in ([2] * 9 + [7], [3, 5, 7, 11, 13, 17, 19, 23, 29, 31], [1] * 10):
    sec = fl.SecureRound(10)
    ups = [torch.tensor([2.0]) for _ in range(10)]
    for mode in ("secagg",):
        agg, st = sec.run(ups, [float(s) for s in sizes], 1, mode, 3, 2.0)
        assert agg is not None and not st["aborted"], (sizes, st)
        assert abs(float(agg[0]) - 2.0) <= 10 * 2 ** -25 * max(sizes) / min(sizes) + 1e-6
# negative boundary values as well, under VeriSketch, over several rounds of one session
sec = fl.SecureRound(10)
for r in range(1, 4):
    ups = [torch.tensor([-1.2, 1.2, 0.0, 0.8]) for _ in range(10)]
    agg, st = sec.run(ups, [float(s) for s in [2] * 9 + [7]], r, "verisketch", 3, 2.0)
    assert agg is not None, st

# --- 3. challenge sampler: rejection sampling never outputs P and is deterministic
v = sa.prg_field(b"\x01" * 32, 200000)
assert int(v.max()) < P and v.dtype == np.uint64
assert (sa.prg_field(b"\x01" * 32, 1000) == v[:1000]).all()

# --- 4. auxiliary-node release enforcement and pinned configuration
sid = os.urandom(16)
def fresh(K=10, minr=None, mode="secagg", d=8, rows=4, width=16):
    cfg = sa.SessionConfig(sid, range(K), d, rows, width, None, 1, minr or (K + 1) // 2, 1, 2.0, mode)
    AN = sa.AuxNode(cfg, {k: os.urandom(32) for k in range(K)})
    AN.open_attempt(1, 0)
    for c in range(K): AN.receive_seed(sid, 1, 0, c, os.urandom(32))
    return AN
AN = fresh()
assert refused(AN.mask_sum, sid, 1, 0, list(range(10))), "mask sum before the transcript digest"
AN.receive_digest(sid, 1, 0, b"D0", os.urandom(32)); AN.mask_sum(sid, 1, 0, list(range(10)))
assert refused(AN.mask_sum, sid, 1, 0, list(range(9))), "second release in the same attempt"
assert refused(AN.mask_sum, sid, 2, 0, list(range(9))), "fabricated round accepted with old seeds"
assert refused(AN.open_attempt, 1, 0), "attempt id reused"
assert refused(AN.open_attempt, 1, 1), "second attempt beyond the limit"
AN.open_attempt(2, 0)
assert refused(AN.mask_sum, sid, 2, 0, list(range(9))), "release without seeds of the new round"
assert refused(AN.open_attempt, 1, 5), "stale round accepted"
AN = fresh(K=10, minr=5); AN.receive_digest(sid, 1, 0, b"D0", os.urandom(32))
assert refused(AN.mask_sum, sid, 1, 0, [0] * 5), "duplicate identities accepted"
assert refused(AN.mask_sum, sid, 1, 0, [0, 1, 2, 3, 99]), "unknown identity accepted"
assert refused(AN.mask_sum, sid, 1, 0, [0]), "singleton set accepted"
assert refused(AN.mask_sum, os.urandom(16), 1, 0, [0, 1, 2, 3, 4]), "wrong session accepted"
assert refused(AN.receive_seed, sid, 1, 0, 3, os.urandom(32)), "duplicate or late seed accepted"
assert fl.SecureRound(5).min_release == 3, "ceil(K/2) for K = 5"
# R3: sketch size and protocol order are fixed by the AN
AN = fresh(mode="verisketch")
assert refused(AN.reveal_coin, 0), "coin opened before the transcript digest"
AN.receive_digest(sid, 1, 0, b"D0", os.urandom(32))
assert refused(AN.mask_sketches, sid, 1, 0), "mask sketches before the sketch seed is fixed"
assert refused(AN.reveal_coin, 1), "challenge coin opened before the commitment"
AN.reveal_coin(0)
assert refused(AN.mask_sketches, sid, 1, 0, 8, 512), "enlarged sketch (8 x 512) released"
assert refused(AN.mask_sketches, sid, 1, 0, 4, 16, 4), "sketch of another coordinate subset released"
assert refused(AN.mask_sum, sid, 1, 0, list(range(10))), "mask sum before the filtering step"
AN.mask_sketches(sid, 1, 0)
assert refused(AN.mask_sketches, sid, 1, 0), "second mask-sketch release"
assert refused(AN.verdict, np.zeros(8, np.uint64), {}), "verdict before the challenge"
# durable state: a restarted AN refuses the released attempt
import tempfile
pth = os.path.join(tempfile.mkdtemp(), "an_state.jsonl")
cfg4 = sa.SessionConfig(sid, range(4), 8, 4, 16, None, 1, 2, 1, 2.0, "secagg")
AN = sa.AuxNode(cfg4, {}, pth); AN.open_attempt(7, 0)
for c in range(4): AN.receive_seed(sid, 7, 0, c, os.urandom(32))
AN.receive_digest(sid, 7, 0, b"D0", os.urandom(32)); AN.mask_sum(sid, 7, 0, [0, 1, 2])
AN2 = sa.AuxNode(cfg4, {}, pth)
assert refused(AN2.open_attempt, 7, 0) and refused(AN2.open_attempt, 6, 0), "state lost on restart"

# --- 5. clients answer only the open, authorised attempt (R2: overlapping-set extraction must fail)
K, d = 10, 96
sec = fl.SecureRound(K)
ups = [torch.randn(d) * 0.05 for _ in range(K)]
agg, st = sec.run(ups, [100.0] * K, 1, "secagg", 3, 2.0)
assert agg is not None, st
c0 = sec.clients[0]; cfg = sec.cfg
# the AS fabricates round 2 and asks for basis-vector challenges for two overlapping sets
fake_tag = sa.mac(os.urandom(32), b"open", cfg.hash, sa.ints(2, 0), b"h1", b"h2")
assert refused(c0.open_attempt, 2, 0, b"h1", b"h2", fake_tag), "client opened an attempt without the AN"
assert refused(c0.shares), "client answered without an open attempt"
# a real attempt: the client answers only for the AN-authorised set and only once
AN = sec.AN; h1, h2 = AN.open_attempt(2, 0)
for c in sec.clients: c.open_attempt(2, 0, h1, h2, AN.open_tags[c.cid])
assert refused(c0.shares), "answer before the authorised set and the commitment"
x, seed = c0.mask_update(sa.encode(ups[0].double().numpy() * 0.5, 2.0))
assert refused(c0.mask_update, sa.encode(ups[0].double().numpy() * 0.5, 2.0)), "second upload in the same attempt"
assert refused(c0.receive_accepted, list(range(1, K)), sa.mac(os.urandom(32), b"accepted", cfg.hash, sa.ints(2, 0), sa.ints(*range(1, K)))), \
    "unauthorised set accepted"
other = sa.Client(0); other.register(sa.SessionConfig(os.urandom(16), range(K), d, 4, 64), c0.k_an, c0.k_as)
assert refused(other.open_attempt, 2, 0, h1, h2, AN.open_tags[0]), "message of another session accepted"
# pairwise masks differ across attempt ids and challenge indices
ka = int.from_bytes(__import__("hmac").new(c0.shared[1], sec.sid + __import__("struct").pack("<QQQ", 2, 0, 0) + b"ip", "sha256").digest()[:8], "little")
kb = int.from_bytes(__import__("hmac").new(c0.shared[1], sec.sid + __import__("struct").pack("<QQQ", 2, 1, 0) + b"ip", "sha256").digest()[:8], "little")
assert ka != kb, "pairwise mask does not depend on the attempt id"

# --- 6. honest-party acceptance (R1): a deviating server cannot make clients accept
ups = [torch.full((5,), 0.1 * (k + 1)) for k in range(4)]
agg_ref, _ = fl.SecureRound(4).run(ups, [10.0] * 4, 1, "secagg", 1, 2.0)
plus1 = lambda U, AS, A: sa.fadd(U, np.eye(1, U.size, 2, dtype=np.uint64)[0])
agg, st = fl.SecureRound(4).run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, tamper=dict(party="AS", fn=plus1))
assert agg is None and st["reason"].startswith("AN verdict: challenge"), st["reason"]          # AS also claims acceptance
agg, st = fl.SecureRound(4).run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, tamper=dict(party="AS_forge_AN_verdict"))
assert agg is None and st["reason"] == "AN verdict not authentic", st["reason"]
agg, st = fl.SecureRound(4).run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, tamper=dict(party="AN", fn=lambda M, AN, A: sa.fadd(M, np.ones(M.size, np.uint64))))
assert agg is None and st["reason"].startswith("AS verdict: challenge"), st["reason"]        # AN also claims acceptance
# scope of the share check: cancelling share offsets pass on an HONEST aggregate (they do not change it)
agg, st = fl.SecureRound(4).run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, collude_share={0: lambda U, V: 1, 1: lambda U, V: P - 1})
assert agg is not None and torch.allclose(agg, agg_ref), "cancelling offsets should pass and leave the aggregate unchanged"
# a single wrong share aborts (denial of service, no forgery)
agg, st = fl.SecureRound(4).run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, collude_share=lambda U, V: 1)
assert agg is None and "challenge" in st["reason"], st["reason"]
# a colluding client covers an AS error e = U - true_sum by adding +<e, V> (sign as in the paper)
state = {}
orig_unmask = sa.AggServer.unmask
def unmask_and_forge(self, A, M):
    U = orig_unmask(self, A, M); state["true"] = U
    e = np.zeros(U.size, np.uint64); e[2] = np.uint64(1); return sa.fadd(U, e)
sa.AggServer.unmask = unmask_and_forge
agg, st = fl.SecureRound(4).run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, collude_share=lambda U, V: sa.fdot(sa.fsub(U, state["true"]), V))
sa.AggServer.unmask = orig_unmask
assert agg is not None, "AS error with a +<e,V> colluding share passes (stated limitation)"
print("all protocol tests passed")
