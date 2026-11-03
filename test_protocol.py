"""Negative and boundary tests for the protocol implementation.
Run: python test_protocol.py"""
import os, numpy as np, torch
import secagg as sa, fl

P = sa.P


def refused(f, *a):
    try: f(*a); return False
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

# --- 4. auxiliary-node release enforcement
sid = os.urandom(16)
def fresh(K=10, minr=None):
    AN = sa.AuxNode(sid, range(K), minr or (K + 1) // 2)
    AN.open_attempt(1, 0, 8)
    for c in range(K): AN.receive_seed(sid, 1, 0, c, os.urandom(32))
    return AN
AN = fresh(); AN.mask_sum(sid, 1, 0, list(range(10)))
assert refused(AN.mask_sum, sid, 1, 0, list(range(9))), "second release in the same attempt"
assert refused(AN.mask_sum, sid, 2, 0, list(range(9))), "fabricated round accepted with old seeds"
assert refused(AN.open_attempt, 1, 0, 8), "attempt id reused"
assert refused(AN.open_attempt, 1, 1, 8), "second attempt beyond the limit"
AN.open_attempt(2, 0, 8)
assert refused(AN.mask_sum, sid, 2, 0, list(range(9))), "release without seeds of the new round"
assert refused(AN.open_attempt, 1, 5, 8), "stale round accepted"
AN = fresh(K=10, minr=5)
assert refused(AN.mask_sum, sid, 1, 0, [0] * 5), "duplicate identities accepted"
assert refused(AN.mask_sum, sid, 1, 0, [0, 1, 2, 3, 99]), "unknown identity accepted"
assert refused(AN.mask_sum, os.urandom(16), 1, 0, [0, 1, 2, 3, 4]), "wrong session accepted"
assert refused(AN.receive_seed, sid, 1, 0, 3, os.urandom(32)), "duplicate seed accepted"
assert refused(AN.receive_seed, sid, 1, 0, 42, os.urandom(32)), "unknown client seed accepted"
assert fl.SecureRound(5).min_release == 3, "ceil(K/2) for K = 5"
AN = fresh()
AN.mask_sketches(sid, 1, 0, b"D", os.urandom(32), 4, 16)
assert refused(AN.mask_sketches, sid, 1, 0, b"D", os.urandom(32), 4, 16), "second mask-sketch release"
# durable state: a restarted AN refuses the released attempt
import tempfile
pth = os.path.join(tempfile.mkdtemp(), "an_state.jsonl")
AN = sa.AuxNode(sid, range(4), 2, pth); AN.open_attempt(7, 0, 8)
for c in range(4): AN.receive_seed(sid, 7, 0, c, os.urandom(32))
AN.mask_sum(sid, 7, 0, [0, 1, 2])
AN2 = sa.AuxNode(sid, range(4), 2, pth)
assert refused(AN2.open_attempt, 7, 0, 8) and refused(AN2.open_attempt, 6, 0, 8), "state lost on restart"

# --- 5. a client answers each (session, round, attempt, index) once (retry leakage)
c0, c1 = sa.Client(0), sa.Client(1)
c0.setup_pairwise({1: c1.pk}); c1.setup_pairwise({0: c0.pk})
c0.mask_update(sa.encode(np.array([0.5, -0.25]), 1.0), sid, 1)
V1, V2 = sa.prg_field(os.urandom(32), 2), sa.prg_field(os.urandom(32), 2)
c0.ip_share(V1, [0, 1], sid, 1, 0, 0)
assert refused(c0.ip_share, V2, [0, 1], sid, 1, 0, 0), "second answer under the same identifiers"
assert refused(c0.ip_share, V2, [0, 0, 1], sid, 1, 1, 0), "duplicate accepted set"
assert refused(c0.ip_share, V2, [1, 2], sid, 1, 1, 0), "answer for a set without the client"
# pairwise masks differ across attempt ids and challenge indices
c1.mask_update(sa.encode(np.array([0.1, 0.2]), 1.0), sid, 1)
a = c1.ip_share(V1, [0, 1], sid, 1, 0, 0); b = c1.ip_share(V1, [0, 1], sid, 1, 0, 1)
assert a != b, "pairwise mask does not depend on the attempt id"

# --- 6. scope of the share check: cancelling share offsets pass on an HONEST aggregate (they do not change it)
sec = fl.SecureRound(4)
ups = [torch.full((5,), 0.1 * (k + 1)) for k in range(4)]
agg_ref, _ = fl.SecureRound(4).run(ups, [10.0] * 4, 1, "secagg", 1, 2.0)
calls = {"n": 0}
def cancel(U, V):
    return 1                                           # client 0 adds +1 ...
orig = sa.verify_aggregate
def verify_with_cancel(U, V, shares):
    shares = list(shares); shares[1] = (shares[1] - 1) % P   # ... client 1 adds -1
    return orig(U, V, shares)
sa.verify_aggregate = verify_with_cancel
agg, st = sec.run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, collude_share=cancel)
sa.verify_aggregate = orig
assert agg is not None and torch.allclose(agg, agg_ref), "cancelling offsets should pass and leave the aggregate unchanged"
# a single wrong share aborts (denial of service, no forgery)
sec = fl.SecureRound(4)
agg, st = sec.run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, collude_share=lambda U, V: 1)
assert agg is None and st["reason"] == "verification"
# a colluding client covers an AS error e = U - true_sum by adding +<e, V> (sign as in the paper)
state = {}
orig_unmask = sa.AggServer.unmask
def unmask_and_forge(self, A, M):
    U = orig_unmask(self, A, M); state["true"] = U
    e = np.zeros(U.size, np.uint64); e[2] = np.uint64(1); return sa.fadd(U, e)
sa.AggServer.unmask = unmask_and_forge
sec = fl.SecureRound(4)
agg, st = sec.run(ups, [10.0] * 4, 1, "secagg", 1, 2.0, collude_share=lambda U, V: sa.fdot(sa.fsub(U, state["true"]), V))
sa.AggServer.unmask = orig_unmask
assert agg is not None, "AS error with a +<e,V> colluding share passes (stated limitation)"
print("all protocol tests passed")
