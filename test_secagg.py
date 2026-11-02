"""Unit tests for the prime-field primitives (exact comparison with Python integers)."""
import numpy as np, os, secagg as sa
P = sa.P; rng = np.random.default_rng(0)
a = (rng.integers(0, 2**62, 5000, dtype=np.uint64) % np.uint64(P)); b = (rng.integers(0, 2**62, 5000, dtype=np.uint64) % np.uint64(P))
a[:3] = [P - 1, 0, 1]; b[:3] = [P - 1, P - 1, P - 1]
ref = [(int(x) * int(y)) % P for x, y in zip(a, b)]
assert all(int(z) == r for z, r in zip(sa.fmul(a, b), ref)), "fmul"
assert sa.fdot(a, b) == sum(ref) % P, "fdot"
assert all(int(z) == (int(x) + int(y)) % P for z, x, y in zip(sa.fadd(a, b), a, b)), "fadd"
assert all(int(z) == (int(x) - int(y)) % P for z, x, y in zip(sa.fsub(a, b), a, b)), "fsub"
d = 3001; cs = sa.CountSketch(os.urandom(32), d, 4, 16); u = sa.prg_field(os.urandom(32), d)
S = cs(u); M = cs.matrix().astype(np.int64)
ref = [(sum(int(M[i, j]) * int(u[j]) for j in range(d) if M[i, j] != 0)) % P for i in range(M.shape[0])]
assert all(int(x) == r for x, r in zip(S, ref)), "countsketch"
v = rng.normal(0, 0.3, d); assert np.allclose(sa.decode(sa.encode(v, 2.0)), v, atol=1e-7), "encode/decode"
try: sa.encode(np.array([3.0]), 2.0); raise AssertionError("range")
except ValueError: pass
# soundness of the linear test against errors of high 2-adic valuation (multiples of 2^63)
miss = 0
for t in range(1000):
    V = sa.prg_field(os.urandom(32), d); e = np.zeros(d, np.uint64); e[5] = np.uint64(1 << 60)
    miss += (sa.fdot(e, V) == 0)
assert miss == 0
print("all prime-field unit tests passed")
