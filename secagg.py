"""VeriSketch secure-aggregation primitives over the prime field F_P, P = 2^61 - 1.

  * Masking, sketching and verification run in F_P, so a random linear test detects any
    non-zero error except with probability 1/P.
  * Updates are encoded with an explicit admissible range; decoded aggregates are range-checked.
  * Every hash binds the session id, round, client ids, public weights and accepted set.
  * The sketch seed and the verification challenge come from two-party coins: the AN commits
    to its share at the start of the round and the AS adds a fresh share after committing.
  * The AS sends each client the sketch it derived for it, and the client checks it.
  * The AN releases one mask sum per round, for an accepted set of minimum size.
Trust assumptions (see the paper): the AS and the AN do not collude; integrity of the
aggregate also assumes that neither server colludes with an accepted client.  These limits
are tested in tamper.py.
"""
import os, hashlib, hmac, struct, warnings
import numpy as np

warnings.filterwarnings("ignore", message="overflow encountered")
U64 = np.uint64
P = (1 << 61) - 1
P64 = U64(P)
FRAC_BITS = 24
SCALE = float(1 << FRAC_BITS)
M32 = U64(0xFFFFFFFF)


# --------------------------------------------------------------------------- field arithmetic
def fadd(a, b):
    s = a + b
    return np.where(s >= P64, s - P64, s)

def fsub(a, b):
    return fadd(a, P64 - b)

def fneg(a):
    return np.where(a == 0, a, P64 - a)

def _red64(x):
    """x (uint64, any value) -> x mod P."""
    r = (x & P64) + (x >> U64(61))
    return np.where(r >= P64, r - P64, r)

def fmul(a, b):
    """Element-wise a*b mod P for a, b in [0, P)."""
    a1, a0 = a >> U64(32), a & M32
    b1, b0 = b >> U64(32), b & M32
    hh = a1 * b1                      # < 2^58
    mid = a1 * b0 + a0 * b1           # < 2^62
    ll = a0 * b0                      # < 2^64
    # a*b = hh*2^64 + mid*2^32 + ll ; 2^64 = 8 mod P
    t1 = _red64(hh * U64(8))
    m = _red64(mid)                   # < 2^61
    m1, m0 = m >> U64(29), m & U64((1 << 29) - 1)
    t2 = fadd(_red64(m1), _red64(m0 << U64(32)))      # m*2^32 = m1*2^61 + m0*2^32
    t3 = _red64(ll)
    return fadd(fadd(t1, t2), t3)

def fsum(x):
    """Sum of a vector of field elements, mod P (exact)."""
    x = np.asarray(x, dtype=U64)
    lo = int(np.sum(x & M32, dtype=U64)); hi = int(np.sum(x >> U64(32), dtype=U64))
    return (hi * (1 << 32) + lo) % P

def fdot(a, b):
    return fsum(fmul(a, b))


# --------------------------------------------------------------------------- encoding
def encode(v, bound):
    """float vector with |v_j| <= bound  ->  fixed-point element of F_P."""
    v = np.asarray(v, np.float64)
    if np.abs(v).max(initial=0) > bound * (1 + 1e-9):
        raise ValueError("value outside the admissible encoding range")
    q = np.rint(v * SCALE).astype(np.int64)
    return np.mod(q, P).astype(U64)

def to_signed(u):
    s = u.astype(np.int64)
    return np.where(s > P // 2, s - P, s)

def decode(u):
    return to_signed(u).astype(np.float64) / SCALE

def in_range(u, bound):
    return bool(np.abs(to_signed(u)).max(initial=0) <= int(bound * SCALE) + 1)


# --------------------------------------------------------------------------- PRG / hashing
def prg_bytes(seed32: bytes, n: int, nonce: bytes):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    enc = Cipher(algorithms.ChaCha20(seed32, nonce), mode=None).encryptor()
    return enc.update(b"\x00" * n)

def prg_field(seed32: bytes, n: int, nonce: bytes = b"\x00" * 16) -> np.ndarray:
    r = np.frombuffer(prg_bytes(seed32, 8 * n, nonce), dtype=U64) & P64
    return np.where(r == P64, U64(0), r).copy()

def H(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "little")); h.update(p)
    return h.digest()

def ints(*xs) -> bytes:
    return b"".join(int(x).to_bytes(8, "little", signed=True) for x in xs)


# --------------------------------------------------------------------------- CountSketch over F_P
class CountSketch:
    def __init__(self, sigma: bytes, d: int, rows: int, width: int):
        self.d, self.r, self.w = d, rows, width
        rnd = np.frombuffer(prg_bytes(sigma, 16 * rows * d, b"countsketch-v2\x00\x00"), dtype=U64)
        self.bucket = (rnd[: rows * d] % U64(width)).astype(np.int64).reshape(rows, d)
        self.neg = ((rnd[rows * d:] >> U64(63)) == 1).reshape(rows, d)
        self.order = [np.argsort(self.bucket[j], kind="stable") for j in range(rows)]
        self.starts = [np.searchsorted(self.bucket[j][self.order[j]], np.arange(width)) for j in range(rows)]

    def matrix(self):
        S = np.zeros((self.r * self.w, self.d))
        for j in range(self.r):
            S[j * self.w + self.bucket[j], np.arange(self.d)] = np.where(self.neg[j], -1.0, 1.0)
        return S

    def __call__(self, u: np.ndarray) -> np.ndarray:
        out = np.zeros((self.r, self.w), U64)
        for j in range(self.r):
            c = np.where(self.neg[j], fneg(u), u)[self.order[j]]
            st = self.starts[j]; ne = np.diff(np.append(st, len(c))) > 0
            if not ne.any(): continue
            lo = np.add.reduceat(c & M32, st[ne]); hi = np.add.reduceat(c >> U64(32), st[ne])
            # (hi*2^32 + lo) mod P, hi < d*2^29
            hi_r = _red64(hi); h1, h0 = hi_r >> U64(29), hi_r & U64((1 << 29) - 1)
            out[j, ne] = fadd(fadd(_red64(h1), _red64(h0 << U64(32))), _red64(lo))
        return out.reshape(-1)

    def to_float(self, s: np.ndarray) -> np.ndarray:
        return decode(s).reshape(self.r, self.w)


def sketch_sqdist(a, b):
    return float(np.median(((a - b) ** 2).sum(1)))

def sketch_sqnorm(a):
    return float(np.median((a ** 2).sum(1)))


# --------------------------------------------------------------------------- parties
class Client:
    def __init__(self, cid: int):
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        self.cid = cid
        self.sk = X25519PrivateKey.generate(); self.pk = self.sk.public_key(); self.shared = {}

    def setup_pairwise(self, pks: dict):
        for j, pk in pks.items():
            if j != self.cid: self.shared[j] = self.sk.exchange(pk)

    def mask_update(self, u: np.ndarray, sid: bytes, rnd: int):
        self.u = u
        self.seed = os.urandom(32)
        self.mask = prg_field(self.seed, u.size, nonce=struct.pack("<QQ", rnd, 1))
        self.x = fadd(u, self.mask)
        self.hx = H(self.x.tobytes())
        return self.x, self.seed

    def check_receipt(self, D: bytes, transcript: list, weight=None, sid=None, rnd=None) -> bool:
        """Own upload hash (and public weight) present in the broadcast transcript, and D is its digest."""
        mine = any(c == self.cid and h == self.hx and (weight is None or w == weight) for c, w, h in transcript)
        return mine and (sid is None or digest(sid, rnd, transcript) == D)

    def check_own_sketch(self, cs: CountSketch, published: np.ndarray) -> bool:
        return bool((cs(self.u) == published).all())

    def ip_share(self, V: np.ndarray, A, sid: bytes, rnd: int, idx: int = 0) -> int:
        ip = fdot(self.u, V)
        for j in A:
            if j == self.cid: continue
            r = int.from_bytes(hmac.new(self.shared[j], sid + struct.pack("<QQ", rnd, idx) + b"ip", "sha256").digest()[:8], "little") % P
            ip = (ip + r) % P if self.cid < j else (ip - r) % P
        return ip


class AuxNode:
    def __init__(self, min_release: int = 2):
        self.seeds = {}; self.released = set(); self.min_release = min_release

    def receive_seed(self, cid, seed, rnd, d):
        self.seeds[cid] = (seed, rnd, d)

    def _mask(self, cid):
        seed, rnd, d = self.seeds[cid]
        return prg_field(seed, d, nonce=struct.pack("<QQ", rnd, 1))

    def coin(self) -> bytes:
        return os.urandom(32)

    def commit_coins(self, rnd: int):
        """At the start of a round the AN commits to its two coin shares (sketch seed, challenge)."""
        self._coins = (os.urandom(32), os.urandom(32)); self._coin_round = rnd
        return H(b"coin1", self._coins[0]), H(b"coin2", self._coins[1])

    def reveal_coin(self, i: int) -> bytes:
        return self._coins[i]

    def mask_sketches(self, cs: CountSketch, cids, sub=None):
        return {c: cs(self._mask(c)[:sub] if sub else self._mask(c)) for c in cids}

    def mask_sum(self, rnd, cids):
        """One authorised aggregate release per round, for |A| >= min_release."""
        if rnd in self.released: raise PermissionError("mask sum already released for this round")
        if len(cids) < self.min_release: raise PermissionError("accepted set below confidentiality threshold")
        self.released.add(rnd)
        s = None
        for c in cids:
            m = self._mask(c); s = m if s is None else fadd(s, m)
        return s


class AggServer:
    def __init__(self):
        self.x = {}

    def receive(self, cid, x):
        self.x[cid] = x

    def transcript(self, weights):
        return [(c, weights[c], H(self.x[c].tobytes())) for c in sorted(self.x)]

    def sketch_updates(self, cs, mask_sk, sub=None):
        return {c: fsub(cs(self.x[c][:sub] if sub else self.x[c]), mask_sk[c]) for c in self.x}

    def unmask(self, A, mask_sum_A):
        s = None
        for c in A:
            s = self.x[c].copy() if s is None else fadd(s, self.x[c])
        return fsub(s, mask_sum_A)


def digest(sid, rnd, transcript):
    return H(sid, ints(rnd), *[ints(c) + struct.pack("<d", w) + h for c, w, h in transcript])

def sketch_seed(sid, rnd, D, rho):
    return H(b"sketch", sid, ints(rnd), D, rho)

def aggregate_commitment(sid, rnd, A, U):
    return H(b"agg", sid, ints(rnd), ints(*sorted(A)), U.tobytes())

def verification_vector(sid, rnd, c, rho2, d, idx=0):
    return prg_field(H(b"verify", sid, ints(rnd, idx), c, rho2), d, nonce=b"verify-vector-v2")


def check_coin(commitment: bytes, label: bytes, coin: bytes) -> bool:
    return H(label, coin) == commitment

def verify_aggregate(U, V, shares) -> bool:
    return fdot(U, V) == (sum(shares) % P)
