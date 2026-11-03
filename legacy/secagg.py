"""VeriSketch secure-aggregation primitives over the prime field F_P, P = 2^61 - 1.

  * Masking, sketching and verification run in F_P.  Field elements are drawn by rejection
    sampling, so for a fixed non-zero error e and a uniform challenge V, Pr[<e,V> = 0] = 1/P
    (in the ideal-PRG model; the implemented bound adds the PRG and hash advantages).
  * Inputs are quantised to integers with a public per-client integer bound; non-finite values and
    invalid bounds are rejected.  The aggregate range check uses the exact sum of these bounds.
  * Every hash binds the session id, round, attempt id, client ids, public weights and accepted set.
  * The sketch seed and the verification challenge come from two-party coins: the AN commits
    to its share at the start of the attempt and the AS adds a fresh share after committing.
  * The AS sends each client the sketch it derived for it, and the client checks it.
  * The AN keeps release state per session: it accepts seeds only for the open (round, attempt),
    releases mask sketches once per attempt under a seed it derives itself, and releases one mask
    sum per attempt for a duplicate-free set of registered clients of size at least ceil(K/2).
    Stale rounds and reused attempt ids are refused.  State can be written to a file.
  * A client answers each (session, round, attempt, challenge index) at most once.
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
def int_bound(bound) -> int:
    """Public integer bound for inputs with |v_j| <= bound: ceil(bound * 2^24).
    rint(v * 2^24) never exceeds it, because IEEE rounding is monotone and 2^24 is exact."""
    b = float(bound)
    if not np.isfinite(b) or b <= 0: raise ValueError("invalid encoding bound")
    B = int(np.ceil(b * SCALE))
    if B >= P // 4: raise ValueError("encoding bound too large for the field")
    return B


def encode(v, bound):
    """float vector with |v_j| <= bound  ->  fixed-point element of F_P (exact integer bound int_bound(bound))."""
    B = int_bound(bound)
    v = np.asarray(v, np.float64)
    if not np.isfinite(v).all():
        raise ValueError("non-finite value")
    q = np.rint(v * SCALE)
    if np.abs(q).max(initial=0) > B:
        raise ValueError("value outside the admissible encoding range")
    return np.mod(q.astype(np.int64), P).astype(U64)

def to_signed(u):
    s = u.astype(np.int64)
    return np.where(s > P // 2, s - P, s)

def decode(u):
    return to_signed(u).astype(np.float64) / SCALE

def in_range_int(u, B: int) -> bool:
    """|signed(u_j)| <= B for every coordinate (B an exact integer bound)."""
    return bool(np.abs(to_signed(u)).max(initial=0) <= B)


# --------------------------------------------------------------------------- PRG / hashing
def prg_bytes(seed32: bytes, n: int, nonce: bytes):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    enc = Cipher(algorithms.ChaCha20(seed32, nonce), mode=None).encryptor()
    return enc.update(b"\x00" * n)

def prg_field(seed32: bytes, n: int, nonce: bytes = b"\x00" * 16) -> np.ndarray:
    """n uniform elements of F_P: 61-bit words from the ChaCha20 stream, the single value P rejected.
    The stream prefix does not depend on the requested length, so the output is deterministic."""
    m = n + 16
    while True:
        r = np.frombuffer(prg_bytes(seed32, 8 * m, nonce), dtype=U64) & P64
        r = r[r != P64]
        if len(r) >= n: return r[:n].copy()
        m *= 2

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

    def mask_update(self, u: np.ndarray, sid: bytes, rnd: int, att: int = 0):
        self.u = u
        self.seed = os.urandom(32)
        self.mask = prg_field(self.seed, u.size, nonce=mask_nonce(rnd, att))
        self.x = fadd(u, self.mask)
        self.hx = H(self.x.tobytes())
        return self.x, self.seed

    def check_receipt(self, D: bytes, transcript: list, weight=None, sid=None, rnd=None, att=0) -> bool:
        """Own upload hash (and public weight) present in the broadcast transcript, and D is its digest."""
        mine = any(c == self.cid and h == self.hx and (weight is None or w == weight) for c, w, h in transcript)
        return mine and (sid is None or digest(sid, rnd, transcript, att) == D)

    def check_own_sketch(self, cs: CountSketch, published: np.ndarray) -> bool:
        return bool((cs(self.u) == published).all())

    def ip_share(self, V: np.ndarray, A, sid: bytes, rnd: int, idx: int = 0, att: int = 0, kappa_max: int = 4) -> int:
        """Masked share <u, V> + sum_j +-r_kj.  Each (sid, round, attempt, index) is answered once, so two
        challenges can never be answered under the same pairwise masks."""
        key = (sid, rnd, att, idx)
        if not hasattr(self, "answered"): self.answered = set()
        if key in self.answered or not 0 <= idx < kappa_max:
            raise PermissionError("challenge already answered or index out of range")
        if self.cid not in A or len(set(A)) != len(list(A)):
            raise PermissionError("malformed accepted set")
        self.answered.add(key)
        ip = fdot(self.u, V)
        for j in A:
            if j == self.cid: continue
            r = int.from_bytes(hmac.new(self.shared[j], sid + struct.pack("<QQQ", rnd, att, idx) + b"ip", "sha256").digest()[:8], "little") % P
            ip = (ip + r) % P if self.cid < j else (ip - r) % P
        return ip


class AuxNode:
    """Auxiliary node with per-session release state.

    The AN is created once per session with the registered client ids.  For every (round, attempt) it
      * accepts one seed per registered client, only for the attempt that is open;
      * releases mask sketches once, under the seed H(sketch, sid, t, a, D, rho_AN || rho_AS) it derives itself;
      * releases one mask sum, for a duplicate-free set of registered clients with stored seeds and
        |A| >= ceil(K/2); the digest of the authorised set is recorded.
    Rounds must not decrease and attempt ids are never reused.  With state_path the released
    (round, attempt, set digest) records are appended to a file and reloaded on restart."""
    def __init__(self, sid: bytes, members, min_release: int, state_path=None, max_attempts: int = 1):
        self.sid, self.members = sid, frozenset(int(m) for m in members)
        self.min_release, self.max_attempts, self.state_path = int(min_release), int(max_attempts), state_path
        self.opened, self.released, self.sketched = set(), {}, set()
        self.last_round, self.cur, self.seeds, self.d = None, None, {}, None
        if state_path and os.path.exists(state_path):
            import json
            for line in open(state_path):
                r = json.loads(line); key = (r["round"], r["attempt"])
                self.opened.add(key); self.last_round = max(self.last_round or 0, r["round"])
                if r.get("released"): self.released[key] = r["released"]

    def _persist(self, rec):
        if self.state_path:
            import json
            with open(self.state_path, "a") as f: f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())

    def open_attempt(self, rnd: int, att: int, d: int):
        if self.last_round is not None and rnd < self.last_round: raise PermissionError("stale round")
        if (rnd, att) in self.opened: raise PermissionError("attempt id already used")
        if sum(1 for (r, _) in self.opened if r == rnd) >= self.max_attempts: raise PermissionError("attempt limit reached")
        self.opened.add((rnd, att)); self.last_round = rnd; self.cur = (rnd, att); self.seeds = {}; self.d = int(d)
        self._persist(dict(round=rnd, attempt=att))
        # coin shares for this attempt (sketch seed, challenge); commitments go to the AS
        self._coins = (os.urandom(32), os.urandom(32))
        return H(b"coin1", self._coins[0]), H(b"coin2", self._coins[1])

    def _check(self, sid, rnd, att):
        if sid != self.sid: raise PermissionError("wrong session")
        if self.cur != (rnd, att): raise PermissionError("round/attempt not open")

    def receive_seed(self, sid, rnd, att, cid, seed):
        self._check(sid, rnd, att)
        if cid not in self.members: raise PermissionError("unknown client")
        if cid in self.seeds: raise PermissionError("duplicate seed")
        self.seeds[cid] = seed

    def _mask(self, cid):
        rnd, att = self.cur
        return prg_field(self.seeds[cid], self.d, nonce=mask_nonce(rnd, att))

    def reveal_coin(self, i: int) -> bytes:
        return self._coins[i]

    def mask_sketches(self, sid, rnd, att, D, rho_as, rows, width, sub=None):
        """Mask sketches for every client with a stored seed, once per attempt, under the AN's own seed derivation."""
        self._check(sid, rnd, att)
        if (rnd, att) in self.sketched: raise PermissionError("mask sketches already released")
        self.sketched.add((rnd, att))
        sigma = sketch_seed(sid, rnd, D, self._coins[0] + rho_as, att)
        cs = CountSketch(sigma, sub or self.d, rows, width)
        return {c: cs(self._mask(c)[:sub] if sub else self._mask(c)) for c in sorted(self.seeds)}

    def mask_sum(self, sid, rnd, att, A):
        """One authorised mask-sum release per attempt."""
        self._check(sid, rnd, att)
        A = [int(a) for a in A]
        if (rnd, att) in self.released: raise PermissionError("mask sum already released for this attempt")
        if len(set(A)) != len(A): raise PermissionError("duplicate client in the accepted set")
        if not set(A) <= self.members: raise PermissionError("unknown client in the accepted set")
        if not set(A) <= set(self.seeds): raise PermissionError("no seed for a client in the accepted set")
        if len(A) < self.min_release: raise PermissionError("accepted set below confidentiality threshold")
        dig = H(b"accepted", sid, ints(rnd, att), ints(*sorted(A))).hex()
        self.released[(rnd, att)] = dig
        self._persist(dict(round=rnd, attempt=att, released=dig))
        s = None
        for c in A:
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


def mask_nonce(rnd, att):
    return struct.pack("<QQ", rnd, (att << 8) | 1)

def digest(sid, rnd, transcript, att=0):
    return H(sid, ints(rnd, att), *[ints(c) + struct.pack("<d", w) + h for c, w, h in transcript])

def sketch_seed(sid, rnd, D, rho, att=0):
    return H(b"sketch", sid, ints(rnd, att), D, rho)

def aggregate_commitment(sid, rnd, A, U, att=0):
    return H(b"agg", sid, ints(rnd, att), ints(*sorted(A)), U.tobytes())

def verification_vector(sid, rnd, c, rho2, d, idx=0, att=0):
    return prg_field(H(b"verify", sid, ints(rnd, att, idx), c, rho2), d, nonce=b"verify-vector-v3")


def check_coin(commitment: bytes, label: bytes, coin: bytes) -> bool:
    return H(label, coin) == commitment

def verify_aggregate(U, V, shares) -> bool:
    return fdot(U, V) == (sum(shares) % P)
