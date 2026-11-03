"""VeriSketch secure-aggregation primitives over the prime field F_P, P = 2^61 - 1.

  * Masking, sketching and verification run in F_P.  Field elements are drawn by rejection
    sampling, so for a fixed non-zero error e and a uniform challenge V, Pr[<e,V> = 0] = 1/P
    (in the ideal-PRG model; the implemented bound adds the PRG and hash advantages).
  * Inputs are quantised to integers with a public per-client integer bound; non-finite values and
    invalid bounds are rejected.  The aggregate range check uses the exact sum of these bounds.
  * A SessionConfig fixes members, dimension, sketch size, sketched prefix, number of challenges,
    release threshold, attempt limit and update bound; every authenticated message carries its hash.
  * Every hash binds the session id, round, attempt id, client ids, public weights and accepted set.
  * The sketch seed and the challenge come from two-party coins: the AN commits to its shares when
    it opens the attempt; the AS adds a share after the transcript digest and after its commitment.
  * The AN enforces the order of the steps and releases mask sketches (pinned size) and one mask sum
    per attempt.  It sends the attempt opening, the coin openings and the authorised accepted set to
    the clients over authenticated channels.
  * Clients answer only the open attempt, for the authorised set, with challenges they derive themselves,
    once.  They send their shares to both servers.  Each server gives its own authenticated verdict;
    a client applies the aggregate only if it matches the commitment, lies in the integer range and
    both verdicts accept.
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


# --------------------------------------------------------------------------- session configuration
class SessionConfig:
    """Public parameters fixed when the session is created and agreed by every party:
    member ids, update dimension, sketch rows and width, sketched prefix, number of challenges,
    release threshold, attempt limit, update bound C_u and mode.  Every authenticated message carries its hash."""
    def __init__(self, sid, members, d, rows, width, sub=None, kappa=1, min_release=1, max_attempts=1,
                 clip_u=2.0, mode="verisketch"):
        self.sid, self.members = sid, tuple(sorted(int(m) for m in members))
        self.d, self.rows, self.width, self.sub = int(d), int(rows), int(width), (int(sub) if sub else None)
        self.kappa, self.min_release, self.max_attempts = int(kappa), int(min_release), int(max_attempts)
        self.clip_u, self.mode = float(clip_u), str(mode)
        self.hash = H(b"session-config", sid, ints(self.d, self.rows, self.width, self.sub or 0, self.kappa,
                                                   self.min_release, self.max_attempts), struct.pack("<d", self.clip_u),
                      self.mode.encode(), ints(*self.members))

    def bounds(self, weights):
        """Public per-client integer bounds B_k = ceil(2^24 w_k C_u'), C_u' = C_u (1 + 1e-6)."""
        return {c: int_bound(float(np.float64(w) * np.float64(self.clip_u * (1 + 1e-6)))) for c, w in weights.items()}


def mac(key: bytes, *parts: bytes) -> bytes:
    return hmac.new(key, H(*parts), "sha256").digest()

def mac_ok(key: bytes, tag: bytes, *parts: bytes) -> bool:
    return tag is not None and hmac.compare_digest(mac(key, *parts), tag)

def _A(A) -> bytes:
    return ints(*sorted(int(a) for a in A))


# --------------------------------------------------------------------------- parties
class Client:
    """Client with per-attempt state.  It uploads only for an attempt that the AN opened (authenticated),
    checks the transcript, derives the sketch seed and the challenge itself from authenticated coin
    openings, answers only for the accepted set that the AN authorised for that attempt, and applies an
    aggregate only if it matches the commitment, lies in the exact integer range, and both servers
    report acceptance in authenticated verdicts.  Each attempt is answered once and then closed."""
    def __init__(self, cid: int):
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        self.cid = cid
        self.sk = X25519PrivateKey.generate(); self.pk = self.sk.public_key(); self.shared = {}
        self.ctx, self.closed = None, set()

    def setup_pairwise(self, pks: dict):
        for j, pk in pks.items():
            if j != self.cid: self.shared[j] = self.sk.exchange(pk)

    def register(self, cfg: SessionConfig, k_an: bytes, k_as: bytes):
        """Authenticated channels to the AN and to the AS (keys established at registration)."""
        self.cfg, self.k_an, self.k_as = cfg, k_an, k_as

    # -- attempt start (AN -> client)
    def open_attempt(self, rnd, att, h1, h2, tag):
        cfg = self.cfg
        if not mac_ok(self.k_an, tag, b"open", cfg.hash, ints(rnd, att), h1, h2): raise PermissionError("attempt not opened by the AN")
        if self.ctx is not None: self.closed.add((self.ctx["rnd"], self.ctx["att"])); self.ctx = None   # an unfinished attempt is closed
        if (rnd, att) in self.closed: raise PermissionError("attempt id already used")
        if self.closed and rnd < max(r for r, _ in self.closed): raise PermissionError("stale round")
        if sum(1 for r, _ in self.closed if r == rnd) >= cfg.max_attempts: raise PermissionError("attempt limit reached")
        self.ctx = dict(rnd=rnd, att=att, h1=h1, h2=h2, uploaded=False, answered=set())

    def _need(self, *keys):
        if self.ctx is None: raise PermissionError("no open attempt")
        for k in keys:
            if self.ctx.get(k) is None: raise PermissionError(f"protocol step missing: {k}")
        return self.ctx

    def mask_update(self, u: np.ndarray):
        ctx = self._need()
        if ctx["uploaded"]: raise PermissionError("already uploaded in this attempt")
        self.u = u; ctx["uploaded"] = True
        self.seed = os.urandom(32)
        self.mask = prg_field(self.seed, u.size, nonce=mask_nonce(ctx["rnd"], ctx["att"]))
        self.x = fadd(u, self.mask)
        self.hx = H(self.x.tobytes())
        return self.x, self.seed

    def check_receipt(self, D0: bytes, transcript: list, weight=None) -> bool:
        """Own upload hash and public weight in the broadcast transcript, and D0 is its digest."""
        ctx = self._need()
        mine = any(c == self.cid and h == self.hx and (weight is None or w == weight) for c, w, h in transcript)
        ok = mine and digest(self.cfg.sid, ctx["rnd"], transcript, ctx["att"]) == D0
        if ok:
            ctx["D"] = H(D0, ctx["h1"], ctx["h2"])
            ctx["weights"] = {c: w for c, w, _ in transcript}
        return ok

    # -- sketch seed (AN -> client) and own-sketch check
    def receive_coin1(self, r_an, r_as, tag) -> bool:
        ctx = self._need("D"); cfg = self.cfg
        if not mac_ok(self.k_an, tag, b"coin1", cfg.hash, ints(ctx["rnd"], ctx["att"]), ctx["D"], r_an, r_as): return False
        if not check_coin(ctx["h1"], b"coin1", r_an): return False
        ctx["sigma"] = sketch_seed(cfg.sid, ctx["rnd"], ctx["D"], r_an + r_as, ctx["att"])
        return True

    def check_own_sketch(self, published: np.ndarray) -> bool:
        ctx = self._need("sigma"); cfg = self.cfg
        cs = CountSketch(ctx["sigma"], cfg.sub or cfg.d, cfg.rows, cfg.width)
        return bool((cs(self.u[:cfg.sub] if cfg.sub else self.u) == published).all())

    # -- accepted set (AN -> client), commitment and challenge coin (AN -> client)
    def receive_accepted(self, A, tag) -> bool:
        ctx = self._need("D"); cfg = self.cfg
        if not mac_ok(self.k_an, tag, b"accepted", cfg.hash, ints(ctx["rnd"], ctx["att"]), _A(A)): return False
        if len(set(A)) != len(list(A)): return False
        ctx["A"] = sorted(int(a) for a in A); return True

    def receive_coin2(self, c, r_an, r_as, tag) -> bool:
        ctx = self._need("A"); cfg = self.cfg
        if not mac_ok(self.k_an, tag, b"coin2", cfg.hash, ints(ctx["rnd"], ctx["att"]), c, r_an, r_as): return False
        if not check_coin(ctx["h2"], b"coin2", r_an): return False
        ctx["c"], ctx["rho2"] = c, r_an + r_as; return True

    def shares(self):
        """kappa masked shares <u, V_i> + sum_j +-r_kj for the AN-authorised set, with the challenges derived
        locally from the commitment and the authenticated coin.  Answered once per attempt."""
        ctx = self._need("A", "c"); cfg = self.cfg
        if self.cid not in ctx["A"]: raise PermissionError("client not in the authorised set")
        if ctx["answered"]: raise PermissionError("challenge already answered in this attempt")
        ctx["answered"].add(True)
        out = []
        for idx in range(cfg.kappa):
            V = verification_vector(cfg.sid, ctx["rnd"], ctx["c"], ctx["rho2"], self.u.size, idx, ctx["att"])
            ip = fdot(self.u, V)
            for j in ctx["A"]:
                if j == self.cid: continue
                r = int.from_bytes(hmac.new(self.shared[j], cfg.sid + struct.pack("<QQQ", ctx["rnd"], ctx["att"], idx) + b"ip",
                                            "sha256").digest()[:8], "little") % P
                ip = (ip + r) % P if self.cid < j else (ip - r) % P
            out.append(ip)
        hdr = (cfg.hash, ints(ctx["rnd"], ctx["att"], self.cid), ctx["c"], ints(*out))
        return out, mac(self.k_an, b"share", *hdr), mac(self.k_as, b"share", *hdr)

    # -- final decision
    def accept(self, U, verdict_an, verdict_as):
        """(accepted, reason).  The client checks the commitment and the range itself and requires an
        authenticated acceptance from both servers; the attempt is closed either way."""
        ctx = self._need("A", "c"); cfg = self.cfg; key = ints(ctx["rnd"], ctx["att"])
        self.closed.add((ctx["rnd"], ctx["att"])); self.ctx = None
        if aggregate_commitment(cfg.sid, ctx["rnd"], ctx["A"], U, ctx["att"]) != ctx["c"]: return False, "commitment (clients)"
        B = cfg.bounds(ctx["weights"])
        if not in_range_int(U, sum(B[i] for i in ctx["A"])): return False, "range (check ii, clients)"
        for name, k, v in (("AN", self.k_an, verdict_an), ("AS", self.k_as, verdict_as)):
            ok, why, tag = v if v is not None else (False, "missing", None)
            if not mac_ok(k, tag, b"verdict", cfg.hash, key, ctx["c"], bytes([bool(ok)]), why.encode()):
                return False, f"{name} verdict not authentic"
            if not ok: return False, f"{name} verdict: {why}"
        return True, ""


class AuxNode:
    """Auxiliary node with a pinned session configuration and per-session release state.

    For every (round, attempt) the AN enforces the order: open -> seeds -> digest and first AS coin share ->
    first coin opening -> mask sketches (once, under its own seed derivation and the pinned rows, width and
    prefix) -> one mask sum for an authorised set -> commitment and second AS coin share -> second coin
    opening -> authenticated shares from the accepted clients -> verdict.  Rounds must not decrease and
    attempt ids are never reused.  With state_path the released records are appended to a file."""
    def __init__(self, cfg: SessionConfig, keys=None, state_path=None):
        self.cfg, self.sid, self.members = cfg, cfg.sid, frozenset(cfg.members)
        self.min_release, self.max_attempts, self.state_path = cfg.min_release, cfg.max_attempts, state_path
        self.keys = keys or {}
        self.opened, self.released, self.sketched = set(), {}, set()
        self.last_round, self.cur, self.seeds, self.d, self.ph = None, None, {}, cfg.d, {}
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

    def _tags(self, *parts):
        return {c: mac(k, *parts) for c, k in self.keys.items()}

    def open_attempt(self, rnd: int, att: int, d: int = None):
        if d is not None and int(d) != self.cfg.d: raise PermissionError("dimension differs from the session configuration")
        if self.last_round is not None and rnd < self.last_round: raise PermissionError("stale round")
        if (rnd, att) in self.opened: raise PermissionError("attempt id already used")
        if sum(1 for (r, _) in self.opened if r == rnd) >= self.max_attempts: raise PermissionError("attempt limit reached")
        self.opened.add((rnd, att)); self.last_round = rnd; self.cur = (rnd, att); self.seeds = {}; self.ph = {}
        self._persist(dict(round=rnd, attempt=att))
        self._coins = (os.urandom(32), os.urandom(32))
        h1, h2 = H(b"coin1", self._coins[0]), H(b"coin2", self._coins[1])
        self.ph["h"] = (h1, h2)
        self.open_tags = self._tags(b"open", self.cfg.hash, ints(rnd, att), h1, h2)
        return h1, h2

    def _check(self, sid, rnd, att):
        if sid != self.sid: raise PermissionError("wrong session")
        if self.cur != (rnd, att): raise PermissionError("round/attempt not open")

    def receive_seed(self, sid, rnd, att, cid, seed):
        self._check(sid, rnd, att)
        if "D" in self.ph: raise PermissionError("seed after the transcript was fixed")
        if cid not in self.members: raise PermissionError("unknown client")
        if cid in self.seeds: raise PermissionError("duplicate seed")
        self.seeds[cid] = seed

    def receive_digest(self, sid, rnd, att, D0, r_as1):
        """Digest of the transcript and the first AS coin share (sketch seed)."""
        self._check(sid, rnd, att)
        if "D" in self.ph: raise PermissionError("digest already received")
        h1, h2 = self.ph["h"]; self.ph["D"] = H(D0, h1, h2); self.ph["r_as1"] = r_as1

    def _mask(self, cid):
        rnd, att = self.cur
        return prg_field(self.seeds[cid], self.d, nonce=mask_nonce(rnd, att))

    def reveal_coin(self, i: int) -> bytes:
        """Coin openings in protocol order: the first after the digest, the second after the commitment."""
        if i == 0:
            if "D" not in self.ph: raise PermissionError("coin opening before the transcript digest")
            self.ph["open1"] = True
            rnd, att = self.cur
            self.coin1_tags = self._tags(b"coin1", self.cfg.hash, ints(rnd, att), self.ph["D"], self._coins[0], self.ph["r_as1"])
        elif i == 1:
            if "c" not in self.ph: raise PermissionError("coin opening before the aggregate commitment")
            self.ph["open2"] = True
            rnd, att = self.cur
            self.coin2_tags = self._tags(b"coin2", self.cfg.hash, ints(rnd, att), self.ph["c"], self._coins[1], self.ph["r_as2"])
        else:
            raise PermissionError("unknown coin")
        return self._coins[i]

    def mask_sketches(self, sid, rnd, att, rows=None, width=None, sub=None):
        """Mask sketches for every client with a stored seed, once per attempt, under the AN's own seed
        derivation and the pinned sketch configuration (other sizes are refused)."""
        self._check(sid, rnd, att); cfg = self.cfg
        if cfg.mode == "secagg": raise PermissionError("no sketches in a SecAgg session")
        if (rows or cfg.rows, width or cfg.width, sub if sub is not None else cfg.sub) != (cfg.rows, cfg.width, cfg.sub):
            raise PermissionError("sketch configuration differs from the session configuration")
        if not self.ph.get("open1"): raise PermissionError("mask sketches before the sketch seed is fixed")
        if (rnd, att) in self.sketched: raise PermissionError("mask sketches already released")
        self.sketched.add((rnd, att))
        sigma = sketch_seed(sid, rnd, self.ph["D"], self._coins[0] + self.ph["r_as1"], att)
        cs = CountSketch(sigma, cfg.sub or self.d, cfg.rows, cfg.width)
        return {c: cs(self._mask(c)[:cfg.sub] if cfg.sub else self._mask(c)) for c in sorted(self.seeds)}

    def mask_sum(self, sid, rnd, att, A):
        """One authorised mask-sum release per attempt; the authorised set is sent to the clients."""
        self._check(sid, rnd, att)
        A = [int(a) for a in A]
        if (rnd, att) in self.released: raise PermissionError("mask sum already released for this attempt")
        if "D" not in self.ph: raise PermissionError("mask sum before the transcript digest")
        if self.cfg.mode != "secagg" and (rnd, att) not in self.sketched: raise PermissionError("mask sum before the filtering step")
        if len(set(A)) != len(A): raise PermissionError("duplicate client in the accepted set")
        if not set(A) <= self.members: raise PermissionError("unknown client in the accepted set")
        if not set(A) <= set(self.seeds): raise PermissionError("no seed for a client in the accepted set")
        if len(A) < self.min_release: raise PermissionError("accepted set below confidentiality threshold")
        dig = H(b"accepted", sid, ints(rnd, att), _A(A)).hex()
        self.released[(rnd, att)] = dig; self.ph["A"] = sorted(A)
        self._persist(dict(round=rnd, attempt=att, released=dig))
        self.accepted_tags = self._tags(b"accepted", self.cfg.hash, ints(rnd, att), _A(A))
        s = None
        for c in A:
            m = self._mask(c); s = m if s is None else fadd(s, m)
        return s

    def receive_commitment(self, sid, rnd, att, c, r_as2):
        self._check(sid, rnd, att)
        if "A" not in self.ph: raise PermissionError("commitment before the mask-sum release")
        if "c" in self.ph: raise PermissionError("commitment already received")
        self.ph["c"], self.ph["r_as2"] = c, r_as2

    def verdict(self, U, shares: dict):
        """The AN's own acceptance decision: commitment and challenge identity with the shares that the
        accepted clients sent to it over authenticated channels.  Returns {client: (ok, reason, tag)}."""
        ph, cfg = self.ph, self.cfg
        if not ph.get("open2"): raise PermissionError("verdict before the challenge is fixed")
        if "verdict" in ph: raise PermissionError("verdict already given")
        rnd, att = self.cur
        ok, why = _check_shares(cfg, rnd, att, ph["A"], ph["c"], self._coins[1] + ph["r_as2"], U, shares, self.keys)
        ph["verdict"] = (ok, why)
        return {c: (ok, why, mac(k, b"verdict", cfg.hash, ints(rnd, att), ph["c"], bytes([ok]), why.encode()))
                for c, k in self.keys.items()}


def _check_shares(cfg, rnd, att, A, c, rho2, U, shares, keys):
    """Commitment and challenge identity, using only shares with valid MACs from the members of A."""
    if aggregate_commitment(cfg.sid, rnd, A, U, att) != c: return False, "commitment"
    tot = [0] * cfg.kappa
    for k in A:
        s = shares.get(k)
        if s is None: return False, "missing share"
        vals, tag = s
        if len(vals) != cfg.kappa or not mac_ok(keys[k], tag, b"share", cfg.hash, ints(rnd, att, k), c, ints(*vals)):
            return False, "share not authentic"
        tot = [(a + b) % P for a, b in zip(tot, vals)]
    for idx in range(cfg.kappa):
        V = verification_vector(cfg.sid, rnd, c, rho2, U.size, idx, att)
        if fdot(U, V) != tot[idx]: return False, "challenge (check i)"
    return True, ""


class AggServer:
    """Aggregation server.  Its own verdict (used against a deviating AN) checks the challenge with the
    shares it receives from the clients, the exact range and the sketch consistency."""
    def __init__(self, cfg: SessionConfig = None, keys=None):
        self.x, self.cfg, self.keys = {}, cfg, keys or {}

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

    def verdict(self, rnd, att, A, c, rho2, U, shares, bound, cs=None, S=None):
        cfg = self.cfg
        ok, why = _check_shares(cfg, rnd, att, A, c, rho2, U, shares, self.keys)
        if ok and not in_range_int(U, bound): ok, why = False, "range (check ii)"
        if ok and S is not None:
            tot = None
            for i in A: tot = S[i] if tot is None else fadd(tot, S[i])
            if not bool((cs(U[:cfg.sub] if cfg.sub else U) == tot).all()): ok, why = False, "sketch consistency (check iii)"
        return {k: (ok, why, mac(key, b"verdict", cfg.hash, ints(rnd, att), c, bytes([ok]), why.encode())) for k, key in self.keys.items()}


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
