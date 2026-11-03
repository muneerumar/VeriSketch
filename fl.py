"""Federated training of the classification head on frozen DistilBERT features,
with record-level DP-SGD (Poisson sampling, per-example clipping via exact
per-example gradient norms for the 2-layer head), Byzantine attacks,
plaintext robust aggregators, secure aggregation and VeriSketch."""
import time, json, math, os
import numpy as np, torch
from scipy.stats import norm as _norm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import secagg as sa

import os as _os
torch.set_num_threads(int(_os.environ.get("TORCH_THREADS", 2)))
DT = torch.float32


# ============================================================================ model
class Head:
    """h = ReLU(x W1 + b1); dropout; logits = h W2 + b2 (DistilBERT classifier head)."""
    def __init__(self, d_in, d_h, C, seed):
        g = torch.Generator().manual_seed(seed)
        self.shapes = [(d_in, d_h), (d_h,), (d_h, C), (C,)]
        self.W1 = torch.randn(d_in, d_h, generator=g) * 0.02
        self.b1 = torch.zeros(d_h)
        self.W2 = torch.randn(d_h, C, generator=g) * 0.02
        self.b2 = torch.zeros(C)

    def params(self): return [self.W1, self.b1, self.W2, self.b2]
    def flat(self): return torch.cat([p.reshape(-1) for p in self.params()])
    def set_flat(self, v):
        v = torch.as_tensor(v, dtype=DT); o = 0; ps = []
        for s in self.shapes:
            n = int(np.prod(s)); ps.append(v[o:o + n].reshape(s).clone()); o += n
        self.W1, self.b1, self.W2, self.b2 = ps
    @property
    def dim(self): return sum(int(np.prod(s)) for s in self.shapes)

    def logits(self, X):
        return torch.relu(X @ self.W1 + self.b1) @ self.W2 + self.b2


def per_example_grads(P, X, y, cw, drop, g):
    """Returns (dW1_i via factors, ...) -> per-example norms and the factors."""
    W1, b1, W2, b2 = P
    z1 = X @ W1 + b1; a1 = torch.relu(z1)
    if drop > 0:
        m = (torch.rand(a1.shape, generator=g) > drop).to(DT) / (1 - drop)
    else:
        m = torch.ones_like(a1)
    h = a1 * m
    z2 = h @ W2 + b2
    p = torch.softmax(z2, 1)
    oh = torch.nn.functional.one_hot(y, z2.shape[1]).to(DT)
    w = cw[y].unsqueeze(1)
    loss = -(w.squeeze(1) * torch.log(p.gather(1, y[:, None]).squeeze(1) + 1e-12))
    d2 = w * (p - oh)
    d1 = (d2 @ W2.T) * m * (z1 > 0).to(DT)
    nsq = (X * X).sum(1) * (d1 * d1).sum(1) + (d1 * d1).sum(1) + (h * h).sum(1) * (d2 * d2).sum(1) + (d2 * d2).sum(1)
    return X, h, d1, d2, nsq.sqrt(), loss


def sum_grads(X, h, d1, d2, c):
    c = c.unsqueeze(1)
    return [X.T @ (c * d1), (c * d1).sum(0), h.T @ (c * d2), (c * d2).sum(0)]


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [torch.zeros_like(p) for p in params]; self.v = [torch.zeros_like(p) for p in params]; self.t = 0
    def step(self, params, grads):
        self.t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            self.m[i].mul_(self.b1).add_(gr, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(gr, gr, value=1 - self.b2)
            mh = self.m[i] / (1 - self.b1 ** self.t); vh = self.v[i] / (1 - self.b2 ** self.t)
            p.sub_(self.lr * mh / (vh.sqrt() + self.eps))


class OSRandom:
    """Poisson sampling and Gaussian noise drawn from the operating-system CSPRNG (os.urandom).
    Uniforms use 53 random bits; normals use the Box-Muller transform in float64.  This removes the
    ordinary-PRNG state from the DP mechanism.  It does not address floating-point attacks on Gaussian
    sampling (Mironov, 2012); a deployment would use a discrete Gaussian sampler instead."""
    @staticmethod
    def uniform(n):
        b = np.frombuffer(os.urandom(8 * n), dtype=np.uint64) >> np.uint64(11)
        return (b.astype(np.float64) + 0.5) * (2.0 ** -53)

    def poisson_mask(self, n, q):
        return torch.from_numpy(self.uniform(n) < q)

    def normal(self, shape):
        n = int(np.prod(shape)); m = (n + 1) // 2
        u1, u2 = self.uniform(m), self.uniform(m); r = np.sqrt(-2.0 * np.log(u1))
        z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])[:n]
        return torch.from_numpy(z.reshape(shape)).to(DT)


def local_train(head, X, y, cfg, g, dp=None, noise_rng=None):
    """Returns the local update Delta = w_local - w_global (flat float tensor).
    With DP, Poisson sampling and noise come from noise_rng (default OSRandom); g drives dropout only.
    The sum of clipped gradients is divided by the expected batch q*n, with n the client's public capacity."""
    w0 = head.flat().clone()
    P = [p.clone() for p in head.params()]
    opt = Adam(P, cfg["lr"])
    n = len(y); cw = cfg["class_weights"]
    if dp is None:
        B = cfg["batch"]
        for _ in range(cfg["local_epochs"]):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, B):
                idx = perm[s:s + B]
                Xb, hb, d1, d2, _, _ = per_example_grads(P, X[idx], y[idx], cw, cfg["dropout"], g)
                gr = sum_grads(Xb, hb, d1, d2, torch.ones(len(idx)))
                opt.step(P, [x / len(idx) for x in gr])
    else:
        q, sigma, C = dp["q"], dp["sigma"], dp["clip"]
        rs = noise_rng or OSRandom()
        steps = int(round(cfg["local_epochs"] / q))
        exp_b = q * n
        for _ in range(steps):
            idx = torch.nonzero(rs.poisson_mask(n, q)).squeeze(1)              # Poisson sampling
            if len(idx) > 0:
                Xb, hb, d1, d2, nrm, _ = per_example_grads(P, X[idx], y[idx], cw, cfg["dropout"], g)
                c = torch.clamp(C / (nrm + 1e-12), max=1.0)
                gr = sum_grads(Xb, hb, d1, d2, c)
            else:
                gr = [torch.zeros_like(p) for p in P]
            if sigma > 0:
                z = rs.normal((sum(x.numel() for x in gr),)) * (sigma * C); o = 0; noisy = []
                for x in gr: noisy.append(x + z[o:o + x.numel()].reshape(x.shape)); o += x.numel()
                gr = noisy
            opt.step(P, [x / exp_b for x in gr])
    tmp = Head.__new__(Head); tmp.shapes = head.shapes; tmp.W1, tmp.b1, tmp.W2, tmp.b2 = P
    return tmp.flat() - w0


# ============================================================================ evaluation
def evaluate(head, X, y, C):
    with torch.no_grad():
        lg = head.logits(X); pr = torch.softmax(lg, 1).numpy(); pred = pr.argmax(1)
    y = y.numpy()
    th, ph = y != 0, pred != 0
    tmr = float((th & ~ph).sum() / max(th.sum(), 1)); far = float((~th & ph).sum() / max((~th).sum(), 1))
    return dict(accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                tmr=tmr, far=far, per_class_f1=f1_score(y, pred, average=None, labels=list(range(C))).tolist()), pred, pr


# ============================================================================ attacks
def malicious_update(kind, own, benign, f, K, g):
    if kind == "signflip":
        return -4.0 * own
    if kind == "signflip_nm":                       # norm-matched (adaptive to norm checks)
        ref = torch.stack(benign).norm(dim=1).median()
        return -own / own.norm() * ref
    if kind == "gauss":
        ref = torch.stack(benign).norm(dim=1).median()
        v = torch.randn(own.shape, generator=g); return v / v.norm() * 10 * ref
    if kind in ("alie", "ipm"):
        Bm = torch.stack(benign); mu = Bm.mean(0)
        if kind == "ipm": return -0.5 * mu
        s = math.floor(K / 2 + 1) - f
        z = float(_norm.ppf((K - s) / K))
        return mu - z * Bm.std(0)
    raise ValueError(kind)

def filter_aware_attack(benign, f, K, clip_u, fixed=False):
    """Adaptive attack tuned to the VeriSketch selection rule: an omniscient attacker moves the
    benign mean along the ALIE direction by the LARGEST step that the (exact-distance) adaptive
    rule still accepts for all f malicious copies."""
    Bm = torch.stack(benign); mu = Bm.mean(0); dirn = -torch.sign(mu) * Bm.std(0)
    for lam in (16, 12, 8, 6, 4, 3, 2, 1.5, 1.0, 0.5):
        m = mu + lam * dirn
        if m.norm() > clip_u: m = m * (clip_u / m.norm())
        U = torch.stack(list(benign) + [m] * f)
        sel = adaptive_select(U.norm(dim=1).numpy(), _pdist(U).numpy(), f, clip_u, fixed=fixed)
        if all(len(benign) + j in sel for j in range(f)): return m, lam
    return mu, 0.0


ROTATE = {1: 2, 2: 3, 3: 4, 4: 5, 5: 1}

def poison_labels(y, kind):
    y = y.clone(); harm = y != 0
    if kind == "flip_evasion": y[harm] = 0
    elif kind == "flip_rotate": y[harm] = torch.tensor([ROTATE[int(v)] for v in y[harm]], dtype=y.dtype)
    return y


# ============================================================================ aggregators (plaintext)
def _pdist(U):
    G = U @ U.T; sq = torch.diag(G)
    return (sq[:, None] + sq[None, :] - 2 * G).clamp_min(0)

def krum_scores(D, f):
    K = D.shape[0]; k = max(K - f - 2, 1)
    s = []
    for i in range(K):
        d = torch.cat([D[i, :i], D[i, i + 1:]]); s.append(torch.sort(d).values[:k].sum())
    return torch.stack(s)

def aggregate_plain(name, ups, w, f):
    U = torch.stack(ups); w = torch.as_tensor(w, dtype=DT)
    if name == "fedavg": return (w[:, None] * U).sum(0) / w.sum(), list(range(len(ups)))
    if name == "median": return U.median(0).values, list(range(len(ups)))
    if name == "trimmed":
        b = f; S = torch.sort(U, 0).values; return S[b:len(ups) - b].mean(0), list(range(len(ups)))
    sc = krum_scores(_pdist(U), f)
    if name == "krum":
        i = int(torch.argmin(sc)); return U[i], [i]
    if name == "mkrum":
        sel = torch.argsort(sc)[: len(ups) - f].tolist()
        return (w[sel, None] * U[sel]).sum(0) / w[sel].sum(), sel
    raise ValueError(name)


# ============================================================================ secure pipelines
def adaptive_select(norms, D, f, clip_u, norm_tau=1.5, mad_c=3.0, fixed=False):
    """The VeriSketch selection rule, applicable to sketch-estimated OR exact norms/distances."""
    K = len(norms)
    ok = [i for i in range(K) if norms[i] <= norm_tau * clip_u]
    if len(ok) < 3: return ok
    Dm = D[np.ix_(ok, ok)]
    sc = krum_scores(torch.tensor(Dm), f).numpy()
    if fixed:   # Multi-Krum semantics: K - f clients overall; clients already rejected by the norm test count against f
        f_left = max(f - (K - len(ok)), 0)
        return [ok[j] for j in np.argsort(sc)[: max(len(ok) - f_left, 1)]]
    med = np.median(sc); mad = np.median(np.abs(sc - med)) + 1e-12
    return [ok[j] for j in range(len(ok)) if sc[j] <= med + mad_c * 1.4826 * mad]


class SecureRound:
    """One session of masked aggregation over F_P (SecAgg-FedAvg, VeriSketch or VeriSketch-MK) with real
    cryptography.  The AN object and its release state persist for the whole session.  run() executes one
    attempt of one round and returns (aggregate or None if the round aborts, statistics).  The default
    max_attempts=1 means an aborted round is not retried: the model stays unchanged and the next round
    starts from new local updates."""
    PROTOCOL_ABORTS = ("receipt", "coin opening", "sketch self-check", "verification", "range", "sketch consistency")

    def __init__(self, K, rows=4, width=64, sid=None, state_path=None, max_attempts=1):
        self.K, self.rows, self.width = K, rows, width
        self.sid = sid or os.urandom(16)
        self.clients = [sa.Client(i) for i in range(K)]
        pks = {c.cid: c.pk for c in self.clients}
        for c in self.clients: c.setup_pairwise(pks)
        self.min_release = (K + 1) // 2                                    # ceil(K/2)
        self.AN = sa.AuxNode(self.sid, range(K), self.min_release, state_path, max_attempts)

    @staticmethod
    def int_bounds(w, clip_u):
        """Public per-client integer bounds B_k = ceil(w_k C_u' 2^24), C_u' = C_u (1 + 1e-6)."""
        return [sa.int_bound(float(np.float64(wi) * np.float64(clip_u * (1 + 1e-6)))) for wi in w]

    def run(self, ups, n, rnd, mode, f, clip_u, mad_c=3.0, norm_tau=1.5, tamper=None,
            sigma_mode="post_commit", sub=None, malicious=(), collude_share=None, kappa=1, att=0, capture=False):
        K, d = self.K, ups[0].numel(); sid = self.sid; tam = tamper or {}
        AS, AN = sa.AggServer(), self.AN
        nmax = float(max(n)); w = [float(x) / nmax for x in n]          # public weights in (0, 1]
        B = self.int_bounds(w, clip_u)
        st = dict(t_client_mask=0.0, t_sketch=0.0, t_unmask=0.0, t_verify=0.0, aborted=False, reason="", abort_kind="")
        def abort(reason, kind="protocol check"):
            st.update(aborted=True, reason=reason, abort_kind=kind, accepted=[]); return None, st
        h1, h2 = AN.open_attempt(rnd, att, d)                              # AN coin commitments (attempt start)
        t = time.time()
        for c, u, wi, Bi in zip(self.clients, ups, w, B):
            v = np.float64(wi) * u.detach().double().numpy()             # weighting in float64
            if c.cid in malicious:      # Byzantine clients may send any field element (no range proof)
                enc = np.mod(np.rint(np.nan_to_num(v) * sa.SCALE).astype(np.int64), sa.P).astype(np.uint64)
            else:
                enc = sa.encode(v, Bi / sa.SCALE)
            x, seed = c.mask_update(enc, sid, rnd, att)
            AS.receive(c.cid, x); AN.receive_seed(sid, rnd, att, c.cid, seed)
        st["t_client_mask"] = time.time() - t
        tr = AS.transcript(dict(enumerate(w)))
        if tam.get("party") == "AS_transcript": tr = tam["fn"](tr)
        D0 = sa.digest(sid, rnd, tr, att)
        if not all(c.check_receipt(D0, tr, w[c.cid], sid, rnd, att) for c in self.clients): return abort("receipt")
        D = sa.H(D0, h1, h2)
        A = list(range(K)); S = None; cs = None
        if mode in ("verisketch", "verisketch_mk"):
            t = time.time()
            # two-party coin: the AS contributes a fresh share with D, then the AN opens its committed share
            r_as, r_an = os.urandom(32), AN.reveal_coin(0)
            if not sa.check_coin(h1, b"coin1", r_an): return abort("coin opening")
            sigma = sa.sketch_seed(sid, rnd, D, r_an + r_as, att) if sigma_mode == "post_commit" else predictable_sigma(rnd)
            cs = sa.CountSketch(sigma, sub or d, self.rows, self.width)
            if sigma_mode == "post_commit":
                MS = AN.mask_sketches(sid, rnd, att, D, r_as, self.rows, self.width, sub)
            else:                                                          # ablation: AN sketches under the public seed
                MS = {cc: cs(AN._mask(cc)[:sub] if sub else AN._mask(cc)) for cc in range(K)}
            if tam.get("party") == "AN_sketch": MS = tam["fn"](MS)
            S = AS.sketch_updates(cs, MS, sub)
            # every client checks the sketch the AS publishes for it
            bad = [c.cid for c in self.clients if not c.check_own_sketch(cs, S[c.cid]) and c.cid not in malicious]
            if bad: st["t_sketch"] = time.time() - t; return abort(f"sketch self-check failed for clients {bad}")
            F = {i: cs.to_float(S[i]) / w[i] for i in S}
            norms = np.array([sa.sketch_sqnorm(F[i]) ** 0.5 for i in range(K)])
            Dm = np.zeros((K, K))
            for a in range(K):
                for b in range(a + 1, K): Dm[a, b] = Dm[b, a] = sa.sketch_sqdist(F[a], F[b])
            A = adaptive_select(norms, Dm, f, clip_u, norm_tau, mad_c, fixed=(mode == "verisketch_mk"))
            st["t_sketch"] = time.time() - t
            st.update(sketch_norms=norms.tolist())
            if capture:                                                    # the AS's actual view (leakage study)
                st["view"] = dict(sketch=np.stack([cs.to_float(S[i]).reshape(-1) for i in range(K)]), weights=list(w))
            if len(A) < self.min_release: return abort("accepted set below release threshold", "filter")
        t = time.time()
        if tam.get("party") == "AN_query": tam["fn"](AN, sid, rnd, att, A)    # malformed-query probes
        M = AN.mask_sum(sid, rnd, att, A)
        if tam.get("party") == "AN": M = tam["fn"](M, AN, A)
        U = AS.unmask(A, M)
        if tam.get("party") == "AS": U = tam["fn"](U, AS, A)
        st["t_unmask"] = time.time() - t
        t = time.time()
        c = sa.aggregate_commitment(sid, rnd, A, U, att)                # AS commits and adds its coin share ...
        r_as, r_an = os.urandom(32), AN.reveal_coin(1)                  # ... then the AN opens its committed share
        if not sa.check_coin(h2, b"coin2", r_an): return abort("coin opening")
        ok_v = True
        for idx in range(kappa):                                        # kappa independent challenges
            V = sa.verification_vector(sid, rnd, c, r_an + r_as, d, idx, att)
            shares = [self.clients[i].ip_share(V, A, sid, rnd, idx, att) for i in A]
            if collude_share is not None:                                # limitation test: colluding client offsets
                shares[0] = (shares[0] + collude_share(U, V)) % sa.P
            ok_v = ok_v and sa.verify_aggregate(U, V, shares)
        ok_r = sa.in_range_int(U, sum(B[i] for i in A))                  # exact integer bound of the accepted set
        ok_s = True
        if S is not None:
            tot = None
            for i in A: tot = S[i] if tot is None else sa.fadd(tot, S[i])
            ok_s = bool((cs(U[:sub] if sub else U) == tot).all())
        st["t_verify"] = time.time() - t
        st.update(verified=ok_v, in_range=ok_r, sketch_consistent=ok_s, accepted=A,
                  upload_bytes_per_client=d * 8 + 32, sketch_scalars=(self.rows * self.width if S is not None else 0))
        if not (ok_v and ok_r and ok_s):
            return abort("verification" if not ok_v else ("range" if not ok_r else "sketch consistency"))
        agg = torch.tensor(sa.decode(U) * nmax / float(sum(n[i] for i in A)), dtype=DT)
        if capture: st.setdefault("view", {}).update(accepted=list(A), weights=list(w), U=sa.decode(U))
        return agg, st


def predictable_sigma(rnd):
    """Ablation only: a sketch seed that clients can compute before uploading."""
    return sa.H(b"public-round-seed", rnd.to_bytes(8, "little"))


def nullspace_attack(benign, d, rows, width, rnd, scale=10.0):
    """Sketch-aware attacker that assumes the sketch seed is predictable: sends the
    benign mean plus a large destructive component in the null space of the
    CountSketch it expects, so that its sketch equals that of the benign mean."""
    cs = sa.CountSketch(predictable_sigma(rnd), d, rows, width)
    S = np.zeros((rows * width, d))
    for j in range(rows):
        S[j * width + cs.bucket[j], np.arange(d)] = np.where(cs.neg[j], -1.0, 1.0)
    mu = torch.stack(benign).mean(0)
    z = (-scale * mu).numpy().astype(np.float64)
    v = z - S.T @ np.linalg.lstsq(S @ S.T, S @ z, rcond=None)[0]
    return mu + torch.tensor(v, dtype=DT)


# ============================================================================ driver
def dirichlet_partition(y, K, alpha, rng, min_size=50):
    C = int(y.max()) + 1
    while True:
        idx = [[] for _ in range(K)]
        for c in range(C):
            ids = np.where(y == c)[0]; rng.shuffle(ids)
            p = rng.dirichlet([alpha] * K)
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)): idx[k].extend(part.tolist())
        if min(len(i) for i in idx) >= min_size: return [np.array(i) for i in idx]

def iid_partition(n, K, rng):
    p = rng.permutation(n); return np.array_split(p, K)


def run_fl(cfg, Xtr, ytr, Xva, yva, Xte, yte, log=print):
    """cfg keys: K, T, seed, partition ('iid' | float alpha), agg, f_assumed, attack,
    n_malicious, dp (None | dict(epsilon, delta, clip, q)), lr, batch, local_epochs,
    dropout, clip_update, d_hidden, eval_every"""
    rng = np.random.default_rng(cfg["seed"]); g = torch.Generator().manual_seed(cfg["seed"])
    K, C = cfg["K"], int(ytr.max()) + 1
    parts = iid_partition(len(ytr), K, rng) if cfg["partition"] == "iid" else dirichlet_partition(ytr.numpy(), K, float(cfg["partition"]), rng)
    mal = list(range(cfg.get("n_malicious", 0)))          # malicious ids; partitions are random so ids are arbitrary
    # no data-dependent class weights (they would make every record's loss depend on all labels)
    cw = torch.ones(C, dtype=DT)
    cfg = dict(cfg); cfg["class_weights"] = cw
    head = Head(Xtr.shape[1], cfg.get("d_hidden", 768), C, cfg["seed"])
    dp = None
    if cfg.get("dp"):
        from opacus.accountants.utils import get_noise_multiplier
        q = cfg["dp"]["q"]
        if cfg["dp"].get("epsilon") is None:          # noise-free control: same sampling and clipping, sigma = 0
            sigma = 0.0
        else:
            sigma = get_noise_multiplier(target_epsilon=cfg["dp"]["epsilon"], target_delta=cfg["dp"]["delta"],
                                         sample_rate=q, steps=int(round(cfg["local_epochs"] / q)) * cfg["T"], accountant="rdp")
        dp = dict(q=q, sigma=sigma, clip=cfg["dp"]["clip"])
    sec = SecureRound(K, rows=cfg.get("rows", 4), width=cfg.get("width", 64)) if cfg["agg"] in ("secagg", "verisketch", "verisketch_mk") else None
    # DP: Poisson sampling and noise come from the OS CSPRNG (OSRandom); dropout uses a per-client generator
    # seeded from the OS (not recorded).  Such runs are not bit-reproducible.
    import secrets
    cgen = [torch.Generator().manual_seed(secrets.randbits(62)) for _ in range(K)] if dp else None
    n_aborted = 0
    hist = []; t_start = time.time(); best = (-1, None, 0)
    for r in range(1, cfg["T"] + 1):
        t0 = time.time()
        ups, ws, honest_ups = [], [], []
        for k in range(K):
            yk = ytr[parts[k]]
            if k in mal and cfg["attack"] in ("flip_evasion", "flip_rotate"): yk = poison_labels(yk, cfg["attack"])
            # client-level DP noise (if any) uses the client's own sample rate q; Poisson sampling
            dk = None
            if dp: dk = dict(dp, q=dp["q"])
            u = local_train(head, Xtr[parts[k]], yk, cfg, cgen[k] if dp else g, dk)
            nu = u.norm()
            if nu > cfg["clip_update"]: u = u * (cfg["clip_update"] / nu)   # honest update norm bound
            ups.append(u); ws.append(float(len(parts[k])))
            if k not in mal: honest_ups.append(u)
        if cfg["attack"] in ("signflip", "signflip_nm", "gauss", "alie", "ipm"):
            for k in mal: ups[k] = malicious_update(cfg["attack"], ups[k], honest_ups, len(mal), K, g)
        if cfg["attack"] in ("adaptive", "adaptive_mk"):
            mv, lam = filter_aware_attack(honest_ups, len(mal), K, cfg["clip_update"], fixed=cfg["attack"] == "adaptive_mk")
            for k in mal: ups[k] = mv
        if cfg["attack"] == "nullspace":
            mv = nullspace_attack(honest_ups, head.dim, cfg.get("rows", 4), cfg.get("width", 64), r)
            for k in mal: ups[k] = mv
        t_train = time.time() - t0
        st = {}
        if sec is not None:
            tam = cfg["_tamper"]["spec"] if cfg.get("_tamper") and cfg["_tamper"]["round"] == r else None
            agg, st = sec.run(ups, ws, r, cfg["agg"], cfg["f_assumed"], cfg["clip_update"], tamper=tam,
                              sigma_mode=cfg.get("sigma_mode", "post_commit"), malicious=set(mal),
                              sub=(head.shapes[0][0] * head.shapes[0][1] + head.shapes[1][0]) if cfg.get("sketch_layers") == "first" else None)
            sel = st["accepted"]
        elif cfg["agg"] in ("adaptive_plain", "adaptive_plain_mk"):
            U = torch.stack(ups); norms = U.norm(dim=1).numpy(); Dm = _pdist(U).numpy()
            sel = adaptive_select(norms, Dm, cfg["f_assumed"], cfg["clip_update"], fixed=cfg["agg"].endswith("_mk"))
            wsel = torch.tensor([ws[i] for i in sel], dtype=DT)
            agg = (wsel[:, None] * U[sel]).sum(0) / wsel.sum() if sel else None
        elif cfg["agg"] == "normbound":
            # functionality of a cryptographically enforced L2 norm bound (ELSA-style): aggregate-only sum of the
            # clients whose update norm is within the public bound C_u; nothing else about the updates is used
            sel = [i for i, u in enumerate(ups) if float(u.norm()) <= cfg["clip_update"] * (1 + 1e-6)]
            wsel = torch.tensor([ws[i] for i in sel], dtype=DT)
            agg = (wsel[:, None] * torch.stack([ups[i] for i in sel])).sum(0) / wsel.sum() if sel else None
        else:
            agg, sel = aggregate_plain(cfg["agg"], ups, ws, cfg["f_assumed"])
        if agg is None:
            n_aborted += 1          # round aborted: the global model is left unchanged
        else:
            head.set_flat(head.flat() + agg)
        hn = [float(u.norm()) for k, u in enumerate(ups) if k not in mal]
        rec = dict(round=r, t_train=t_train, t_round=time.time() - t0, selected=sel,
                   upd_norm_median=float(np.median(hn)), upd_norm_max=float(np.max(hn)),
                   mal_selected=[i for i in sel if i in mal])
        rec.update({k: v for k, v in st.items() if k.startswith("t_") or k in ("verified", "sketch_consistent", "in_range", "aborted", "reason", "abort_kind")})
        if r % cfg.get("eval_every", 1) == 0 or r == cfg["T"]:
            mv, _, _ = evaluate(head, Xva, yva, C); rec.update({"val_" + k: v for k, v in mv.items() if k != "per_class_f1"})
        if cfg.get("select_best_val") and rec.get("val_macro_f1", -1) > best[0]:
            best = (rec["val_macro_f1"], head.flat().clone(), r)
        if cfg.get("_record_params"): rec["param_hash"] = __import__("hashlib").sha256(head.flat().numpy().tobytes()).hexdigest()
        hist.append(rec)
        log(f"r{r:02d} {cfg['agg']:<10} val_f1={rec.get('val_macro_f1', float('nan')):.4f} sel={sel} t={rec['t_round']:.1f}s")
    if cfg.get("select_best_val") and best[1] is not None:
        head.set_flat(best[1])
    if cfg.get("eval_test", True):
        mt, pred, prob = evaluate(head, Xte, yte, C)
    else:
        mt, pred, prob = {}, None, None          # design/tuning runs never touch the test set
    out = dict(cfg={k: v for k, v in cfg.items() if k != "class_weights"}, test=mt, history=hist, n_aborted=n_aborted,
               malicious=mal, sigma=(dp["sigma"] if dp else None), wall=time.time() - t_start,
               best_round=best[2], partition_sizes=[len(p) for p in parts],
               partition_hist=[np.bincount(ytr[p].numpy(), minlength=C).tolist() for p in parts])
    return out, head, pred, prob
