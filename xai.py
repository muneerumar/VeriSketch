"""Explanation plausibility and faithfulness under DP (HateXplain).

Three independently trained federated heads per privacy level; explanations of the same
fixed stratified sample of posts for every model; Integrated Gradients (32 steps) with
gradient x input and random baselines; token-F1 at the oracle rationale length and at a
fixed budget estimated on the validation split; predicted-class explanations and, for one
seed, gold-class explanations; completeness residuals; a 32/64/128-step stability check;
a truncation audit.  The public analyze.py summarizes aggregate results."""
import os, sys, json, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from transformers import DistilBertTokenizerFast, DistilBertModel
from common import *
import fl

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 2)))
MD = os.path.join(ROOT, "model")
OUT = os.path.join(RES, "xai")
EPS = [None, 8.0, 2.0, 0.5]
STEPS = int(os.environ.get("IG_STEPS", 32))
N_EVAL = int(os.environ.get("N_EVAL", 300))
BINS = (0.1, 0.2, 0.5)


class FullModel(torch.nn.Module):
    def __init__(self, enc, head, mu, sd):
        super().__init__()
        self.enc, self.head = enc, head
        self.mu = torch.tensor(mu, dtype=torch.float32); self.sd = torch.tensor(sd, dtype=torch.float32)

    def forward_embeds(self, emb, mask):
        h = self.enc(inputs_embeds=emb, attention_mask=mask).last_hidden_state
        return self.head.logits((h[:, 0] - self.mu) / self.sd)


def enc_words(tok, words):
    b = tok(words, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt")
    return b, b.word_ids(0)


@torch.no_grad()
def probs_for(model, tok, word_lists):
    b = tok([w if len(w) else ["[UNK]"] for w in word_lists], is_split_into_words=True, truncation=True,
            max_length=128, padding=True, return_tensors="pt")
    emb = model.enc.embeddings.word_embeddings(b["input_ids"])
    return torch.softmax(model.forward_embeds(emb, b["attention_mask"]), 1).numpy()


def attributions(model, tok, words, target, steps):
    """Integrated Gradients (midpoint rule) and gradient×input, aggregated to words."""
    b, wid = enc_words(tok, words); ids, mask = b["input_ids"], b["attention_mask"]
    E = model.enc.embeddings.word_embeddings
    with torch.no_grad():
        x = E(ids); base = ids.clone(); base[0, 1:-1] = tok.pad_token_id; x0 = E(base)
        f_x = model.forward_embeds(x, mask)[0, target].item(); f_0 = model.forward_embeds(x0, mask)[0, target].item()
    al = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    path = (x0 + al[:, None, None] * (x - x0)).requires_grad_(True)
    model.forward_embeds(path, mask.expand(steps, -1))[:, target].sum().backward()
    ig = ((x - x0)[0] * path.grad.mean(0)).sum(-1).detach().numpy()
    xg = x.clone().requires_grad_(True)
    model.forward_embeds(xg, mask)[0, target].backward()
    gxi = (xg.grad[0] * x[0]).sum(-1).detach().numpy()
    def to_words(a):
        w = np.zeros(len(words))
        for t, j in enumerate(wid):
            if j is not None: w[j] += a[t]
        return w
    return to_words(ig), to_words(gxi), dict(residual=float(ig.sum() - (f_x - f_0)), delta=float(f_x - f_0))


def topk_mask(a, k):
    m = np.zeros(len(a), int); m[np.argsort(-a)[:k]] = 1; return m

def tf1(pred, gold):
    tp = int((pred & gold).sum()); p = tp / max(pred.sum(), 1); r = tp / max(gold.sum(), 1)
    return 2 * p * r / (p + r + 1e-12)


def evaluate(model, tok, df, budget_frac, rng, gold_too=False):
    rows = []; t0 = time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        if i % 25 == 0: print(f"    {i}/{len(df)} {time.time()-t0:.0f}s", flush=True)
        words, rat = list(r.tokens), np.array(r.rationale); n = len(words)
        p = probs_for(model, tok, [words])[0]; pred = int(p.argmax())
        targets = [("pred", pred)] + ([("gold", int(r.label))] if gold_too and int(r.label) != pred else [])
        rec = dict(post_id=r.post_id, label=int(r.label), pred=pred, correct=int(pred == r.label), n_words=n,
                   n_rat=int(rat.sum()), targets=r.targets)
        for tname, tgt in targets:
            ig, gxi, comp = attributions(model, tok, words, tgt, STEPS)
            rnd = rng.random(n)
            k_or, k_fix = int(rat.sum()), max(1, int(round(budget_frac * n)))
            for aname, a in (("ig", ig), ("gxi", gxi), ("random", rnd)):
                rec[f"{tname}_{aname}_tf1_oracle"] = tf1(topk_mask(a, k_or), rat)
                rec[f"{tname}_{aname}_tf1_budget"] = tf1(topk_mask(a, k_fix), rat)
                rec[f"{tname}_{aname}_auprc"] = average_precision_score(rat, a) if 0 < rat.sum() < n else np.nan
            if tname == "pred":
                order = np.argsort(-ig); variants = []
                for bf in BINS:
                    kk = max(1, int(round(bf * n))); keep = set(order[:kk].tolist())
                    variants += [[w for j, w in enumerate(words) if j not in keep], [w for j, w in enumerate(words) if j in keep]]
                pv = probs_for(model, tok, variants)
                rec["comprehensiveness"] = float(np.mean([p[tgt] - pv[2 * j, tgt] for j in range(len(BINS))]))
                rec["sufficiency"] = float(np.mean([p[tgt] - pv[2 * j + 1, tgt] for j in range(len(BINS))]))
                rec.update(ig_residual=comp["residual"], ig_delta=comp["delta"])
        if not gold_too or int(r.label) == pred:     # gold == pred: copy
            for aname in ("ig", "gxi", "random"):
                for m in ("tf1_oracle", "tf1_budget", "auprc"):
                    if f"pred_{aname}_{m}" in rec and gold_too: rec[f"gold_{aname}_{m}"] = rec[f"pred_{aname}_{m}"]
        rows.append(rec)
    return pd.DataFrame(rows)


def step_stability(model, tok, df, n=40):
    out = []
    for _, r in df.head(n).iterrows():
        words = list(r.tokens); tgt = int(probs_for(model, tok, [words])[0].argmax())
        A = {s: attributions(model, tok, words, tgt, s) for s in (32, 64, 128)}
        out.append(dict(rho_32_128=spearmanr(A[32][0], A[128][0]).correlation if len(words) > 2 else np.nan,
                        rho_64_128=spearmanr(A[64][0], A[128][0]).correlation if len(words) > 2 else np.nan,
                        **{f"abs_res_{s}": abs(A[s][2]["residual"]) for s in A},
                        **{f"rel_res_{s}": abs(A[s][2]["residual"]) / (abs(A[s][2]["delta"]) + 1e-9) for s in A}))
    return pd.DataFrame(out)


def main():
    from exp import BASE
    cfgb = json.load(open(CONFIG_FILE))
    d = load_hx("cls"); df = d["df"]; mu, sd = d["norm"]
    tok = DistilBertTokenizerFast.from_pretrained(MD)
    assert tok("hello world")["input_ids"] == [101, 7592, 2088, 102]
    enc = DistilBertModel.from_pretrained(MD).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    pool = df[(df.split == "test") & (df.label > 0) & df.rationale.notna()].reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(np.arange(len(pool)), train_size=N_EVAL, random_state=0, stratify=pool.label.values)
    ev = pool.iloc[np.sort(keep)].reset_index(drop=True)
    val = df[(df.split == "val") & (df.label > 0) & df.rationale.notna()]
    budget = float(np.mean([np.mean(r) for r in val.rationale]))          # fixed rationale budget from validation
    lens = [len(tok(list(w), is_split_into_words=True)["input_ids"]) for w in ev.tokens]
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(n_eval=len(ev), pool=len(pool), budget_fraction=budget, max_subword_len=int(max(lens)),
                   truncated_posts=int(sum(l > 128 for l in lens)), steps=STEPS),
              open(os.path.join(OUT, "xai_meta.json"), "w"), indent=1)
    order = EPS[::-1] if os.environ.get("XAI_REVERSE") else EPS
    for eps in order:
        for s in SEEDS[:3]:
            tag = f"eps{eps if eps else 'inf'}_s{s}"
            pth = os.path.join(OUT, tag + "_cls.json")
            if not os.path.exists(pth):
                c = dict(BASE); c.update(cfgb); c.update(seed=s, agg="secagg", partition="iid", clip_update=cfgb["clip_update"])
                if eps: c["lr"] = cfgb["dp_lr"]; c["dp"] = dict(epsilon=eps, delta=1e-5, clip=cfgb["dp_clip"], q=cfgb["dp_q"])
                out, head, pred, prob = fl.run_fl(c, *d["train"], *d["val"], *d["test"], log=lambda *a: None)
                torch.save(head.params(), os.path.join(OUT, tag + "_head.pt")); save_json(out, pth)
                print(tag, out["test"]["macro_f1"], flush=True)
            xp = os.path.join(OUT, tag + "_xai.csv")
            if os.path.exists(xp) or os.path.exists(xp + ".lock"): continue
            open(xp + ".lock", "w").close()
            head = fl.Head(768, cfgb["d_hidden"], 3, 0); head.W1, head.b1, head.W2, head.b2 = torch.load(os.path.join(OUT, tag + "_head.pt"))
            model = FullModel(enc, head, mu, sd)
            res = evaluate(model, tok, ev, budget, np.random.default_rng(s), gold_too=(s == SEEDS[0]))
            res.to_csv(xp, index=False); os.remove(xp + ".lock")
            print(tag, "done", res[["pred_ig_tf1_oracle", "pred_ig_tf1_budget", "comprehensiveness"]].mean().round(4).to_dict(), flush=True)
            if s == SEEDS[0] and eps is None:
                step_stability(model, tok, ev).to_csv(os.path.join(OUT, "ig_step_stability.csv"), index=False)


if __name__ == "__main__":
    main()
