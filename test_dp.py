"""Checks the closed-form per-example gradients / clipping used for DP-SGD against
PyTorch autograd (torch.func.vmap over per-example grad) — the mechanism Opacus uses."""
import torch, numpy as np
from torch.func import functional_call, vmap, grad
import fl

torch.manual_seed(0)
n, d_in, d_h, C = 64, 768, 16, 6
X = torch.randn(n, d_in); y = torch.randint(0, C, (n,)); cw = torch.ones(C)
head = fl.Head(d_in, d_h, C, seed=1)
P = [p.clone() for p in head.params()]
Xb, hb, d1, d2, nrm, loss = fl.per_example_grads(P, X, y, cw, 0.0, torch.Generator().manual_seed(0))
clip = 1.0; c = torch.clamp(clip / (nrm + 1e-12), max=1.0)
mine = fl.sum_grads(Xb, hb, d1, d2, c)

params = {"W1": P[0], "b1": P[1], "W2": P[2], "b2": P[3]}
def f(pr, x, t):
    z = torch.relu(x @ pr["W1"] + pr["b1"]) @ pr["W2"] + pr["b2"]
    return torch.nn.functional.cross_entropy(z[None], t[None], reduction="sum")
g = vmap(grad(f), in_dims=(None, 0, 0))(params, X, y)
ref_norm = torch.sqrt(sum((g[k].reshape(n, -1) ** 2).sum(1) for k in g))
ref_c = torch.clamp(clip / (ref_norm + 1e-12), max=1.0)
ref_sum = [(g[k] * ref_c.view(-1, *[1] * (g[k].dim() - 1))).sum(0) for k in ("W1", "b1", "W2", "b2")]
err_norm = (nrm - ref_norm).abs().max().item()
err_sum = max((a - b).abs().max().item() for a, b in zip(mine, ref_sum))
print(f"max |per-example norm - autograd| = {err_norm:.2e};  max |clipped-sum - autograd| = {err_sum:.2e}")
assert err_norm < 1e-4 and err_sum < 1e-4
print("DP-SGD per-example gradient/clipping check passed")
