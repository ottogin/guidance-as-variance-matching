#!/usr/bin/env python
"""fig:story: 1,024 MLP denoisers per training-set size on the 2-D mixture, each on its own N points and seed.

    python reproduce/toys/story2d.py train --optim adam --task k   # k = 0 .. 6: N = 64 * 2^k (one GPU, ~10 min)
    python reproduce/toys/story2d.py train --optim sgd --task k    # the same training sets and seeds, SGD
    python reproduce/toys/story2d.py test                          # CPU, seconds: posterior, training, recording

Data: ten Gaussians N(mu_k, s^2 I), s = 0.15, means evenly spaced in angle on an ellipse with semi-axes 2 and 1
(App. B.2), VE noise x_t = x0 + sigma eps. Denoisers: EDM preconditioning around an MLP with four hidden layers
of 128 SiLU units; its input is c_in x and sin / cos features of c_noise = ln(sigma) / 4 at eight frequencies, and
its last layer starts at zero. sigma_data = sqrt(tr Cov(x0) / 2). EDM loss with ln sigma ~ N(-1.0, 1.4^2), 40,000
steps of batch 512, Adam (lr 2e-3, betas (0.9, 0.99)) or SGD (momentum 0.9, lr 0.1), a 500-step linear warm-up
and cosine decay. Model r of size index k trains on the N points of seed 10^7 (k + 1) + r, so Adam and SGD see the
same training sets; all models of one task share the training seed 31 + k and train side by side as one batched
ensemble.

At each sigma in SIGMAS, 256 inputs x_t ~ p_sigma (seed 1, shared by every N and optimizer) go to every model.
Recorded per task, in $VMG_RUNS/_toys/story2d/<optim>-n<N>.npz: xt_<sigma> [256, 2], pred_<sigma> [1024, 256, 2]
(the models' predictions) and ideal_<sigma> [1024, 256, 2] (the exact posterior mean of each model's training set).
"""
import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

K_COMP, S_COMP = 10, 0.15
_ANG = 2 * math.pi * np.arange(K_COMP) / K_COMP
MEANS = np.stack([2.0 * np.cos(_ANG), 1.0 * np.sin(_ANG)], 1)                 # [K, 2]
VAR_X0 = float(np.trace(np.cov(MEANS.T, bias=True)) + 2 * S_COMP ** 2)        # tr Cov(x0)
SIGMA_DATA = math.sqrt(VAR_X0 / 2)
NS = [64, 128, 256, 512, 1024, 2048, 4096]
SIGMAS = [0.1, 0.2, 0.4, 0.8, 1.6]
MODELS, INPUTS = 1024, 256
P_MEAN, P_STD, NFREQ = -1.0, 1.4, 8
TRAIN = dict(steps=40000, batch=512, width=128, depth=4, warmup=500)
LR = {"adam": 2e-3, "sgd": 0.1}


def out_dir():
    from vmg import runs
    return runs.root() / "_toys" / "story2d"


# ------------------------------------------------------------------ the mixture and its exact posterior
def sample_data(n, gen):
    k = torch.randint(0, K_COMP, (n,), generator=gen)
    return torch.tensor(MEANS, dtype=torch.float32)[k] + S_COMP * torch.randn(n, 2, generator=gen)


def posterior(xt, sigma):
    """p(x0 | x_t) for x_t [B, 2]: component weights [B, K], component means [B, K, 2], the shared variance."""
    xt = np.asarray(xt, dtype=np.float64)
    v = S_COMP ** 2 + sigma ** 2
    logw = -0.5 * ((xt[:, None] - MEANS[None]) ** 2).sum(-1) / v
    w = np.exp(logw - logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return w, MEANS[None] + (S_COMP ** 2 / v) * (xt[:, None] - MEANS[None]), S_COMP ** 2 * sigma ** 2 / v


def posterior_mean_cov(xt, sigma):
    """mu = E[x0 | x_t] [B, 2] and Sigma = Cov(x0 | x_t) [B, 2, 2]."""
    w, cm, cvar = posterior(xt, sigma)
    mu = (w[..., None] * cm).sum(1)
    dev = cm - mu[:, None]
    return mu, np.einsum("bk,bki,bkj->bij", w, dev, dev) + cvar * np.eye(2)[None]


def eval_inputs():
    """{sigma: the 256 inputs x_t ~ p_sigma [256, 2]}, one generator (seed 1) drawn through SIGMAS in order."""
    g = torch.Generator().manual_seed(1)
    return {s: (sample_data(INPUTS, g) + s * torch.randn(INPUTS, 2, generator=g)).numpy().astype(np.float64)
            for s in SIGMAS}


def training_sets(k, n, models):
    """[models, n, 2]: model r of size index k trains on the points of seed 10^7 (k + 1) + r."""
    return torch.stack([sample_data(n, torch.Generator().manual_seed(10_000_000 * (k + 1) + r))
                        for r in range(models)])


# ------------------------------------------------------------------ batched denoisers (EDM preconditioning)
def precond(sigma):
    s2, d2 = sigma ** 2, SIGMA_DATA ** 2
    return d2 / (s2 + d2), sigma * SIGMA_DATA / torch.sqrt(s2 + d2), 1 / torch.sqrt(s2 + d2), torch.log(sigma) / 4


class Models:
    """M stacked MLPs: (c_in x, sin / cos features of c_noise) -> width^depth (SiLU) -> 2."""

    def __init__(self, M, width, depth, gen, device):
        dims = [2 + 2 * NFREQ] + [width] * depth + [2]
        self.P = []
        for i in range(len(dims) - 1):
            W = torch.zeros(M, dims[i], dims[i + 1], device=device) if i == len(dims) - 2 else \
                torch.randn(M, dims[i], dims[i + 1], generator=gen, device=device) / math.sqrt(dims[i])
            self.P += [W.requires_grad_(), torch.zeros(M, 1, dims[i + 1], device=device).requires_grad_()]
        self.freqs = (2.0 ** torch.arange(NFREQ, device=device)) * math.pi / 4

    def F(self, x, cn):                                            # x [M, B, 2], cn [M, B, 1]
        h = torch.cat([x, torch.sin(cn * self.freqs), torch.cos(cn * self.freqs)], -1)
        for i in range(0, len(self.P) - 2, 2):
            h = torch.nn.functional.silu(h @ self.P[i] + self.P[i + 1])
        return h @ self.P[-2] + self.P[-1]

    def D(self, x, sigma):                                         # sigma [M, B, 1]
        cs, co, ci, cn = precond(sigma)
        return cs * x + co * self.F(ci * x, cn.expand(*x.shape[:-1], 1))


def train(X, optim, seed, device, steps, batch, width, depth, warmup, log=5000):
    """Train one model per training set X [M, N, 2] with the EDM loss; returns the Models."""
    M, N = X.shape[:2]
    g = torch.Generator(device=device).manual_seed(seed)
    net = Models(M, width, depth, g, device)
    lr = LR[optim]
    opt = torch.optim.Adam(net.P, lr=lr, betas=(0.9, 0.99)) if optim == "adam" else \
        torch.optim.SGD(net.P, lr=lr, momentum=0.9)
    for k in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, k / warmup) * 0.5 * (1 + math.cos(math.pi * k / steps))
        idx = torch.randint(0, N, (M, batch), generator=g, device=device)
        x0 = torch.gather(X, 1, idx[..., None].expand(-1, -1, 2))
        sigma = torch.exp(P_MEAN + P_STD * torch.randn(M, batch, 1, generator=g, device=device))
        xt = x0 + sigma * torch.randn(M, batch, 2, generator=g, device=device)
        lam = (sigma ** 2 + SIGMA_DATA ** 2) / (sigma * SIGMA_DATA) ** 2
        loss = (lam * (net.D(xt, sigma) - x0) ** 2).sum(-1).mean(1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and (k + 1) % log == 0:
            print(f"  step {k + 1}  loss per model {loss.item() / M:.4f}", flush=True)
    return net


@torch.no_grad()
def predict(net, x, sigma, device):
    """Every model's prediction at the inputs x [P, 2] and one sigma: [M, P, 2]."""
    M = net.P[0].shape[0]
    xd = torch.as_tensor(x, dtype=torch.float32, device=device).expand(M, -1, -1)
    return net.D(xd, torch.full((M, xd.shape[1], 1), float(sigma), device=device)).cpu().numpy()


@torch.no_grad()
def ideal_means(X, xt, sigma, device, chunk=64):
    """Exact posterior means of the empirical distributions X [M, N, 2] at the inputs xt [P, 2]: [M, P, 2]."""
    Xd = torch.as_tensor(X, dtype=torch.float64, device=device)
    xd = torch.as_tensor(xt, dtype=torch.float64, device=device)
    out = []
    for r in range(0, len(Xd), chunk):
        d2 = ((xd[None, :, None] - Xd[r:r + chunk, None]) ** 2).sum(-1)                  # [chunk, P, N]
        out.append((torch.softmax(-0.5 * d2 / sigma ** 2, -1) @ Xd[r:r + chunk]).cpu().numpy())
    return np.concatenate(out)


def run(k, optim, out, models=MODELS, device=None, **over):
    """Train the models of size index k, record them at the shared inputs, write <out>/<optim>-n<N>.npz."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {**TRAIN, **over}
    n = NS[k]
    X = training_sets(k, n, models)
    t0 = time.time()
    net = train(X.to(device), optim, 31 + k, device, **cfg)
    print(f"{optim} N={n}: trained {models} models in {time.time() - t0:.0f}s", flush=True)
    rec = dict(n=n, steps=cfg["steps"], sigmas=np.array(SIGMAS))
    Xn = X.numpy().astype(np.float64)
    for s, xt in eval_inputs().items():
        rec[f"xt_{s}"] = xt
        rec[f"pred_{s}"] = predict(net, xt, s, device).astype(np.float32)
        rec[f"ideal_{s}"] = ideal_means(Xn, xt, s, device).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{optim}-n{n}.npz"
    np.savez(path, **rec)
    print(f"{path} ({time.time() - t0:.0f}s)", flush=True)
    return path


def test():
    """The exact posterior against self-normalized importance sampling, then a few training steps of a small pool
    per optimizer and their recording."""
    rng = np.random.default_rng(0)
    x0 = MEANS[rng.integers(0, K_COMP, 400000)] + S_COMP * rng.standard_normal((400000, 2))
    for s in SIGMAS:
        xt = np.array([[0.3, -0.7]])
        w = np.exp(-0.5 * ((x0 - xt) ** 2).sum(-1) / s ** 2)
        w /= w.sum()
        mu, C = posterior_mean_cov(xt, s)
        mc = w @ x0
        mcC = (w[:, None, None] * np.einsum("bi,bj->bij", x0 - mc, x0 - mc)).sum(0)
        assert np.abs(mu[0] - mc).max() < 0.02 and np.abs(C[0] - mcC).max() < 0.02, (s, mu, mc, C, mcC)
    assert eval_inputs()[0.4].tobytes() == eval_inputs()[0.4].tobytes()
    with tempfile.TemporaryDirectory() as d:
        for optim in ("adam", "sgd"):
            z = np.load(run(0, optim, Path(d), models=8, device="cpu", steps=300, batch=128, log=0))
            for s in SIGMAS:
                assert z[f"pred_{s}"].shape == z[f"ideal_{s}"].shape == (8, INPUTS, 2)
                assert np.isfinite(z[f"pred_{s}"]).all()
            X, xt = training_sets(0, NS[0], 8).numpy().astype(np.float64), z["xt_0.4"]
            d2 = ((xt[None, :, None] - X[:, None]) ** 2).sum(-1)
            p = np.exp(-0.5 * (d2 - d2.min(-1, keepdims=True)) / 0.4 ** 2)
            assert np.allclose((p / p.sum(-1, keepdims=True)) @ X, z["ideal_0.4"], atol=1e-5)
            err = ((z["pred_0.4"] - posterior_mean_cov(xt, 0.4)[0]) ** 2).sum(-1).mean()
            print(f"{optim}: mean squared error to the posterior mean at sigma 0.4 after 300 steps {err:.3f} "
                  f"(tr Cov(x0) = {VAR_X0:.3f})", flush=True)
            assert err < VAR_X0
    print("ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--optim", choices=list(LR), default="adam")
    ap.add_argument("--task", type=int, help="size index k: N = 64 * 2^k")
    ap.add_argument("--out", help="default $VMG_RUNS/_toys/story2d")
    ap.add_argument("--models", type=int, default=MODELS)
    ap.add_argument("--steps", type=int, default=TRAIN["steps"])
    a = ap.parse_args()
    if a.cmd == "test":
        return test()
    run(a.task, a.optim, Path(a.out) if a.out else out_dir(), models=a.models, steps=a.steps)


if __name__ == "__main__":
    main()
