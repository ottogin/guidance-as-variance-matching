"""The paper's two-dimensional example (App. "Two-dimensional example", fig:ellipse), with exact denoisers.

Data: K = 20 isotropic Gaussians (std 0.15) whose means are evenly spaced in angle on an ellipse with
semi-axes 4 and 1; the class of a point is its component. D1 = E[x0 | x_t, c] and D0 = E[x0 | x_t] in closed
form, so nothing is trained and everything runs on a CPU in seconds. Noise levels are the SD 3 flow-matching
grid (16 Euler steps, shift 3), written in the variance-exploding form x_t = x0 + sigma eps; the first level is
sigma = inf, where x_t carries no information and D1, D0 are the class mean and the data mean.

Because both predictors are exact, the moments have closed forms the tests check: a(t) = 0, w_MSE(t) = 1 and
w(t) = sqrt(V / (V - V_c)) with V = E tr Cov(x0 | x_t) and V_c = E tr Cov(x0 | x_t, c).
"""
import numpy as np
import torch

from .. import noise
from .base import Model, Spec, Split

CALIB0, EVAL0 = 0, 1_000_000           # row ids: calibration and evaluation draws never overlap


def flow_grid(steps, shift=3.0):
    """(sigma, t) of the SD 3 flow-matching Euler grid; sigma = t / (1 - t), inf at t = 1."""
    u = np.linspace(1, 1 / steps, steps)
    t = shift * u / (1 + (shift - 1) * u)
    with np.errstate(divide="ignore"):
        return np.where(t >= 1, np.inf, t / (1 - t)), t


class Toy(Model):
    batch = 4096

    def __init__(self, K=20, ax=4.0, ay=1.0, std=0.15, steps=16, shift=3.0, n_calib=8192, n_eval=8192):
        th = 2 * np.pi * np.arange(K) / K
        self.mu = torch.tensor(np.stack([ax * np.cos(th), ay * np.sin(th)], 1), dtype=torch.float32)
        self.K, self.s2 = K, std * std
        self.sig, self.t = flow_grid(steps, shift)
        self.spec = Spec(
            key="toy", name="2-D mixture", modality="toy", checkpoint="closed form",
            reference="E[x0 | x_t] (the unconditional posterior mean)",
            sampler=f"DDIM (Euler), SD 3 flow grid, shift {shift:g}", steps=steps, output="2-D point",
            published_w=float("nan"), published_note="no published weight",
            datasets={"mixture": Split(calib=f"{n_calib} draws from the mixture", n_calib=n_calib,
                                       eval=f"{n_eval} other draws", n_eval=n_eval, reference="those draws",
                                       seed="per row")})

    # --- calibration -------------------------------------------------------------------------------------
    def levels(self):
        return dict(sigma=self.sig.copy(), t=self.t.copy())

    def rows(self, dataset="mixture", split="calib"):
        sp = self.spec.datasets[dataset]
        return list(range(CALIB0, CALIB0 + sp.n_calib)) if split == "calib" else \
            list(range(EVAL0, EVAL0 + sp.n_eval))

    def draw(self, rows):
        """(x0 [B, 2], class [B]) for these rows, each row from its own generator."""
        xs, cs = [], []
        for r in rows:
            g = noise.generator(self.seed, r)
            c = int(torch.randint(self.K, (1,), generator=g))
            xs.append(self.mu[c] + np.sqrt(self.s2) * torch.randn(2, generator=g))
            cs.append(c)
        return torch.stack(xs), torch.tensor(cs)

    def data(self, dataset, rows):
        return self.draw(rows)

    def noise(self, x0, k, eps):
        return eps.clone() if np.isinf(self.sig[k]) else x0 + float(self.sig[k]) * eps

    def denoise(self, x, c, sig):
        """Exact (D1, D0) at VE level `sig`."""
        if np.isinf(sig):
            return self.mu[c], self.mu.mean(0).expand(len(c), 2)
        v = self.s2 + sig * sig
        dlt = x[:, None, :] - self.mu[None]
        m = self.mu[None] + (self.s2 / v) * dlt                      # [B, K, 2] per-component posterior means
        g = torch.softmax(-0.5 * (dlt ** 2).sum(-1) / v, 1)          # [B, K] component posterior
        return m[torch.arange(len(x)), c], (g[..., None] * m).sum(1)

    def denoise_pair(self, x, k, cond):
        return self.denoise(x, cond, float(self.sig[k]))

    def posterior_trace(self, x, sig):
        """(tr Cov(x0 | x_t), tr Cov(x0 | x_t, c) for the posterior's own c): the closed forms the tests use."""
        v = self.s2 + sig * sig
        within = 2 * self.s2 * sig * sig / v
        dlt = x[:, None, :] - self.mu[None]
        m = self.mu[None] + (self.s2 / v) * dlt
        g = torch.softmax(-0.5 * (dlt ** 2).sum(-1) / v, 1)
        mbar = (g[..., None] * m).sum(1, keepdim=True)
        between = (g * ((m - mbar) ** 2).sum(-1)).sum(1)
        return within + between, torch.full_like(between, within)

    def smoke(self):
        """D0 must be the class average of D1 under the class posterior (the tower property)."""
        x0, c = self.draw(range(256))
        for k in range(1, len(self.sig)):
            x = self.noise(x0, k, torch.randn_like(x0))
            v = self.s2 + float(self.sig[k]) ** 2
            g = torch.softmax(-0.5 * ((x[:, None, :] - self.mu[None]) ** 2).sum(-1) / v, 1)
            avg = sum(g[:, j, None] * self.denoise(x, torch.full_like(c, j), float(self.sig[k]))[0]
                      for j in range(self.K))
            gap = float((avg - self.denoise(x, c, float(self.sig[k]))[1]).abs().max())
            assert gap < 1e-4, f"level {k}: class-averaged D1 differs from D0 by {gap}"
        print("toy smoke OK: D0 is the posterior average of D1 at every level")

    # --- sampling ----------------------------------------------------------------------------------------
    def sample(self, dataset, rows, w, outdir):
        """Guided DDIM from the rows' own initial noise; writes samples.npz (x [R, 2], class [R], w [S])."""
        S = len(self.sig)
        ws = np.full(S, 1.0) if w is None else np.broadcast_to(np.asarray(w, dtype=np.float64), (S,))
        _, c = self.draw(rows)
        z = noise.eps_batch(self.seed, rows, 0, (2,))
        x, lv = z.clone(), list(self.sig) + [0.0]
        for j in range(S):
            s, sn = lv[j], lv[j + 1]
            d1, d0 = self.denoise(x, c, s)
            dw = d0 + float(ws[j]) * (d1 - d0)
            x = dw + sn * z if np.isinf(s) else (sn / s) * x + (1 - sn / s) * dw
        np.savez(f"{outdir}/samples.npz", x=x.numpy(), c=c.numpy(), w=ws, rows=np.asarray(rows))
        return x
