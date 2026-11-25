"""Optional sums a moments pass can collect besides the standard ones (compute_w.py --extras NAME,...).

Each extra sees every batch's x0, D0 and D1 at every calibrated level and keeps sums over rows that merge across
shards by addition. They cost no extra network evaluations.

  chancov   raw channel Grams per level, G[S, 3C, 3C] = sum over rows and positions of u u^T with u the stacked
            channels of (x0, D0 - x0, D1 - x0), each [C, P]. With the standard first moments s_x and s_r they
            give every C x C covariance of fig:covmatch, centred afterwards (vmg.moments.residual_cov). It is what
            compute_w.py --match cov needs.
  diag      per-coordinate sums Q[S, 5, d] over rows of x0 r0, x0 r1, r0 r0, r0 r1, r1 r1 (coordinatewise, on the
            model's flat coordinates and its row mask). With s_x and s_r they give every coordinate's variance of
            the guided prediction (vmg.moments.residual_diag): compute_w.py --match diag. 40 MB per 10^5
            coordinates and 25 levels, so a video latent's is several GB.
"""
import numpy as np
import torch


class ChanCov:
    name = "chancov"

    def __init__(self):
        self.G = None

    @staticmethod
    def channels(model, x):
        """[B, C, P]: the model's own channel axis (an adapter may define channels(x)), else axis 1."""
        return model.channels(x) if hasattr(model, "channels") else x.reshape(x.shape[0], x.shape[1], -1)

    def update(self, model, j, S, x0, D0, D1, cond):
        u = torch.cat([self.channels(model, v).double() for v in (x0, D0 - x0, D1 - x0)], 1)     # [B, 3C, P]
        g = torch.einsum("bcp,bdp->cd", u, u).cpu().numpy()
        if self.G is None:
            self.G = np.zeros((S,) + g.shape)
        self.G[j] += g

    def result(self):
        return {"G": self.G}


class Diag:
    name = "diag"

    def __init__(self):
        self.Q = None

    def update(self, model, j, S, x0, D0, D1, cond):
        from .moments import _flat
        X, m = _flat(model, x0, cond)
        R0, R1 = _flat(model, D0 - x0, cond)[0], _flat(model, D1 - x0, cond)[0]
        w = m[..., None]
        q = torch.stack([(U * V * w).sum((0, 1)) for U, V in ((X, R0), (X, R1), (R0, R0), (R0, R1), (R1, R1))])
        if self.Q is None:
            self.Q = torch.zeros((S,) + tuple(q.shape), dtype=torch.float64, device=q.device)
        self.Q[j] += q

    def result(self):
        return {"Q": self.Q.cpu().numpy()}


EXTRAS = {"chancov": ChanCov, "diag": Diag}
# the extra sums each matching statistic of compute_w.py --match needs
FOR_MATCH = {"trace": (), "cov": ("chancov",), "diag": ("diag",)}


def make(names):
    names = names.split(",") if isinstance(names, str) else list(names or [])
    return [EXTRAS[n]() for n in dict.fromkeys(n for n in names if n)]
