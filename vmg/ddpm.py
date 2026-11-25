"""A DDPM sampler whose clean prediction is a Gaussian draw around the conditional prediction.

The guided sampler steps with the deterministic D_w = D1 + (w - 1)(D1 - D0). Here the same guidance difference
sets the spread of a random clean prediction instead,

    x~0 | x_t ~ N(D1(x_t), Sigma(x_t)),    Sigma built from kappa (D1 - D0),

and the sampler takes the exact DDPM reverse step given that draw, x_s' ~ q(x_s' | x_s, x~0). Marginalised over
the draw this is one Gaussian step N(A x_s + B D1, C^2 I + B^2 Sigma): the reverse covariance of DDPM plus the
x0 uncertainty, as in Bao et al. 2022 (covariance under an imperfect mean). It is the paper's M(2) sampler with
the posterior covariance replaced by one built from the two predictions, and DDPM in place of DDIM.

Shapes of Sigma (all add the same expected energy E||x~0 - D1||^2 = kappa^2 ||D1 - D0||^2, which for
kappa = w - 1 is exactly what guidance at w adds on top of D1):

    iso    kappa^2 (||D1 - D0||^2 / d) I      isotropic, sized per sample and per level
    diag   kappa^2 diag((D1 - D0)^2)          per coordinate, where the two predictions disagree
    rank1  kappa^2 (D1 - D0)(D1 - D0)^T       along the guidance direction only (a random signed weight)
    none   0                                  plain DDPM at the mean

The mean is D1, or D_w (mean="dw", the DDPM version of ordinary guidance, a control).

Flow parameterisation x_s = (1 - s) x0 + s eps (SD 3). With sigma = s / (1 - s) and r = sigma'^2 / sigma^2,
q(x_s' | x_s, x0) = N(A x_s + B x0, C^2 I) with

    A = s'^2 (1 - s) / (s^2 (1 - s')),   B = (1 - s')(1 - r),   C = s' sqrt(1 - r),

finite at s = 1 (A = 0: the first step forgets the pure-noise input). The last step (s' = 0) returns the mean
without a draw, as DDPM does.
"""
from dataclasses import dataclass

import numpy as np
import torch

SHAPES = ("none", "iso", "diag", "rank1")
MEANS = ("d1", "dw")


def flow_posterior(s, s_next):
    """(A, B, C) of q(x_s' | x_s, x0) = N(A x_s + B x0, C^2 I) for x_s = (1 - s) x0 + s eps, s' < s."""
    s, sn = float(s), float(s_next)
    assert 0 <= sn < s <= 1, (s, sn)
    if sn == 0:
        return 0.0, 1.0, 0.0
    r = (sn * (1 - s)) ** 2 / (s * (1 - sn)) ** 2
    return sn * sn * (1 - s) / (s * s * (1 - sn)), (1 - sn) * (1 - r), sn * float(np.sqrt(1 - r))


@dataclass
class PosteriorDDPM:
    """The sampler's rule at one level: the clean draw from the two predictions, then the reverse step.

    kappa scales the spread (kappa = w - 1 adds what guidance at w adds; kappa = w is the literal w (D1 - D0));
    w is the guidance weight of the mean when mean = "dw"."""

    shape: str = "iso"
    kappa: float = 3.5
    mean: str = "d1"
    w: float = 4.5

    def __post_init__(self):
        assert self.shape in SHAPES and self.mean in MEANS, (self.shape, self.mean)

    def name(self):
        m = "ddpm" if self.mean == "d1" else f"ddpm-dw{self.w:g}"
        return m if self.shape == "none" else f"{m}-{self.shape}-k{self.kappa:g}"

    def centre(self, d0, d1):
        return d1 if self.mean == "d1" else d0 + self.w * (d1 - d0)

    def draw(self, d0, d1, xi, g):
        """x~0 for one batch: xi [B, ...] and g [B] standard normals (g is the rank-1 weight). Float32."""
        m = self.centre(d0, d1)
        if self.shape == "none" or self.kappa == 0:
            return m
        dl = d1 - d0
        if self.shape == "iso":
            rms = dl.flatten(1).pow(2).mean(1).sqrt().view(-1, *(1,) * (dl.dim() - 1))
            return m + self.kappa * rms * xi
        if self.shape == "diag":
            return m + self.kappa * dl.abs() * xi
        return m + self.kappa * g.view(-1, *(1,) * (dl.dim() - 1)) * dl

    def step(self, x, s, s_next, d0, d1, xi, g, z):
        """x_s' from x_s at flow times s > s_next: draw x~0, then q(x_s' | x_s, x~0); the mean at s_next = 0."""
        if float(s_next) == 0:
            return self.centre(d0, d1)
        A, B, C = flow_posterior(s, s_next)
        return A * x + B * self.draw(d0, d1, xi, g) + C * z

    def info(self):
        return dict(sampler="posterior-ddpm", shape=self.shape, kappa=float(self.kappa), mean=self.mean,
                    w=float(self.w), final="mean")


def euler_cfg(x, s, s_next, vu, vc, w):
    """The released sampler's step, as diffusers takes it with a float32 state: the guided velocity formed in the
    network's dtype, then x + (s' - s) v in float32 (s, s_next: the scheduler's float32 sigmas)."""
    v = (vu + w * (vc - vu)).float()
    return x + (s_next - s) * v
