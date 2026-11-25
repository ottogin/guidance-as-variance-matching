"""Other papers' guidance rules, compared with ours (SD 3.5 Medium, reproduce/baselines/).

A guide turns the two network outputs at sampler level k into the one the sampler steps with:
guide(k, vu, vc) -> v, with vu the unconditional and vc the conditional prediction, both [B, ...] in the
network's parameterisation (SD 3: the velocity v = eps - x0). SD3Family.generate(guide=...) applies it inside
the released pipeline (sd.py guide_hook). `reset()` starts a new batch; `nfe(L)` is the number of network
evaluations the rule itself needs over L levels.

  Mix      v_u + w (v_c - v_u) with any per-level w, below 1 too (SIMCA's least-squares fits, w_MSE)
  Free2    alpha v_c + beta v_u, two free weights per level (SIMCA's own form; Moroy et al. 2026, 2610.03503)
  StepAG   guidance on the first levels only, the conditional prediction after (Zhang et al. 2025, 2506.08351)
  OEC      CFG with orthogonal error correction, the "Dynamic" variant (Yang et al. 2026, 2511.14075, Alg. 1)

Also the two small solvers behind the fits: simca_solve (SIMCA's per-level least squares) and refit_pruned (our
constant weight when Step-AG switches guidance off on some levels).
"""
import numpy as np
import torch

from . import solve


def _per_level(w, L=None):
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    assert L is None or len(w) == L, (len(w), L)
    return w


class Mix:
    """v_u + w[k] (v_c - v_u), in the network's dtype and in diffusers' own order of operations, so a constant
    weight is bitwise the released pipeline's guidance (gate in reproduce/baselines/smoke.py)."""

    def __init__(self, w):
        self.w = _per_level(w)

    def reset(self):
        pass

    def nfe(self, L):
        return 2 * L

    def __call__(self, k, vu, vc):
        return vu + float(self.w[k]) * (vc - vu)


class Free2:
    """alpha[k] v_c + beta[k] v_u, computed in float32."""

    def __init__(self, alpha, beta):
        self.alpha, self.beta = _per_level(alpha), _per_level(beta)
        assert len(self.alpha) == len(self.beta)

    def reset(self):
        pass

    def nfe(self, L):
        return 2 * L

    def __call__(self, k, vu, vc):
        out = float(self.alpha[k]) * vc.float() + float(self.beta[k]) * vu.float()
        return out.to(vc.dtype)


class StepAG:
    """Step AG: guidance at weight w[k] on levels k < k_stop, the conditional prediction alone after.

    The paper keeps CFG on the first fraction p of the steps (k_stop = round(p L)) and recommends the conditional
    branch, not the unconditional one, for the rest. The equivalent SNR rule switches at lambda = (1 - s) / s of
    level k_stop. Here both branches still run on the pruned levels (the pipeline batches them), so nfe() is the
    count the rule needs, not what this run spent."""

    def __init__(self, w, k_stop):
        self.w, self.k_stop = _per_level(w), int(k_stop)

    def reset(self):
        pass

    def nfe(self, L):
        return 2 * min(self.k_stop, L) + max(L - self.k_stop, 0)

    def __call__(self, k, vu, vc):
        return vc if k >= self.k_stop else vu + float(self.w[k]) * (vc - vu)


def oec_uncond(vu, vc, pu, pc, tau):
    """CFG-OEC's corrected unconditional prediction (Alg. 1, Dynamic) and whether it fired, per sample.

    With A = pc - vc and B = pu - vu, the step-to-step change of each branch (pc, pu: the previous level's raw
    predictions), and s = cos(A, B) over the flattened sample: where s < tau,
        B_perp = B - <A, B> / ||A||^2 A,  u~ = B_perp + (2 vu - pu) = vu - proj_A(B),  u~ <- (1 - s) u~ + s vu,
    i.e. u~ = vu - (1 - s) proj_A(B); elsewhere u~ = vu. Float32, shape of vu."""
    sh = vu.shape
    u, c = vu.float().flatten(1), vc.float().flatten(1)
    A, B = pc.float().flatten(1) - c, pu.float().flatten(1) - u
    ab, aa, bb = (A * B).sum(1), (A * A).sum(1), (B * B).sum(1)
    tiny = torch.finfo(torch.float32).tiny
    s = ab / (aa.sqrt() * bb.sqrt()).clamp_min(tiny)
    fire = s < tau
    corr = ((1 - s) * ab / aa.clamp_min(tiny))[:, None] * A
    out = torch.where(fire[:, None], u - corr, u)
    return out.reshape(sh), fire, s


class OEC:
    """CFG-OEC on base weights w[k]: plain CFG at the first level, then u~ + w (v_c - u~) with u~ from
    oec_uncond. Applied in the network's own parameterisation (velocity for SD 3; the paper's models predict eps;
    the error decomposition is the same up to a per-level scale). Counts how often the correction fires."""

    def __init__(self, w, tau=0.9):
        self.w, self.tau = _per_level(w), float(tau)
        self.fired = self.calls = 0
        self.cos = []
        self.prev = None

    def reset(self):
        self.prev = None

    def nfe(self, L):
        return 2 * L

    def stats(self):
        c = np.concatenate(self.cos) if self.cos else np.zeros(0)
        return dict(tau=self.tau, fired=int(self.fired), calls=int(self.calls),
                    rate=float(self.fired / self.calls) if self.calls else None,
                    cos_quantiles=None if not len(c) else [float(q) for q in np.quantile(c, [0.05, 0.25, 0.5, 0.75, 0.95])])

    def __call__(self, k, vu, vc):
        w = float(self.w[k])
        if self.prev is None:
            out = vu + w * (vc - vu)
        else:
            u, fire, s = oec_uncond(vu, vc, *self.prev, self.tau)
            out = (u + w * (vc.float() - u)).to(vc.dtype)
            self.fired += int(fire.sum())
            self.calls += int(fire.numel())
            self.cos.append(s.detach().cpu().numpy())
        self.prev = (vu, vc)
        return out


def simca_solve(P, form):
    """SIMCA's least-squares weights at one level from per-sample inner products.

    P [n, 5] holds, per sample, (<vu,vu>, <vu,vc>, <vc,vc>, <vu,psi>, <vc,psi>) with psi the regression target.
    form 1: the CFG form v_u + w (v_c - v_u), w = sum<v_c - v_u, psi - v_u> / sum||v_c - v_u||^2 -> (w,).
    form 2: free alpha v_c + beta v_u, the 2 x 2 normal equations                                 -> (alpha, beta).
    """
    uu, uc, cc, up, cp = np.asarray(P, dtype=np.float64).sum(0)
    if form == 1:
        return np.array([(cp - up - uc + uu) / (cc - 2 * uc + uu)])
    assert form == 2, form
    H = np.array([[cc, uc], [uc, uu]])
    return np.linalg.solve(H, np.array([cp, up]))


def refit_pruned(a, b, c, keep, clip=solve.CLIP):
    """Our constant weight when guidance is off (w = 1) on the levels where keep is False (Step AG).

    The constant weight solves the matching equation averaged over levels, mean_k (b w^2 - 2 a w - c) = 0
    (eq. wconstant). With w = 1 forced on the pruned levels their terms are the constant b - 2a - c, so the
    weight on the kept levels solves
        mean(b keep) w^2 - 2 mean(a keep) w - [mean(c keep) - mean((b - 2a - c) ~keep)] = 0.
    With every level kept this is solve.fit_global. Returns (w, admissible)."""
    a, b, c = (np.asarray(v, dtype=np.float64) for v in (a, b, c))
    keep = np.asarray(keep, dtype=bool)
    k, p = keep.astype(np.float64), (~keep).astype(np.float64)
    return solve.root((a * k).mean(), (b * k).mean(), (c * k).mean() - ((b - 2 * a - c) * p).mean(), clip)


def k_stop(p, L):
    """Step AG's first unguided level when the first fraction p of L levels keep guidance."""
    return int(round(float(p) * L))
