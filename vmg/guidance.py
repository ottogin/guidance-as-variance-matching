"""Applying a constant weight or a per-level schedule inside a released sampler.

Two traps this module exists for:

* diffusers calls callback_on_step_end AFTER step i and re-reads the scale at the top of step i + 1, so a
  callback must set the weight of the NEXT step; and the lookup is keyed on the timestep, not the step index,
  because PNDM (SD 1.5) visits one timestep twice.
* several pipelines multiply by their local `guidance_scale` argument instead of `self._guidance_scale`, so
  rewriting the attribute is silently a no-op there; LiveScale routes that product through the attribute.

WeightTrace records the weight actually in force at every network call, which is what the tests compare.
"""
import json

import numpy as np

from . import solve

ONE = 1.0 + 1e-6      # never exactly 1 mid-loop: that flips do_classifier_free_guidance off in diffusers


def weights(w, levels):
    """One weight per sampler level from None (-> None), a float, or an array already per level."""
    if w is None:
        return None
    S = len(levels["sigma"])
    w = np.asarray(w, dtype=np.float64)
    return np.full(S, float(w)) if w.ndim == 0 else w.reshape(S)


def from_json(path, kind, levels, match="trace", pool="mean"):
    """The weights of a compute_w.py result for every sampler level.

    kind "schedule" or "global" picks that fit of matching statistic `match` (trace, cov, diag); a global weight
    pooled by `pool` (mean, l1, l2). kind "selected" takes the fit compute_w.py was asked for. The default, the
    trace match pooled by its mean, is the paper's fit. A schedule calibrated on a subset of the levels is
    interpolated in log sigma to the rest."""
    j = json.load(open(path))
    S = len(levels["sigma"])
    if kind == "selected":
        sel = j["selected"]
        kind, match, pool = sel["mode"], sel["match"], sel.get("pool") or "mean"
    assert kind in ("global", "schedule"), kind
    if match == "trace" and (kind == "schedule" or pool == "mean"):
        fit = j[kind]                                      # the paper's fits
    else:
        assert "fits" in j and match in j["fits"], f"{path} has no {match} fit (refit with compute_w.py --match {match})"
        fit = j["fits"][match][kind] if kind == "schedule" else j["fits"][match]["global"][pool]
    if kind == "global":
        return np.full(S, float(fit["w"]))
    w = np.asarray(fit["w"], dtype=np.float64)
    if len(j["levels"]) == S and list(j["levels"]) == list(range(S)):
        return w
    return solve.interp_log_sigma(np.asarray(j["sigma"]), w, np.asarray(levels["sigma"]))


def matched_noise_std(path, levels):
    """Per-level std of the noise controls (sample.py --isonoise / --shapenoise), from a compute_w.py result.

    The controls sample without guidance and add zero-mean Gaussian noise eps to D1 at every network call,
    independent of everything else: isotropic, eps ~ N(0, v I), or shaped like the data, eps ~ N(0, v Cov(x0) /
    (tr Cov(x0) / d)). Either adds v per coordinate to D1's second moment while leaving its direction alone. v is
    chosen so that D1 + eps meets the same target as the fitted weights, tr Cov(D1 + eps) = tr Cov(x0)
    (E||D1 + eps - xbar0||^2 = E||x0 - xbar0||^2 for a --datamean fit). The left side of the matching equation, b w^2 - 2 a w - c, is the guided
    prediction's excess over that target, so at w = 1 it is D1's, and v = c + 2 a - b. Where D1 already meets
    the target there is nothing to add: v is clipped at 0.
    """
    j = json.load(open(path))
    a, b, c = (np.asarray(j["coefficients"][k], dtype=np.float64) for k in "abc")
    std = np.sqrt(np.clip(c + 2 * a - b, 0.0, None))
    S = len(levels["sigma"])
    if len(j["levels"]) == S and list(j["levels"]) == list(range(S)):
        return std
    return solve.interp_log_sigma(np.asarray(j["sigma"], dtype=np.float64), std, np.asarray(levels["sigma"]))


def diffusers_hook(w, t_levels, to_pipeline=None, floor=ONE):
    """(w0, callback, weight_at) that guide sampler step i at the weight of the level it is about to denoise.

    `w` has one entry per level, `t_levels` the timesteps of those levels. Pass `w0` as the pipeline's
    guidance_scale (wrapped in LiveScale where needed) and `callback` as callback_on_step_end.
    `to_pipeline` converts the paper's w to the pipeline's own number (e.g. w - 1 where the pipeline writes
    D1 + g (D1 - D0)); `floor` applies to the pipeline's number.
    """
    tarr = np.asarray(t_levels, dtype=np.float64)
    pw = np.asarray(w if to_pipeline is None else to_pipeline(np.asarray(w)), dtype=np.float64)

    def weight_at(t):
        return max(float(pw[int(np.argmin(np.abs(tarr - float(t))))]), floor)

    def callback(pipe, i, t, kw):
        ts = pipe.scheduler.timesteps
        pipe._guidance_scale = weight_at(ts[i + 1] if i + 1 < len(ts) else ts[-1])
        return kw

    return weight_at(tarr[0]), callback, weight_at


class LiveScale(float):
    """The guidance_scale argument for loops that multiply by the local argument (FLUX.2 [klein], Wan, ...).

    Every other use of the argument (the `> 1` tests, storing it) sees the plain step-0 value, so a constant
    weight is bitwise unchanged; only the product reads whatever the hook last set."""

    def __new__(cls, pipe, w0):
        x = super().__new__(cls, w0)
        x.pipe = pipe
        return x

    def __mul__(self, other):
        return float(self.pipe._guidance_scale) * other

    __rmul__ = __mul__


class WeightTrace:
    """Records (timestep, weight in force) at every call of `module` (the pipeline's denoiser network)."""

    def __init__(self, pipe, module):
        self.pipe, self.seen = pipe, []
        self.h = module.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, mod, args, kwargs):
        t = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        t = None if t is None else float(np.asarray(t.detach().float().cpu() if hasattr(t, "detach") else t).ravel()[0])
        self.seen.append((t, float(self.pipe._guidance_scale)))

    def remove(self):
        self.h.remove()
