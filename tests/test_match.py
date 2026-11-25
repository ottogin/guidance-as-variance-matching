"""compute_w.py's matching statistics (trace, cov, diag), pools (mean, l1, l2) and model sources."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vmg import extras as X
from vmg import moments as M
from vmg import solve, sources
from vmg.models import get

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).with_name("fixtures") / "sd35m-coco-s40.npz"


@pytest.fixture(scope="module")
def sd35m():
    z = np.load(FIX)
    return solve.Moments(sigma=z["sigma"], t=z["t"], var_x0=float(z["var_x0"]), xr=z["xr"], rr=z["rr"],
                         rr_centred=z["rr_centred"], n=int(z["n"]), d=int(z["d"]))


@pytest.fixture(scope="module")
def toy_pass():
    toy = get("toy")
    rows = toy.rows("mixture")[:2048]
    return M.accumulate(toy, "mixture", rows, log=lambda *a: None, extras=X.make("chancov,diag"))


# ---------------------------------------------------------------- the trace match and its pools

def test_sd35m_pools_reproduce_tab_objective(sd35m):
    """App. D.2, SD 3.5 Medium: Ours / L1 / L2 printed as 4.14 / 3.79 / 3.24. This fixture is the 4.134 pass (the
    printed table used the 4.138 one), so each pool sits a few thousandths below its printed value."""
    g = solve.fit_all(sd35m)["trace"]["global"]
    assert abs(g["mean"]["w"] - 4.134) < 1e-3 and g["mean"]["admissible"]
    assert abs(g["l1"]["w"] - 3.79) < 0.02
    assert abs(g["l2"]["w"] - 3.24) < 0.02


def test_argmin_fits_recover_the_closed_form_roots(sd35m):
    """Where the paper's root exists, argmin r_t(w)^2 is that root: per level, and for the mean pool."""
    E = solve.residual_trace(sd35m)
    s = solve.fit_schedule(sd35m)
    am = solve.fit_residual(E, "schedule")
    assert np.allclose(am[s["admissible"]], s["w"][s["admissible"]], atol=1e-9)
    assert abs(solve.fit_residual(E, "global", "mean") - solve.fit_global(sd35m)[0]) < 1e-9


def test_l1_and_l2_are_minimisers(sd35m):
    E = solve.residual_trace(sd35m)
    r = lambda w: E[:, 0] + w * E[:, 1] + w * w * E[:, 2]          # noqa: E731
    g = np.linspace(1e-3, 40, 200001)
    R = E[:, :1] + g * E[:, 1:2] + g * g * E[:, 2:3]
    l1, l2 = solve.fit_residual(E, "global", "l1"), solve.fit_residual(E, "global", "l2")
    assert np.abs(r(l1)).mean() <= np.abs(R).mean(0).min() + 1e-12
    assert (r(l2) ** 2).mean() <= (R ** 2).mean(0).min() + 1e-12


def test_argmin_poly_ends_and_interior():
    assert solve.argmin_poly([0, -2, 1], 0, 40) == pytest.approx(1.0)          # (w - 1)^2 - 1
    assert solve.argmin_poly([0, 1, 0], 0, 40) == 0.0                           # increasing: the left end
    assert solve.argmin_poly([0, -1, 0], 0, 40) == 40.0                         # decreasing: the right end


# ---------------------------------------------------------------- cov and diag from the extra sums

def test_diag_and_cov_reduce_to_the_trace(toy_pass):
    """The coordinates' mean of the diag residual, and the cov residual's trace over C, are the trace residual."""
    sums, meta = toy_pass
    m = M.reduce(sums, meta)
    for rule in solve.RULES:
        Et = solve.residual_trace(m, rule)
        Ed, Ec = M.residual_diag(sums, meta, rule), M.residual_cov(sums, meta, rule)
        assert np.allclose(Ed.mean(-1), Et, atol=1e-12)
        assert np.allclose(np.trace(Ec, axis1=-2, axis2=-1) / Ec.shape[-1], Et, atol=1e-12)


def test_cov_fit_is_the_brute_force_frobenius_argmin():
    """On the 2-D toy the channel covariance is the full 2 x 2 one: fit it by brute force from the samples."""
    toy = get("toy")
    rows = toy.rows("mixture")[:1024]
    sums, meta = M.accumulate(toy, "mixture", rows, log=lambda *a: None, extras=X.make("chancov"))
    fit = solve.fit_residual(M.residual_cov(sums, meta), "schedule")
    x0, c = toy.draw(rows)
    from vmg import noise
    grid = np.linspace(0, 40, 4001)
    for k in (6, 12, 15):
        e = noise.eps_batch(toy.seed, rows, k, (2,))
        d1, d0 = toy.denoise_pair(toy.noise(x0, k, e), k, c)
        d1, d0, x = d1.double().numpy(), d0.double().numpy(), x0.double().numpy()
        T = np.cov(x.T, bias=True)
        cost = [np.sum((np.cov((d0 + w * (d1 - d0)).T, bias=True) - T) ** 2) for w in grid]
        assert abs(grid[int(np.argmin(cost))] - fit[k]) < 2e-2, (k, grid[int(np.argmin(cost))], fit[k])


def test_diag_residual_matches_direct_variances():
    """residual_diag from raw coordinatewise sums equals Var(D_w) - Var(x0) per coordinate, at any w."""
    rng = np.random.default_rng(0)
    n, d, S = 400, 6, 3
    x0 = rng.normal(size=(n, d)) * rng.uniform(0.5, 2, d) + rng.normal(size=d)
    r = rng.normal(size=(S, 2, n, d)) * 0.3 + rng.normal(size=(S, 2, 1, d)) - 0.4 * x0
    Q = np.stack([np.stack([(x0 * r[j, 0]).sum(0), (x0 * r[j, 1]).sum(0), (r[j, 0] ** 2).sum(0),
                            (r[j, 0] * r[j, 1]).sum(0), (r[j, 1] ** 2).sum(0)]) for j in range(S)])
    sums = dict(x_diag_Q=Q, cnt=np.ones(n), s_x=x0.sum(0), s_r=r.sum(2))
    for rule in solve.RULES:
        E = M.residual_diag(sums, dict(d=d), rule)
        for j in range(S):
            for w in (0.0, 1.0, 3.5):
                Dw = x0 + r[j, 0] + w * (r[j, 1] - r[j, 0])
                ctr = x0.mean(0) if rule == "datamean" else Dw.mean(0)
                want = ((Dw - ctr) ** 2).mean(0) - x0.var(0)
                assert np.allclose(E[j, 0] + w * E[j, 1] + w * w * E[j, 2], want, atol=1e-10)


def test_diag_extra_on_a_masked_model():
    """The diag extra follows the model's flat coordinates and its row mask, as the standard sums do."""
    class Masked:
        def flat(self, x, cond):
            return x.reshape(x.shape[0], 2, -1), torch.tensor([[1.0, 0.0]] * x.shape[0])
    ex = X.Diag()
    x0 = torch.randn(5, 2, 3)
    D0, D1 = x0 + torch.randn(5, 2, 3), x0 + torch.randn(5, 2, 3)
    ex.update(Masked(), 0, 1, x0, D0, D1, None)
    want = ((x0[:, 0] * (D1 - x0)[:, 0]).double()).sum(0)
    assert torch.allclose(torch.as_tensor(ex.result()["Q"][0, 1]), want)


# ---------------------------------------------------------------- sources

def test_plugin_and_preset_sources(tmp_path):
    a = lambda **k: type("A", (), dict(dict(model="toy", guidance=None, reference=None, steps=None, published_w=None,
                                            size=None, dataset=None), **k))()      # noqa: E731
    assert sources.resolve(a()).spec.key == "toy"
    assert sources.resolve(a(model="vmg.models.toy:Toy")).spec.key == "toy"
    with pytest.raises(SystemExit):
        sources.resolve(a(model=str(tmp_path)))                     # cfg by default, but no model_index.json
    with pytest.raises(AssertionError):
        sources.resolve(a(model="vmg.models.base:Split"))           # not a Model


def test_diffusers_family_from_model_index(tmp_path, monkeypatch):
    ck = tmp_path / "my-sd3"
    (ck / "scheduler").mkdir(parents=True)
    (ck / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusion3Pipeline"}))
    monkeypatch.setattr(sources, "pipeline_defaults", lambda c: (28, 7.0))
    m = sources.diffusers_cfg(str(ck))
    from vmg.models.sd import SD3Family, SD35Medium
    assert isinstance(m, SD3Family) and m.spec.checkpoint == str(ck) and m.spec.steps == 28
    assert np.isnan(m.spec.published_w) and "7.0" in m.spec.published_note
    assert SD35Medium.spec.steps == 40 and SD35Medium.spec.checkpoint.startswith("stabilityai")   # class untouched
    (ck / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))
    (ck / "scheduler" / "scheduler_config.json").write_text(json.dumps({"prediction_type": "v_prediction"}))
    with pytest.raises(AssertionError, match="epsilon"):
        sources.diffusers_cfg(str(ck))
    (ck / "model_index.json").write_text(json.dumps({"_class_name": "KandinskyPipeline"}))
    with pytest.raises(SystemExit, match="no adapter"):
        sources.diffusers_cfg(str(ck))


@pytest.mark.parametrize("kind,l1,l0,ok", [("cfg-pair", 1000, 0, True), ("cfg-pair", 1000, 1000, False),
                                           ("autoguidance", 1000, 1000, True), ("autoguidance", 1000, 0, False)])
def test_generic_edm_pair_checks_the_label_spaces(monkeypatch, kind, l1, l0, ok):
    from vmg.models import _edm, edm2
    nets = {"a.pkl": l1, "b.pkl": l0}
    fake = lambda rel, device="cuda": (type("N", (), dict(label_dim=nets[rel], img_resolution=64,   # noqa: E731
                                                          img_channels=3))(), None)
    monkeypatch.setattr(_edm, "edm_pickle", fake)
    pair = edm2.generic_pair("a.pkl", "b.pkl", "imagenet64", kind, key="a__b")()
    if ok:
        pair.load()
    else:
        with pytest.raises(AssertionError):
            pair.load()


def test_guidance_auto(monkeypatch):
    """--guidance auto (the default): cfg without --reference; with one, the label spaces decide."""
    from vmg.models import _edm, edm2
    calls = []
    monkeypatch.setattr(sources, "diffusers_cfg", lambda m, **k: calls.append(("cfg", m)) or get("toy"))
    a = lambda **k: type("A", (), dict(dict(model="org/some-pipeline", guidance="auto", reference=None, steps=None,  # noqa
                                            published_w=None, size=None, dataset="imagenet64"), **k))()
    sources.resolve(a())
    assert calls == [("cfg", "org/some-pipeline")]
    for l0, want in ((0, "cfg-pair"), (1000, "autoguidance")):
        nets = {"a.pkl": 1000, "b.pkl": l0}
        monkeypatch.setattr(_edm, "edm_pickle", lambda rel, device="cuda": (type("N", (), dict(   # noqa: B023
            label_dim=nets[rel], img_resolution=64, img_channels=3))(), None))
        monkeypatch.setattr(sources, "local_pickle", lambda p: p)
        pair = sources.resolve(a(model="a.pkl", reference="b.pkl"))
        assert isinstance(pair, edm2.GenericPair) and pair.kind == "auto"
        pair.load()
        assert pair.kind == want


def test_compute_w_defaults():
    """A bare compute_w.py call is the paper's fit: trace, schedule, mean pool, 1,024 rows, the covariance."""
    import compute_w  # noqa: F401
    from vmg import cli
    ap = cli.parser("x")
    g = ap.add_argument_group("the fit")
    src = (REPO / "compute_w.py").read_text()
    assert 'default="trace"' in src and 'default="schedule"' in src and 'default="mean"' in src
    a = ap.parse_args(["--model", "toy"])
    assert a.n == "1024" and a.guidance == "auto" and a.stage == "all" and a.boot == 200 and g is not None


# ---------------------------------------------------------------- the command line

def test_compute_w_cli_every_option(tmp_path):
    env = dict(os.environ, VMG_RUNS=str(tmp_path))
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=REPO, env=env, check=True, capture_output=True,   # noqa
                                    text=True).stdout
    run("compute_w.py", "--model", "toy", "--n", "512", "--match", "all", "--boot", "4")
    for mt in solve.MATCHES:
        for mode, pool in [("schedule", "mean")] + [("global", p) for p in solve.POOLS]:
            out = run("compute_w.py", "--model", "toy", "--n", "512", "--stage", "fit", "--match", mt, "--mode", mode,
                      "--pool", pool, "--boot", "0")
            assert f"match {mt}" in out
    j = json.load(open(tmp_path / "toy" / "mixture-n512" / "w.json"))
    assert set(j["fits"]) == set(solve.MATCHES) and j["selected"] == dict(match="diag", mode="global", pool="l2",
                                                                         w=j["fits"]["diag"]["global"]["l2"]["w"])
    assert j["global"]["w"] == j["fits"]["trace"]["global"]["mean"]["w"]          # the paper's fit
    out = run("sample.py", "--model", "toy", "--vm", "selected", "--n", "16", "--weights-tag", "n512")
    w = json.load(open(tmp_path / "toy" / "mixture" / "samples" / "vm-selected" / "weights.json"))["w"]
    assert np.allclose(w, j["selected"]["w"]) and "vm-selected" in out
    # a subset pass is never reused for another row count
    r = subprocess.run([sys.executable, "compute_w.py", "--model", "toy", "--n", "512", "--tag", "n512", "--stage",
                        "fit", "--n", "256"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode != 0
