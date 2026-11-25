"""sample.py's baseline modes end to end with a stand-in model: the arm names plan.py expects, the weights that
reach the model, and what weights.json records (no GPU, no weights)."""
import json
import sys

import numpy as np
import pytest

import sample as S
from reproduce import plan
from vmg import baselines as B
from vmg import solve

L = 40
FIX = "tests/fixtures/sd35m-coco-s40.npz"


class Spec:
    key, name, published_w, default_dataset = "sd35m", "stand-in", 4.5, "coco"


class Fake:
    spec = Spec()

    def __init__(self):
        self.calls = []

    def rows(self, dataset, split):
        return list(range(5705, 5705 + 32))

    def load(self):
        pass

    def levels(self):
        s = np.linspace(1.0, 0.01, L)
        with np.errstate(divide="ignore"):
            return dict(sigma=np.where(s < 1, s / (1 - s), np.inf), t=1000 * s)

    def guide_hook(self, guide):
        pass

    def sample(self, dataset, rows, w, out, **kw):
        self.calls.append(dict(rows=rows, w=w, **kw))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VMG_RUNS", str(tmp_path))
    run = tmp_path / "sd35m" / "coco"
    run.mkdir(parents=True)
    fx = np.load(FIX)
    m = solve.Moments(sigma=fx["sigma"], t=fx["t"], var_x0=float(fx["var_x0"]), xr=fx["xr"], rr=fx["rr"],
                      rr_centred=fx["rr_centred"], n=int(fx["n"]), d=int(fx["d"]))
    sch = solve.fit_schedule(m, "centred")
    j = dict(levels=list(range(L)), sigma=list(fx["sigma"]), clip=list(solve.CLIP), rule="centred",
             **{"global": dict(w=solve.fit_global(m, "centred")[0])}, schedule=dict(w=list(sch["w"])),
             coefficients={k: list(sch[k]) for k in "abc"})
    (run / "w.json").write_text(json.dumps(j))
    (run / "simca-sf.json").write_text(json.dumps(dict(method="simca-sf", w=list(solve.w_mse(m)))))
    (run / "simca-sb2.json").write_text(json.dumps(dict(method="simca-sb2", alpha=[3.0] * L, beta=[-1.5] * L)))
    fake = Fake()
    monkeypatch.setattr(S.sources, "resolve", lambda a: fake)
    return fake, tmp_path, m, sch


def run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sample.py", "--model", "sd35m"] + argv)
    S.main()


@pytest.mark.parametrize("flag, arm, ref, needs", plan.BASELINE_ARMS)
def test_every_planned_arm(env, monkeypatch, flag, arm, ref, needs):
    fake, root, m, sch = env
    if "simca-sb1" in flag:
        pytest.skip("the GPU fit writes it")
    run(flag.split() + ["--fp32-state", "--tag", "baselines"], monkeypatch)
    out = root / "sd35m" / "coco-baselines" / "samples" / arm
    assert out.is_dir(), f"sample.py wrote another arm than plan.py's {arm}"
    rec = json.load(open(out / "weights.json"))
    call = fake.calls[-1]
    assert call["fp32_state"] is True and rec["fp32_state"] is True
    g = call["guide"]
    if "--stepag" in flag:
        p = float(flag.split()[1])
        ks = B.k_stop(p, L)
        assert isinstance(g, B.StepAG) and g.k_stop == ks and rec["nfe"] == 2 * ks + L - ks
        assert np.allclose(rec["w"][ks:], 1.0)
        if "vm-refit" in flag:
            wr, _ = B.refit_pruned(*(sch[k] for k in "abc"), np.arange(L) < ks)
            assert np.allclose(g.w, wr)
        elif "vm-schedule" in flag:
            assert np.allclose(g.w, sch["w"])
    elif "--oec" in flag:
        assert isinstance(g, B.OEC) and g.tau == 0.9 and rec["nfe"] == 2 * L
        assert np.allclose(g.w, 4.5 if "pub" in flag else sch["w"])
    elif "simca-sf" in flag:
        assert isinstance(g, B.Mix) and np.allclose(g.w, solve.w_mse(m)) and (g.w < 1).sum() == L - 3
    elif "simca-sb2" in flag:
        assert isinstance(g, B.Free2) and np.allclose(rec["w"], 3.0 / 1.5)
    else:
        assert g is None, "the reference arms use the plain weight path"
        if "global" in flag:
            assert np.allclose(call["w"], solve.fit_global(m, "centred")[0])


def test_a_stale_datamean_fit_is_refused(env, monkeypatch):
    p = env[1] / "sd35m" / "coco" / "w.json"
    j = json.loads(p.read_text())
    p.write_text(json.dumps({**j, "rule": "datamean"}))
    with pytest.raises(AssertionError, match="fitted with rule 'datamean'"):
        run(["--vm", "global"], monkeypatch)


def test_base_vm_refit_needs_stepag(env, monkeypatch):
    with pytest.raises(AssertionError):
        run(["--oec", "0.9", "--base", "vm-refit"], monkeypatch)
