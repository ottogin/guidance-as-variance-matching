"""vmg/metrics/common.py and paired.py on synthetic features (CPU; no detectors)."""
import numpy as np

from vmg.metrics import common as C
from vmg.metrics.paired import compare

SPACES = {"a": dict(fd="fd_a", prdc=True), "b": dict(fd="fd_b", prdc=False)}


def sets(n=300, seed=0):
    rng = np.random.default_rng(seed)
    real = {"a": rng.normal(size=(n, 6)), "b": rng.normal(size=(n, 4))}
    fake = {k: 1.2 * v + rng.normal(scale=0.4, size=v.shape) for k, v in real.items()}
    return real, fake


def test_fd_stats_equals_the_point_set_route():
    real, fake = sets()
    st = {k: (v.mean(0), np.cov(v, rowvar=False)) for k, v in real.items()}
    with_stats = {k: dict(v, fd_stats=st[k]) for k, v in SPACES.items()}
    a, b = C.scores(fake, real, SPACES), C.scores(fake, {"a": real["a"]}, with_stats)
    assert set(a) == set(b) and all(np.isclose(a[k], b[k], rtol=1e-9) for k in a)


def test_identical_arms_have_zero_differences():
    real, fake = sets()
    r = compare(real, fake, fake, SPACES, reps=4, fd=True, per_row={"s": (np.ones(300), np.ones(300))})
    assert all(abs(v["diff"]) < 1e-12 and abs(v["lo"]) < 1e-12 and abs(v["hi"]) < 1e-12 for v in r.values())


def test_feature_files_round_trip(tmp_path):
    real, _ = sets(10)
    p = tmp_path / "x.npz"
    C.save(p, real, {"clip": np.arange(10.0)}, rows=list(range(10)))
    f, pr, m = C.load(p)
    assert np.array_equal(f["a"], real["a"]) and np.array_equal(pr["clip"], np.arange(10.0)) and m["rows"][-1] == 9


def test_windowed_space_selects_by_row():
    """rows=<key>: several features per evaluation row (Cosmos's windows); idx keeps whole rows."""
    real, fake = sets(60)
    per = 3                                                    # 20 evaluation rows of 3 windows each
    pos = np.repeat(np.arange(20), per)
    sp = {"b": dict(fd="fd_b", prdc=True, rows="b_row")}
    R, F = dict(b=real["b"], b_row=pos), dict(b=fake["b"], b_row=pos)
    idx = np.arange(0, 20, 2)
    keep = np.isin(pos, idx)
    got = C.scores(F, R, sp, idx=idx)
    want = C.scores({"b": fake["b"][keep]}, {"b": real["b"][keep]}, {"b": dict(fd="fd_b", prdc=True)})
    assert all(np.isclose(got[k], want[k], rtol=1e-12) for k in want)
    r = compare(R, F, F, sp, reps=3, n=20)
    assert all(abs(v["diff"]) < 1e-12 for v in r.values())


def test_fixed_reference_is_never_resampled():
    """fixed_ref: one real set for every half-sample (AudioCaps' 884 clips behind 4,411 captions)."""
    real, fake = sets(300)
    sp = {"a": dict(fd="fd_a", prdc=True, fixed_ref=True)}
    ref = {"a": real["a"][:120]}                               # fewer real clips than generated rows
    idx = np.arange(0, 300, 2)
    got = C.scores(fake, ref, sp, idx=idx)
    want = C.scores({"a": fake["a"][idx]}, ref, {"a": dict(fd="fd_a", prdc=True, fixed_ref=True)})
    assert got == want and len(compare(ref, fake, fake, sp, reps=2)) == 5


def test_stats_of_the_arm_features():
    real, fake = sets(50)
    sp = {"a": dict(fd=None, prdc=False, stats={"mean_a": lambda f: f.mean()})}
    assert np.isclose(C.scores(fake, real, sp)["mean_a"], fake["a"].mean())
