"""The published schedules of tab:pub-sched (vmg/schedules.py) on SD 3.5 Medium's 40-step grid."""
from pathlib import Path

import numpy as np
import pytest

from vmg import schedules as P

FIX = Path(__file__).with_name("fixtures") / "sd35m-coco-s40.npz"


@pytest.fixture(scope="module")
def levels():
    z = np.load(FIX, allow_pickle=True)
    return dict(t=1000 * z["t"], sigma=z["sigma"])     # the fixture keeps flow time s; the adapter's t is 1000 s


def test_step_means_match_the_paper(levels):
    # tab:pub-sched, column "mean w", at the released w = 4.5
    for kind, mean in (("interval", 4.02), ("tvcfg", 4.82), ("linear", 3.28)):
        assert round(P.published(kind, levels, 4.5).mean(), 2) == mean


def test_interval_and_tvcfg_keep_the_total(levels):
    u = levels["t"] / levels["t"].max()
    for kind in ("interval", "tvcfg"):
        w = P.published(kind, levels, 4.5)
        assert np.isclose(np.sum(w * P.widths(u)), 4.5)


def test_interval_is_one_outside(levels):
    w = P.published("interval", levels, 4.5)
    inside = (levels["sigma"] > P.INTERVAL[0]) & (levels["sigma"] <= P.INTERVAL[1])
    assert (w[~inside] == 1).all() and (w[inside] > 4.5).all()


def test_linear_rises_from_one(levels):
    w = P.published("linear", levels, 4.5)
    assert np.isclose(w[0], 1 + 2 * 3.5 * (1 - levels["t"][0] / 1000)) and (np.diff(w) > 0).all()
