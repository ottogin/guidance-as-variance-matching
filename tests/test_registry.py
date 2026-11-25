"""reproduce/paper.py covers the paper: one function per float, every label, every script it names exists."""
import re
import shlex
from pathlib import Path

import pytest

from reproduce import paper

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"


def paper_labels():
    """Every fig: / tab: label of main.tex, supp.tex and the Figures/*.tex they \\input, comments dropped."""
    seen, todo, labels = set(), ["main.tex"], set()
    while todo:
        name = todo.pop()
        if name in seen or not (PAPER / name).exists():
            continue
        seen.add(name)
        text = "\n".join(line.split("%")[0] if not line.lstrip().startswith("%") else ""
                         for line in (PAPER / name).read_text().splitlines())
        labels |= set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", text))
        todo += [m if m.endswith(".tex") else m + ".tex" for m in re.findall(r"\\input\{([^}]+)\}", text)]
    return labels


@pytest.mark.skipif(not (PAPER / "main.tex").exists(), reason="paper/ is a separate repository")
def test_every_paper_float_has_a_function():
    reg = {lab for f in paper.FLOATS if f.section != "extra" for lab in f.labels}
    got = paper_labels()
    assert got - reg == set(), f"floats of the paper with no function in reproduce/paper.py: {sorted(got - reg)}"
    assert reg - got == set(), f"functions for labels the paper no longer has (mark them extra): {sorted(reg - got)}"
    extra = {lab for f in paper.FLOATS if f.section == "extra" for lab in f.labels}
    assert not extra & got, f"extras that are in the paper: {sorted(extra & got)}"


def scripts(cmd):
    return [w for w in shlex.split(cmd) if w.endswith((".py", ".sbatch", ".sh", ".js"))]


@pytest.mark.parametrize("f", paper.FLOATS, ids=lambda f: f.name)
def test_recipe_or_pending_stub(f):
    p = f.pending()
    if p is not None:
        assert p.startswith("pending"), p
        return
    r = f.recipe()
    assert r.make, f"{f.name}: no command draws it"
    for cmd in r.make + [j.cmd for j in r.jobs] + [j.smoke for j in r.jobs if j.smoke]:
        for s in scripts(cmd):
            assert (REPO / s).exists(), f"{f.name}: {s} does not exist"
    for j in r.jobs:
        assert j.runner in ("python", "sbatch") and j.kind in ("gpu", "cpu")
        if j.runner == "python":
            assert j.cmd.split()[0].endswith(".py"), j.cmd


def test_dependencies_point_at_jobs_of_some_float():
    allcmds = {j.cmd for f in paper.FLOATS if not f.pending() for j in f.recipe().jobs}
    for f in paper.FLOATS:
        if f.pending():
            continue
        for j in f.recipe().jobs:
            for c in j.after:
                assert c in allcmds, f"{f.name}: {j.cmd!r} waits for {c!r}, which no float runs"


def test_paper_calibrations_use_every_row():
    """compute_w.py's default is 1,024 rows; the paper's fits use every calibration row, except tab:edm2-fail's,
    which App. C.7 fits on the first plan.EDM2_FAIL_N."""
    from reproduce import plan
    first_n = {j.cmd for j in plan.edm2_fail_jobs()}
    for f in paper.FLOATS:
        if f.pending():
            continue
        for j in f.recipe().jobs:
            if j.cmd.startswith(("compute_w.py", "compute_wmse.py")):
                n = f"--n {plan.EDM2_FAIL_N}" if j.cmd in first_n else "--n all"
                assert n in j.cmd, j.cmd
