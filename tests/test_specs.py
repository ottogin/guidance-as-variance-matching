"""Each adapter's Spec agrees with the paper's model table (tab:impl-models). Reads paper/supp.tex, never writes it."""
import importlib
import math
import re
from pathlib import Path

import pytest

from vmg.models import REGISTRY

TEX = Path(__file__).resolve().parents[1] / "paper" / "supp.tex"
# the eleven models of tab:impl-models; the other adapters (Fig 2's extra pairs, Latte) are not in that table
TABLE = ("sd15", "sd35m", "sd35l", "flux2-klein", "svd", "wan", "cogvideox", "cosmos", "audioldm2", "sao", "mdm")


def table_rows():
    tex = TEX.read_text()
    body = tex[tex.index(r"\label{tab:impl-models}"):]
    body = body[body.index(r"\midrule") + len(r"\midrule"):body.index(r"\bottomrule")]
    rows = {}
    for line in body.split(r"\\"):
        cells = [c.strip() for c in line.replace(r"\addlinespace", "").split("&")]
        if len(cells) == 9:
            rows[re.sub(r"\s+", " ", cells[0])] = cells
    return rows


def specs():
    out = []
    for key in TABLE:
        mod, cls = REGISTRY[key].split(":")
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError:
            continue                                   # adapter module missing
        if hasattr(m, cls):
            out.append(getattr(m, cls).spec)
    return out


@pytest.mark.skipif(not TEX.exists(), reason="paper/ is a separate repository")
@pytest.mark.parametrize("spec", specs(), ids=lambda s: s.key)
def test_spec_matches_paper(spec):
    rows = table_rows()
    assert spec.name in rows, f"{spec.name!r} not in tab:impl-models ({sorted(rows)})"
    _, cite, ckpt, sampler, output, pub, data, n_cal, n_ev = rows[spec.name]
    assert spec.checkpoint in ckpt.replace(r"}\allowbreak\nolinkurl{", ""), (spec.checkpoint, ckpt)
    assert str(spec.steps) in sampler.replace(",", ""), (spec.steps, sampler)
    split = spec.datasets[spec.default_dataset]
    assert f"{split.n_eval:,}" in n_ev.replace("{,}", ","), (split.n_eval, n_ev)
    assert spec.key == "sao" or f"{split.n_calib:,}" in n_cal, (split.n_calib, n_cal)    # SAO prints its captions
    num = re.fullmatch(r"\s*([0-9.]+)\s*(\$\^\w\$)?\s*", pub)          # "7.5", or "1$^b$" with a footnote mark
    if num:
        assert float(num.group(1)) == spec.published_w, (spec.published_w, pub)
    else:                                                               # SVD's per-frame ramp "$1\to3$"
        assert math.isnan(spec.published_w) and spec.published_note, (spec.published_w, spec.published_note, pub)
