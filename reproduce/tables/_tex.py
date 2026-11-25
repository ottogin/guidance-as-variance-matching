"""Cell formatting shared by the generated tables: the paper's colour conventions.

shade() grades an Ours value by its relative change against Pub., on the printed values: vmsame within 1%,
vmbetterlt / vmworselt for 1-5%, vmbetter / vmworse beyond 5% (the cell shading of the extra tables). pair()
writes the appendix's every-metric tables as the paper prints them, coloured as tab:pub-all: an Ours value in
vmgood where better and vmbad where worse beyond 1%, bold beyond 5%, black within 1%; no stars.
The colours and macros (\\sig, \\na, \\dz, \\hd) are defined in the paper's preamble.
"""
import json
from pathlib import Path

from vmg import runs

LOWER_BETTER = {"fid", "fd_dinov2", "fvd", "fad", "kl", "kl_sigmoid", "fd_passt", "fd_openl3", "mdm_fid", "mm_dist"}
UNRANKED = {"diversity"}         # best closest to the real data's value: neither shaded nor bolded


def shade(key, a, b):
    rel = (b - a) / abs(a) if a else 0.0
    if abs(rel) < 0.01:
        return "vmsame"
    better = rel < 0 if key in LOWER_BETTER else rel > 0
    return ("vmbetter" if better else "vmworse") + ("lt" if abs(rel) < 0.05 else "")


def fmt(v, digits):
    return f"{v:.{digits}f}"


def pair(rec, key, digits, pad=False):
    """(Pub. cell, Ours cell) for one compare record, Ours coloured on the printed values. pad appends \\dz, the
    phantom digit the paper uses to align a cell printed with fewer decimals than its column."""
    a, b = round(rec["a"], digits), round(rec["b"], digits)
    dz = r"\dz" if pad else ""
    s = "vmsame" if key in UNRANKED else shade(key, a, b)
    ours = fmt(b, digits)
    if s != "vmsame":
        col = "vmgood" if "better" in s else "vmbad"
        ours = rf"{{\color{{{col}}}{ours}}}" if s.endswith("lt") else rf"{{\bfseries\color{{{col}}}{ours}}}"
    return fmt(a, digits) + dz, ours + dz


def compare_json(model, dataset, a="pub", b="vm-schedule", tag=None):
    path = runs.metrics_json(model, dataset, f"compare-{b}-vs-{a}", tag)
    return json.load(open(path))["metrics"] if path.exists() else None


def full_table(models, rows, out, name, allow_missing=False, ncols=9):
    """The appendix's every-metric tables: one Pub. / Ours column pair per (model, dataset), rows grouped under
    italic heads ((head or None, [(key, label, decimals)])). Writes <out>/<name>, the tabular body between
    \\midrule and \\bottomrule."""
    recs = [compare_json(m, d) for m, d in models]
    missing = [f"{m}/{d}" for (m, d), r in zip(models, recs) if r is None]
    assert not missing or allow_missing, f"no compare-vm-schedule-vs-pub.json for {', '.join(missing)}"
    lines = []
    for head, group in rows:
        if head:
            lines.append(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{head}}}}} \\")
        for key, label, digits in group:
            cells = []
            for r in recs:
                cells += list(pair(r[key], key, digits)) if r is not None and key in r else [r"\na", r"\na"]
            lines.append(("\\quad " if head else "") + label + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    path = Path(out) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines[:-1]) + "\n")
    print(f"wrote {path}" + (f" (missing: {', '.join(missing)})" if missing else ""))
