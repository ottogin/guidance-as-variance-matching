#!/usr/bin/env python
"""Compare the generated tables with the paper's, number by number (reads the paper's LaTeX source, never writes).

    python reproduce/check_paper.py --out runs/paper/full [--tex paper/main.tex]

The paper's source is not part of this repository: unpack it so that --tex is its main.tex,
with supp.tex and Figures/ beside it. For every generated table fragment in --out that has a counterpart in the
paper, the numbers of each row are read
in order (shading, bold and stars ignored) and compared with the paper's row of the same label. A rebuild is
expected to differ by a little (other GPUs, bf16); a large jump, a row that is missing or a sign change is what
this is for.
"""
import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# generated file -> (where its rows sit in the paper: a \label inside main.tex / supp.tex, or a file under Figures/)
TABLES = {"tuned-w.tex": ("label", "tab:tuned-w"), "pub-sched.tex": ("label", "tab:pub-sched"),
          "full-image.tex": ("label", "tab:full-image"), "full-video.tex": ("label", "tab:full-video"),
          "pub-all.tex": ("label", "tab:pub-all"), "full-audio.tex": ("label", "tab:full-audio"),
          "full-motion.tex": ("label", "tab:full-motion"), "impl.tex": ("label", "tab:impl-models"),
          "objective.tex": ("label", "tab:objective"), "noise-control.tex": ("label", "tab:noise-control"),
          "user-study-elo.tex": ("file", "user-study-elo.tex"),
          "user-study-h2h.tex": ("file", "user-study-h2h.tex"),
          "user-study-elo-fable.tex": ("file", "user-study-elo-fable.tex")}
# tab:impl-models / tab:impl-data are descriptive; tests/test_specs.py checks their facts against the paper
NUM = re.compile(r"(?<![A-Za-z_{])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def strip(cell):
    # \best (tab:pub-sched's underline) wraps a cell, so only its opening brace goes; \oursrow is tab:pub-all's row band
    for pat in (r"\\best\{", r"\\oursrow", r"\\(?:cellcolor|rowcolor)\{[^}]*\}", r"\\textbf\{([^}]*)\}", r"\\underline\{([^}]*)\}",
                r"\\color\{[^}]*\}", r"\\(sig|dz|na|quad)\b",
                r"\$\^\\dagger\$", r"\\boldmath", r"\\(?:mathrm|rm)\{[^}]*\}", r"_\{[^}]*\}", r"\\cite\w*\{[^}]*\}"):
        cell = re.sub(pat, lambda m: m.group(1) if m.groups() and m.group(1) else " ", cell)
    return cell


def rows(body):
    """{row label: [numbers]} for the rows of a tabular body."""
    out = {}
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
    for line in body.split(r"\\"):
        if r"\toprule" in line:                      # the header row
            continue
        cells = [c.strip() for c in line.replace(r"\addlinespace", "").replace(r"\midrule", "").split("&")]
        if len(cells) < 2:
            continue
        label = re.sub(r"\s+", " ", strip(cells[0])).strip() or f"(row {len(out)})"
        nums = [float(x.replace(",", "")) for c in cells[1:] for x in NUM.findall(strip(c))]
        if nums:
            key = label
            while key in out:
                key += "'"
            out[key] = nums
    return out


def paper_body(tex, kind, where, figures):
    if kind == "file":
        return (figures / where).read_text()
    i = tex.index(rf"\label{{{where}}}")
    j = tex.index(r"\end{table", i)
    return tex[i:j]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the output folder of reproduce/run_all.py")
    ap.add_argument("--tex", default=str(REPO / "paper" / "main.tex"))
    a = ap.parse_args()
    if not Path(a.tex).exists():
        raise SystemExit(f"{a.tex} missing: point --tex at the main.tex of the paper's LaTeX source (not in this "
                         f"repository)")
    tex = Path(a.tex).read_text()
    figures = Path(a.tex).with_name("Figures")
    supp = Path(a.tex).with_name("supp.tex")              # the appendix, \input at the end of main.tex
    if supp.exists():
        tex += "\n" + supp.read_text()
    worst = 0.0
    for name, (kind, where) in TABLES.items():
        f = Path(a.out) / name
        if not f.exists():
            print(f"{name:20s} not generated")
            continue
        body, ref = f.read_text(), paper_body(tex, kind, where, figures)
        if r"\toprule" not in body and r"\midrule" in ref:      # a body-only fragment: skip the paper's header rows
            ref = ref[ref.index(r"\midrule"):]
        new, old = rows(body), rows(ref)
        diffs, missing = [], []
        for lab, o in old.items():
            n = new.get(lab)
            if n is None or len(n) != len(o):
                missing.append(lab)
                continue
            diffs += [abs(x - y) / max(abs(y), 1e-12) for x, y in zip(n, o)]
        top = max(diffs) if diffs else float("nan")
        worst = max(worst, top) if diffs else worst
        print(f"{name:20s} {len(old) - len(missing)}/{len(old)} rows compared, max relative difference "
              f"{100 * top:.2f}%" + (f"; unmatched rows: {missing}" if missing else ""))
    print(f"worst: {100 * worst:.2f}%")


if __name__ == "__main__":
    main()
