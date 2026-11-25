#!/usr/bin/env python
"""tab:pub-all: released guidance (Pub.) against our variance-matched schedule (Ours) on the eleven models.

    python reproduce/tables/tab_pub_all.py --out paper-tables/ [--allow-missing]

One block per modality, each with its own metric columns, and a Pub. row and an Ours row per model; the world model
and the motion model share the last block. Values come from
runs/<model>/<dataset>/metrics/compare-vm-schedule-vs-pub.json (reproduce/experiments/compare.py). An Ours row is
shaded (\\oursrow, as tab:pub-sched) and its numbers coloured by their relative change against Pub. on the printed
values, with the thresholds of reproduce/tables/_tex.py: vmgood where better and vmbad where worse beyond 1%, bold beyond 5%, black within 1%.
There are no stars. Latte-XL/2 is not in this table (tab_extra_latte.py). Writes pub-all.tex, everything between
the table's \\label and its \\end{table}.

Every modality's metrics come from its module in vmg/metrics (image, video, audio, motion).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.tables._tex import compare_json, fmt, shade  # noqa: E402

U, D = r"$\uparrow$", r"$\downarrow$"


def IMG(m, d, name, fd=2):
    """fd: decimals of FD_DINOv2 as the paper prints it for this model."""
    return (m, d, name, [("fid", 2), ("precision_inception", 3), ("recall_inception", 3), ("fd_dinov2", fd),
                         ("precision_dinov2", 3), ("recall_dinov2", 3), ("clip", 4)])


I3D = [("fvd", 1), ("precision_i3d", 3), ("recall_i3d", 3), ("density_i3d", 3), ("coverage_i3d", 3)]
# (title, column headers, [(model, dataset, name shown, one (key, decimals) or None per column)])
BLOCKS = [
    ("Text-to-image", [f"FID{D}", f"P{U}", f"R{U}", f"FD{D}", f"P{U}", f"R{U}", f"CLIP{U}"],
     [IMG("sd15", "coco", "SD 1.5"), IMG("sd35m", "coco", "SD 3.5 Medium", fd=1),
      IMG("sd35l", "coco", "SD 3.5 Large"), IMG("flux2-klein", "mjhq", "FLUX.2 [klein] 4B", fd=1)]),
    ("Video", [f"FVD{D}", f"P{U}", f"R{U}", f"Dens.{U}", f"Cov.{U}", f"CLIP-T{U}"],
     [("svd", "ucf101", "SVD", [*I3D, None]),
      ("wan", "openvid", "Wan2.1 1.3B", [*I3D, ("clip_t", 2)]),
      ("cogvideox", "openvid", "CogVideoX-2B", [*I3D, ("clip_t", 2)])]),
    ("Audio", [rf"FAD$^\ddagger${D}", f"KL{D}", f"CLAP{U}", f"P{U}", f"R{U}"],
     [("audioldm2", "audiocaps", "AudioLDM 2",
       [("fad", 2), ("kl", 2), ("clap", 3), ("precision_passt", 3), ("recall_passt", 3)]),
      ("sao", "songdescriber", "Stable Audio Open", [("fd_openl3", 2), ("kl", 3), ("clap", 4), None, None])]),
]
# the image block's header has a second level, the feature space of columns 3-5 and 6-8
IMAGE_HEAD = [r"\multicolumn{2}{@{}l}{\textit{Text-to-image}} & \multicolumn{3}{c}{Inception} & "
              r"\multicolumn{3}{c}{DINOv2} & \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}"]
# printed in the free first cell of a model's Ours row
NOTES = {"Stable Audio Open": r"\color{black!60}\,$^\ddagger$FD$_\mathrm{OpenL3}$"}
# the last block holds two models side by side; the world model's name is broken over its two rows, which is what
# makes room for four motion columns (so Pub./Ours sit left of the blocks above here)
WORLD = ("cosmos", "bridge", ("Cosmos-Predict", "2.5"), [f"FVD{D}", f"PSNR{U}"], [("fvd", 1), ("psnr", 2)])
MOTION = ("mdm", "kitml", "MDM", [f"FID{D}", f"MM-Dist{D}", f"R-prec.{U}", f"MMod.{U}"],
          [("mdm_fid", 3), ("mm_dist", 3), ("r_precision3", 3), ("multimodality", 2)])

SETUP = [r"\scriptsize\setlength{\tabcolsep}{1.5pt}\renewcommand{\arraystretch}{0.85}\setlength{\lineskip}{0pt}",
         # one name column for every block (the widest name), so Pub. / Ours line up down the table
         r"\def\namew{0pt}\def\trynm#1{\setbox0\hbox{#1}\ifdim\wd0>\namew\relax\edef\namew{\the\wd0}\fi}%"]
LEAD = r"@{}l@{\ \,}l@{\hspace{3pt}\extracolsep{\fill}}"            # name, Pub. / Ours, then the stretch


def hd(h):
    return rf"\hd{{{h}}}"


def pair(rec, slot):
    """(Pub., Ours, decimals) for one slot, Ours coloured on the printed values; None where there is no value."""
    if slot is None or rec is None or slot[0] not in rec:
        return None
    key, digits = slot
    a, b = round(rec[key]["a"], digits), round(rec[key]["b"], digits)
    sa, sb, s = fmt(a, digits), fmt(b, digits), shade(key, a, b)
    if s != "vmsame":
        col = "vmgood" if "better" in s else "vmbad"
        sb = rf"{{\color{{{col}}}{sb}}}" if s.endswith("lt") else rf"{{\bfseries\color{{{col}}}{sb}}}"
    return sa, sb, digits


def two_rows(cols):
    """The Pub. cells and the Ours cells of one model. cols: per column, pair() and the decimals of the column;
    a value printed with fewer decimals is padded with \\dz so the points line up."""
    pub = [p[0] + r"\dz" * (mx - p[2]) if p else r"\na" for p, mx in cols]
    our = [p[1] + r"\dz" * (mx - p[2]) if p else r"\na" for p, mx in cols]
    return " & ".join(pub), " & ".join(our)


def block(title, heads, models, get, first=False):
    pairs = {name: [pair(get(m, d), s) for s in slots] for m, d, name, slots in models}
    mx = [max((p[i][2] for p in pairs.values() if p[i]), default=0) for i in range(len(heads))]
    L = [r"\toprule" if first else r"\midrule"]
    if first:
        L += IMAGE_HEAD + [" & & " + " & ".join(map(hd, heads)) + r" \\"]
    else:
        L.append(rf"\multicolumn{{2}}{{@{{}}l}}{{\textit{{{title}}}}} & " + " & ".join(map(hd, heads)) + r" \\")
    for name, ps in pairs.items():
        pub, our = two_rows(list(zip(ps, mx)))
        L += [rf"\nm{{{name}}} & Pub. & {pub} \\", r"\oursrow" + NOTES.get(name, "") + rf" & Ours & {our} \\"]
    spec = LEAD + "r" * len(heads) + "@{}"
    return "\n".join([rf"\begin{{tabular*}}{{\linewidth}}{{{spec}}}", *L, r"\end{tabular*}"])


def last_block(get):
    (wm, wd, wname, whead, wslots), (mm, md, mname, mhead, mslots) = WORLD, MOTION
    wp, mp = [pair(get(wm, wd), s) for s in wslots], [pair(get(mm, md), s) for s in mslots]
    (wpub, wour), (mpub, mour) = (two_rows([(p, p[2] if p else 0) for p in ps]) for ps in (wp, mp))
    gap = r"@{\hspace{2.5pt}}"
    spec = LEAD + "r" * len(whead) + gap + r"!{\color{black!35}\vrule}" + gap + "l" + "r" * len(mhead) + "@{}"
    L = [r"\midrule",
         r"\multicolumn{2}{@{}l}{\textit{World model}} & " + " & ".join(map(hd, whead)) + r" & \textit{Motion} & "
         + " & ".join(map(hd, mhead)) + r" \\",
         rf"{wname[0]} & Pub. & {wpub} & {mname} & {mpub} \\", rf"\oursrow{wname[1]} & Ours & {wour} & & {mour} \\",
         r"\bottomrule"]
    # tighter padding in this block only: it is the full width of the column
    return "\n".join([r"{\setlength{\tabcolsep}{1pt}\begin{tabular*}{\linewidth}{" + spec + "}", *L,
                      r"\end{tabular*}}"])


def render(get):
    """The table body. get(model, dataset) -> the metrics of compare-vm-schedule-vs-pub.json, or None."""
    names = "".join(rf"\trynm{{{name}}}" for _, _, models in BLOCKS for _, _, name, _ in models)
    setup = SETUP + [names + r"\def\nm#1{\makebox[\namew][l]{#1}}%"]
    blocks = [block(t, h, m, get, first=i == 0) for i, (t, h, m) in enumerate(BLOCKS)]
    return "\n".join(setup) + "\n" + "\n\n".join(blocks + [last_block(get)]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    runs = [(m, d) for _, _, models in BLOCKS for m, d, _, _ in models] + [WORLD[:2], MOTION[:2]]
    recs = {md: compare_json(*md) for md in runs}
    missing = sorted(f"{m}/{d}" for (m, d), r in recs.items() if r is None)
    assert not missing or a.allow_missing, f"no compare-vm-schedule-vs-pub.json for {missing}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pub-all.tex").write_text(render(lambda m, d: recs[(m, d)]))
    print(f"wrote {out / 'pub-all.tex'}" + (f" (missing: {missing})" if missing else ""))


if __name__ == "__main__":
    main()
