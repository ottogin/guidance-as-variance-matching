#!/usr/bin/env python
"""fig:samples-sd35m{,-2} (and the photo rows of the teaser): SD 3.5 Medium on hand-picked held-out COCO captions,
one row per caption with the eight settings of the user study (tab:user-study).

    python reproduce/figures/fig_samples_sd35m.py --out paper-figures/ [--teaser]

The appendix rows are the user study's images: the sample.py arms of reproduce/paper.py's user-study renders
(runs/sd35m/coco-userstudy/samples/<arm>/<row>.png, 480 px, downscaled to SIDE px for the PDF), or with --renders the
study's own JPEGs (<dir>/<arm>/<coco id>.jpg, rendered by a separate pipeline; 2-8/255 mean pixel difference): our constant (w = 4.14) and schedule, the FID- and FD_DINOv2-tuned constants (1.6, 2.4), and the
interval, linear and TV-CFG schedules at the released w = 4.5, plus the released constant w = 4.5 itself (arm c4.5,
rendered with the same protocol and added to the study as an eighth setting). COCO captions 5,705-6,204,
40 Euler steps, one initial noise per caption shared by all eight. The rows are the first N_ROWS captions shown to
raters, in COCO order and not selected (the captions hidden from raters are skipped); the study's row order, hidden
rows and prompts are reproduce/data/user_study_rows.json.
Our two settings are framed in black. Each caption's row is one page of samples-sd35m-rows.pdf (a JPEG strip of the
eight cells), so App. F is a handful of files rather than 800 separate images.
--teaser also writes the teaser's photo rows from the sample.py arms w1.6, w2.4 and vm-global (a re-render with the
same seeds: 2-8/255 mean pixel difference from the study's images). Writes, under --out:
  samples-sd35m-rows.pdf           one page per caption: the eight cells side by side, ours framed
  samples-sd35m-1.tex              the first page's grid (supp.tex gives it the full caption)
  samples-sd35m-rest.tex           every later page, each a \twocolumn[...] figure with a "continued" caption
  teaser/p<i>-{fid,fdd,ours}.jpg   the three photo rows of the teaser (fig:teaser), with --teaser
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402

ARMS = (("fid", "w1.6"), ("fdd", "w2.4"), ("ours", "vm-global"))    # the teaser's, from sample.py
FIG1C = (5705, 5707, 5772)
ROWS_JSON = REPO / "reproduce" / "data" / "user_study_rows.json"
USERSTUDY_TAG = "userstudy"
# (file name, the study's arm folder, column head), in the order of the grid
STUDY = (("ours", "c4.14", r"Ours\\$w=4.14$"), ("ours-sched", "trace", r"Ours\\$w(\sigma)$"),
         ("fid", "c1.6", r"FID\\$w=1.6$"), ("fdd", "c2.4", r"FD$_{\mathrm{DINOv2}}$\\$w=2.4$"),
         ("pub", "c4.5", r"Released\\$w=4.5$"),
         ("interval", "pub-interval", r"Interval\\$w=4.5$"), ("linear", "pub-linear", r"Linear\\$w=4.5$"),
         ("tvcfg", "pub-tvcfg", r"TV-CFG\\$w=4.5$"))
# the study's arm -> the sample.py arm that renders it (sample.py --tag userstudy --n 500 --fp32-state; plan.py
# USERSTUDY_ARMS has the flags)
SAMPLE_ARM = {"c4.14": "vm-global-f32", "trace": "vm-schedule-f32", "c1.6": "w1.6-f32", "c2.4": "w2.4-f32",
              "c4.5": "pub-f32", "pub-interval": "pub-interval-f32", "pub-linear": "pub-linear-f32",
              "pub-tvcfg": "pub-tvcfg-f32"}
N_ROWS = 100  # captions in the appendix
ROWS = 8      # captions a page
ROWS_FIRST = 7  # the first page also holds the section heading and the full caption
N_OURS = 2    # the first N_OURS settings of STUDY are ours, framed in black
SIDE = 256    # px; a cell is ~0.85 in wide, so ~300 dpi
GAP = 3       # px of white between cells (~0.7 pt at \textwidth)
FRAME = 4     # px of black frame around our cells, drawn over their outer edge (~1 pt)
ROWS_PDF = "samples-sd35m-rows.pdf"
QUALITY = 90


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def study_keys():
    """The first N_ROWS captions shown to raters, in COCO order."""
    key = json.load(open(ROWS_JSON))
    pos2coco = {r["pos"]: r["coco"] for r in key["rows"]}
    hidden = {pos2coco[p] for p in key["hidden"]}
    return [k for k in sorted(pos2coco.values()) if k not in hidden][:N_ROWS]


def row_strip(cells):
    """The eight cells of one caption side by side, GAP px apart, with one black frame around our N_OURS cells."""
    from PIL import Image, ImageDraw
    n = len(cells)
    strip = Image.new("RGB", (n * SIDE + (n - 1) * GAP, SIDE), "white")
    for i, im in enumerate(cells):
        strip.paste(im.resize((SIDE, SIDE), Image.LANCZOS), (i * (SIDE + GAP), 0))
    right = N_OURS * SIDE + (N_OURS - 1) * GAP - 1
    ImageDraw.Draw(strip).rectangle((0, 0, right, SIDE - 1), outline="black", width=FRAME)
    return strip


def grid_tex(rows, captions, heads=None):
    r"""One page of the appendix grid: the column heads, then per caption its strip (a page of ROWS_PDF) with the
    caption underneath. rows = [(coco id, page of ROWS_PDF)]. The heads sit over the strip's cells: each is a cell's
    share of \textwidth, spread by \hfill. heads: the column heads (default STUDY's)."""
    n = len(STUDY)
    width = n * SIDE + (n - 1) * GAP
    cell = r"\makebox[%.6f\textwidth]{\shortstack{%%s}}" % (SIDE / width - 5e-7)
    head = r"\hfill".join(cell % h for h in (heads or [h for _, _, h in STUDY]))
    out = [
        "% Generated by reproduce/figures/fig_samples_sd35m.py -- edit the script, not this file.",
        r"{\scriptsize\centering%",
        r"\hbox to \textwidth{%s}\par\vspace{3pt}" % head,
    ]
    for k, page in rows:
        out.append(r"\includegraphics[width=\textwidth,page=%d]{Figures/%s}\par\vspace{1pt}" % (page, ROWS_PDF))
        out.append(r"{\itshape %s}\par\vspace{5pt}" % tex_escape(captions[k]))
    out += [r"}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the paper's Figures folder (or a scratch copy)")
    ap.add_argument("--teaser", action="store_true", help="also write the teaser's rows (needs the sample.py arms)")
    ap.add_argument("--renders", help="the study's own JPEG renders (<dir>/<arm>/<coco id>.jpg) instead of the "
                                      "sample.py arms")
    a = ap.parse_args()
    out = Path(a.out)
    captions = {int(i): t for i, t in json.load(open(ROWS_JSON))["prompts"].items()}

    def source(arm, k):
        if a.renders:
            return Path(a.renders) / arm / f"{k:06d}.jpg"
        return runs.root() / "sd35m" / f"coco-{USERSTUDY_TAG}" / "samples" / SAMPLE_ARM[arm] / f"{k:05d}.png"
    from PIL import Image
    heads = [h for _, _, h in STUDY]
    if not a.renders:                  # our constant as this rebuild fitted it (the study's renders used 4.14)
        wj = runs.root() / "sd35m" / f"coco-{USERSTUDY_TAG}" / "samples" / SAMPLE_ARM[STUDY[0][1]] / "weights.json"
        heads[0] = r"Ours\\$w=%.2f$" % json.load(open(wj))["w_mean"]
    out.mkdir(parents=True, exist_ok=True)
    keys = study_keys()
    strips = [row_strip([Image.open(source(arm, k)).convert("RGB") for _, arm, _ in STUDY]) for k in keys]
    strips[0].save(out / ROWS_PDF, save_all=True, append_images=strips[1:], resolution=300, quality=QUALITY)
    rows = [(k, i) for i, k in enumerate(keys, 1)]
    pages = [rows[:ROWS_FIRST]] + [rows[i:i + ROWS] for i in range(ROWS_FIRST, len(rows), ROWS)]
    (out / "samples-sd35m-1.tex").write_text(grid_tex(pages[0], captions, heads))
    rest = ["% Generated by reproduce/figures/fig_samples_sd35m.py -- edit the script, not this file.",
            "% Pages 2.. of App. F's SD 3.5 Medium rows; page 1 and its caption are in supp.tex."]
    done = len(pages[0])
    for i, page in enumerate(pages[1:], 2):
        (out / f"samples-sd35m-{i}.tex").write_text(grid_tex(page, captions, heads))
        rest += [r"\twocolumn[{%", r"\centering", r"\input{Figures/samples-sd35m-%d.tex}" % i,
                 r"\captionsetup{hypcap=false}%  \captionof outside a float",
                 r"\captionof{figure}{More samples from SD~3.5 Medium, continued from \cref{fig:samples-sd35m} "
                 r"(captions %d--%d of %d); same settings and layout.}" % (done + 1, done + len(page), len(keys)),
                 "}]", ""]
        done += len(page)
    (out / "samples-sd35m-rest.tex").write_text("\n".join(rest))
    for stale in out.glob("samples-sd35m-*.tex"):
        tail = stale.stem.rsplit("-", 1)[1]
        if tail.isdigit() and int(tail) > len(pages):
            stale.unlink()
    print(f"appendix: {len(keys)} captions on {len(pages)} pages -> {out}/samples-sd35m-*.tex")
    if a.teaser:
        from PIL import Image
        src = {name: runs.root() / "sd35m" / "coco" / "samples" / arm for name, arm in ARMS}
        for i, k in enumerate(FIG1C, 1):
            for name, _ in ARMS:
                dst = out / "teaser" / f"p{i}-{name}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.open(src[name] / f"{k:05d}.png").convert("RGB").save(dst, quality=QUALITY)
        print(f"teaser rows {FIG1C} -> {out}/teaser/")


if __name__ == "__main__":
    main()
