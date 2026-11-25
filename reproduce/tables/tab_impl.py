#!/usr/bin/env python
"""tab:impl-models (= tab:impl-data): models, samplers and data, one 9-column table, from the model Specs.

    python reproduce/tables/tab_impl.py --out paper-tables/

Writes impl.tex, the tabular body (between \\midrule and \\bottomrule) in the paper's order and columns: model,
reference, checkpoint, sampler and steps, output, published w, data, calibration n, evaluation n. The checkpoint,
step count, published weight and calibration / evaluation counts come from the Specs in vmg/models; the citations
and the short display forms of samplers, outputs and data are the paper's (DISPLAY below), and every count they
print is checked against the Spec. tests/test_specs.py checks the paper's copy against the Specs the other way.
Needs the model packages (torch), since the Specs live next to their adapters.
"""
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ORDER = ["sd15", "sd35m", "sd35l", "flux2-klein", "svd", "wan", "cogvideox", "cosmos", "audioldm2", "sao", "mdm"]
BREAK_AFTER = {"flux2-klein", "cosmos", "sao", "mdm"}        # \addlinespace between modalities

# key -> (citation, sampler as printed, output as printed, data, calib n as printed, eval n as printed, checkpoint as
# printed or None for the Spec's). Counts are written as the paper writes them and checked against the Spec.
DISPLAY = {
    "sd15": ("rombach2022ldm", r"PNDM \cite{liu2022pndm}", r"$480^2$ image", "COCO", "5,705", "10,000", None),
    "sd35m": ("esser2024sd3", "Euler", r"$480^2$ image", "COCO", "5,705", "10,000", None),
    "sd35l": ("esser2024sd3", "Euler", r"$480^2$ image", "COCO", "5,705", "10,000", None),
    "flux2-klein": ("bfl2026flux2klein", "Euler", r"$1024^2$ image", "MJHQ-30K", "1,024", "10,000", None),
    "svd": ("blattmann2023svd", r"Euler, Karras $\sigma$ \cite{karras2022edm}", r"14 frames, $1024{\times}576$",
            "UCF-101", "1,024", "2,048", None),
    "wan": ("wan2025wan", r"UniPC \cite{zhao2023unipc}", r"81 frames, $832{\times}480$", "OpenVid-1M", "256", "1,024",
            None),
    "cogvideox": ("yang2025cogvideox", r"DDIM \cite{song2021ddim}", r"49 frames, $720{\times}480$", "OpenVid-1M",
                  "256", "1,024", None),
    "cosmos": ("nvidia2025cosmos", "released", r"13-frame chunks, $320{\times}256$", "Bridge", "1,024 chunks",
               "100 episodes", "nvidia/Cosmos-Predict2.5-2B/robot/action-cond"),
    "audioldm2": ("liu2024audioldm2", "DDIM", r"10.24\,s clip", "AudioCaps", "2,223", "4,411", None),
    "sao": ("evans2025sao", r"DPM-Solver++ 3M SDE \cite{lu2022dpmsolverpp}", r"47\,s stereo clip", "Song Describer",
            "356", "586", None),
    "mdm": ("tevet2023mdm", r"DDPM \cite{ho2020ddpm}", "KIT-ML pose sequence", "KIT-ML", "864", r"$20\times1{,}000$",
            None),
}
STEPS_SUFFIX = {"cosmos": " per chunk"}
SHORT_PUB = {"svd": r"$1\to3$"}           # a weight that is not one constant: its short form in the table
PUB_PRINT = {"flux2-klein": "4.0"}       # as printed
NAME = {"wan": "Wan2.1 1.3B"}


def tex(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def pub(spec):
    if math.isfinite(spec.published_w):
        return PUB_PRINT.get(spec.key, f"{spec.published_w:g}")
    return SHORT_PUB[spec.key]


def first_int(s):
    import re
    m = re.search(r"\d[\d,]*", s.replace("{,}", ","))
    return int(m.group(0).replace(",", "")) if m else None


def row(spec):
    cite, sampler, output, data, n_cal, n_ev, ckpt = DISPLAY[spec.key]
    sp = spec.datasets[spec.default_dataset]
    # the counts printed are the Spec's (Song Describer prints captions, the Spec counts their 708 clips; MDM's
    # evaluation is 20 replications, which the Spec counts)
    if spec.key != "sao":
        assert first_int(n_cal) == sp.n_calib, (spec.key, n_cal, sp.n_calib)
    assert first_int(n_ev) == sp.n_eval or (spec.key == "mdm" and sp.n_eval == 20), (spec.key, n_ev, sp.n_eval)
    ckpt = ckpt or spec.checkpoint
    assert ckpt.startswith(spec.checkpoint), (spec.key, ckpt, spec.checkpoint)
    steps = f"{spec.steps:,}" + STEPS_SUFFIX.get(spec.key, "")
    return (f"{NAME.get(spec.key, tex(spec.name))} & \\cite{{{cite}}} & \\nolinkurl{{{ckpt}}} & {sampler}, {steps} & "
            f"{output} & {pub(spec)} & {data} & {n_cal} & {n_ev} \\\\")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output folder")
    a = ap.parse_args()
    from vmg.models import get
    lines = []
    for k in ORDER:
        lines.append(row(get(k).spec))
        if k in BREAK_AFTER:
            lines.append(r"\addlinespace")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "impl.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'impl.tex'} ({len(ORDER)} models)")


if __name__ == "__main__":
    main()
