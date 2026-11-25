"""The guided pairs of fig:wmse-published and fig:wmse-sigma, each one compute_wmse.py run.

A pair is a registry model on a dataset (the dataset keys of its Spec). `pub` is the published weight in the
paper's D0 + w (D1 - D0) form, None where the pair has none. SVD's per-frame ramp 1 -> 3 enters the figure as its
frame average. `old` is the pair's row in the earlier results/wmse/summary.json, kept for provenance and for
reproduce/checks/legacy_wmse.py, which turns those stored rows (not part of this repository) into wmse.json files
to check the figures.
"""
from dataclasses import dataclass

import numpy as np

AUTOG, CFG, T2I, AVM, CROSS = "autoguidance", "class CFG", "text-to-image CFG", "audio, video, motion", "cross-model"
SVD_RAMP = float(np.mean(np.linspace(1, 3, 14)))


@dataclass(frozen=True)
class Pair:
    model: str
    dataset: str
    group: str
    pub: object              # float or None
    old: str
    label: str = ""          # printed next to the pair in fig:wmse-published ("" = unnamed)


def _ag512(k, fid, dino, old):
    return [Pair(f"edm2-512-ag-{k}-fid", "imagenet512", AUTOG, fid, f"IN512 autog {old} (fid)"),
            Pair(f"edm2-512-ag-{k}-dino", "imagenet512", AUTOG, dino, f"IN512 autog {old} (dino)")]


# the guide-checkpoint sweep of IN64 autoguidance: XS at these training images (kimg x 1000)
GUIDE_SWEEP = ["0033554", "0067108", "0134217", "0268435", "0536870", "2147483"]

PAIRS = [
    # ImageNet autoguidance (EDM2): the strong net guided away from a smaller or less-trained one. Published
    # weights are NVIDIA's autoguidance presets (edm2 generate_images.py).
    *_ag512("s-xs", 2.10, 2.45, "S<-XS"),
    *_ag512("su-xsu", 2.85, 2.90, "S-uncond<-XS-uncond"),
    *_ag512("xxl-m", 2.05, 2.30, "XXL<-M"),
    Pair("edm2-64-ag-s-xs-fid", "imagenet64", AUTOG, 1.7, "IN64 autog S<-XS (fid)"),
    Pair("edm2-64-ag-s-xs-dino", "imagenet64", AUTOG, 2.2, "IN64 autog S<-XS (dino)"),
    *[Pair(f"edm2-64-ag-s-xs-{k}", "imagenet64", AUTOG, None, f"IN64 autog S<-XS@{k} (n=50k)") for k in GUIDE_SWEEP],
    # class-conditional CFG
    Pair("edm2-64-xs-cfg", "imagenet64", CFG, None, "IN64 CFG EDM2-XS"),
    Pair("sit-xl2", "imagenet256", CFG, 1.5, "IN256 SiT-XL/2 CFG"),
    Pair("dit-xl2", "imagenet256", CFG, 1.5, "IN256 DiT-XL/2 CFG"),
    Pair("maskdit", "imagenet256", CFG, 1.5, "IN256 MaskDiT CFG"),
    Pair("repa-sit-xl2", "imagenet256", CFG, 1.35, "IN256 REPA SiT-XL/2 CFG"),
    Pair("uvit-l2", "imagenet256", CFG, 1.4, "IN256 U-ViT-L/2 CFG"),
    Pair("uvit-h2", "imagenet256", CFG, 1.4, "IN256 U-ViT-H/2 CFG"),
    Pair("cifar10-edm-vp", "cifar10", CFG, None, "CIFAR-10 EDM VP CFG"),
    # text-to-image CFG on DOCCI (the n=2,000 selection)
    Pair("sd15", "docci", T2I, 7.5, "SD1.5 CFG, DOCCI full caption"),
    Pair("sd15", "docci-short", T2I, 7.5, "SD1.5 CFG, DOCCI first sentence"),
    Pair("sd14", "docci", T2I, 7.5, "sd15-v14 CFG, DOCCI"),
    Pair("sd13", "docci", T2I, 7.5, "sd15-v13 CFG, DOCCI"),
    Pair("commoncanvas", "docci", T2I, 7.5, "sd15-cc CFG, DOCCI"),
    Pair("sd21", "docci", T2I, 7.5, "sd15-sd21 CFG, DOCCI"),
    Pair("bksdm", "docci", T2I, 7.5, "sd15-bksdm CFG, DOCCI"),
    Pair("sd35m", "docci", T2I, 4.5, "sd35-medium CFG, DOCCI"),
    Pair("sd3m", "docci", T2I, 7.0, "sd3-medium CFG, DOCCI"),
    # text-to-image CFG, the paper's calibration sets and the other DOCCI models
    Pair("sd15", "coco", T2I, 7.5, "SD1.5 COCO"),
    Pair("sd35m", "coco", T2I, 4.5, "SD3.5-M COCO (40 steps)"),
    Pair("chroma", "docci", T2I, 5.0, "Chroma DOCCI"),
    Pair("cogview4", "docci", T2I, 5.0, "CogView4 DOCCI"),
    Pair("ovis", "docci", T2I, 5.0, "Ovis DOCCI"),
    Pair("sana", "docci", T2I, 4.5, "Sana DOCCI"),
    Pair("zimage", "docci", T2I, 6.0, "Z-Image DOCCI"),
    Pair("flux2-klein", "mjhq", T2I, 4.0, "FLUX.2-klein MJHQ"),
    Pair("sd35l", "coco", T2I, 3.5, "SD3.5-L COCO"),
    Pair("sana", "mjhq", T2I, 4.5, "Sana MJHQ"),
    Pair("sd35m", "mjhq", T2I, 4.5, "SD3.5-M MJHQ"),
    # audio, video and motion: the paper's main models
    Pair("audioldm2", "audiocaps", AVM, 3.5, "AudioLDM 2 (AudioCaps)", "AudioLDM 2"),
    Pair("svd", "ucf101", AVM, SVD_RAMP, "SVD img2vid", "SVD"),
    Pair("mdm", "kitml", AVM, 2.5, "MDM KIT-ML (text-to-motion)", "MDM"),
    Pair("wan", "openvid", AVM, 5.0, "Wan2.1-T2V-1.3B (OpenVid)", "Wan 2.1"),
    Pair("cogvideox", "openvid", AVM, 6.0, "CogVideoX-2B (OpenVid)", "CogVideoX"),
    Pair("sao", "songdescriber", AVM, 7.0, "Stable Audio Open (Song Describer)", "Stable Audio Open"),
    Pair("cosmos", "bridge", AVM, 1.0, "Cosmos-Predict2.5 (Bridge, world model)", "Cosmos"),
]

# fig:wmse-sigma only: one cross-model pair (SD 1.5 guided away from CommonCanvas) and SVD's first frame
EXTRA = [
    Pair("cross-sd15-cc", "docci", CROSS, None, "cross: sd15 <- sd15-cc (cond vs cond)"),
    Pair("cross-sd15-cc", "coco", CROSS, None, "cross: SD1.5 <- CommonCanvas (COCO)"),
    Pair("svd-frame1", "ucf101", AVM, 1.0, "SVD frame 1"),
]

HIGHLIGHT = PAIRS[[p.old for p in PAIRS].index("SD3.5-M COCO (40 steps)")]       # outlined in fig:wmse-published


def find(old):
    return next(p for p in PAIRS + EXTRA if p.old == old)


def runs_of(pairs):
    """The distinct (model, dataset) runs behind `pairs`."""
    return list(dict.fromkeys((p.model, p.dataset) for p in pairs))
