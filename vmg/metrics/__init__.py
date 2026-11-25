"""The paper's metrics, one module per modality (image.py, video.py, audio.py, motion.py).

A modality module provides
    spaces(model, dataset)            {feature space: dict(fd=<metric name> or None, prdc=bool)}: each space's
                                      Frechet distance and, if prdc, precision / recall / density / coverage
                                      (k = 3) named <stat>_<space>
    reference(model, dataset, ids)    the real set: ({space: features [n, d]}, {name: per-row values [n]})
    score_arm(model, dataset, ids, src, ref)
                                      one sampled arm: ({space: features [n, d]}, {name: per-row values [n]})
Per-row values are metrics whose value is their mean over rows (CLIP score, PSNR, R-precision, ...);
reproduce/experiments/evaluate.py reduces both kinds, and compare.py gives every difference a paired interval.
"""
import importlib

MODULES = {"image": "image", "video": "video", "world": "video", "audio": "audio", "motion": "motion"}


def for_modality(modality):
    assert modality in MODULES, f"no metrics for modality {modality!r}"
    return importlib.import_module(f"vmg.metrics.{MODULES[modality]}")
