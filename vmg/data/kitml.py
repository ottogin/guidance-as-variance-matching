"""KIT-ML for MDM (paper tab:impl-data), read through MDM's own data code, which is imported from its repository.

Setup (manual; KIT-ML and the checkpoint come from Google Drive links that cannot be scripted):
    git clone https://github.com/GuyTevet/motion-diffusion-model $VMG_MDM_ROOT
    git -C $VMG_MDM_ROOT checkout ef8edce                                       # the pinned commit (2025-09-30)
    cd $VMG_MDM_ROOT && bash prepare/download_smpl_files.sh && bash prepare/download_glove.sh \
        && bash prepare/download_t2m_evaluators.sh
    # KIT-ML as HumanML3D's README distributes it (texts.rar, new_joint_vecs.rar, new_joints.rar, Mean/Std,
    # train/val/test.txt), unpacked into $VMG_MDM_ROOT/dataset/KIT-ML: 6,018 texts, 7,812 new_joint_vecs
    # the kit_trans_enc_512 checkpoint (MDM README) into $VMG_MDM_ROOT/save/kit_trans_enc_512/
    # CLIP ViT-B/32 is fetched by the clip package on first use, or read from $VMG_CLIP_CACHE

Calibration rows (864): MDM's trainer view of the validation split (288 motions), read three times in dataset
order with seed 0, so each motion appears with three caption draws and crops: row = rep * 288 + motion.
Evaluation rows: the 20 replications of MDM's own text-to-motion evaluation, row r at seed 10 + r (dataset
"kitml"); MultiModality is its own protocol (mm_short: 100 prompts x 30 generations, 5 replications, seeds 10-14),
the dataset "kitml-mm".
"""
import contextlib
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

CKPT = "save/kit_trans_enc_512/model000400000.pt"
REPS, N_EVAL_REPS, N_MM_REPS, EVAL_SEED0 = 3, 20, 5, 10
N_FRAMES = 196


def root():
    r = os.environ.get("VMG_MDM_ROOT")
    assert r and Path(r).is_dir(), "set VMG_MDM_ROOT to a motion-diffusion-model checkout (see vmg/data/kitml.py)"
    return Path(r)


@contextlib.contextmanager
def mdm_env():
    """MDM's code on sys.path and its root as the working directory (its dataset paths are relative)."""
    r = str(root())
    if r not in sys.path:
        sys.path.insert(0, r)
    old = os.getcwd()
    os.chdir(r)
    try:
        yield Path(r)
    finally:
        os.chdir(old)


@lru_cache(maxsize=2)
def calib_set(seed=0, reps=REPS):
    """(x0 [n, 251, 1, 196] float32, lengths [n], captions, tokens) as MDM's trainer sees the validation split:
    one caption drawn per motion, the length cropped to a multiple of 4 at a random offset, Z-normalised by the
    shipped Mean/Std, zero-padded to 196 frames; read `reps` times in dataset order after seeding."""
    import torch
    with mdm_env():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        from data_loaders.humanml.data import dataset as D
        ds = D.KIT(mode="train", split="val")
        items = [ds[i] for _ in range(reps) for i in range(len(ds))]
    x0 = np.stack([torch.tensor(b[4].T).float().unsqueeze(1).numpy() for b in items])
    return x0, np.array([int(b[5]) for b in items]), [b[2] for b in items], [b[6] for b in items]


def rows(split, mm=False):
    if split == "calib":
        return list(range(len(calib_set()[1])))
    assert split == "eval", split
    return list(range(N_MM_REPS if mm else N_EVAL_REPS))


def check():
    """The files MDM's KIT-ML code and evaluation read; raises with the missing ones."""
    r = root()
    want = ["dataset/KIT-ML/Mean.npy", "dataset/KIT-ML/Std.npy", "dataset/KIT-ML/val.txt", "dataset/KIT-ML/test.txt",
            "dataset/KIT-ML/texts", "dataset/KIT-ML/new_joint_vecs", "dataset/kit_opt.txt", "glove", "kit", CKPT,
            str(Path(CKPT).parent / "args.json")]
    missing = [w for w in want if not (r / w).exists()]
    assert not missing, f"missing under {r}: {missing}"
