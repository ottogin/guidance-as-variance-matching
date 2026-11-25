"""Where checkpoints come from: the Hugging Face hub by default, or a local mirror.

Set VMG_WEIGHTS to a directory laid out as <VMG_WEIGHTS>/<org>/<repo>/ (a plain copy, or a tree of
symlinks, of the repository) to load without network access. diffusers' own offline mode still calls
the hub API, so a local directory is the reliable way to run on a cluster without outbound internet.
"""
import os
from pathlib import Path


def path(repo):
    root = os.environ.get("VMG_WEIGHTS")
    if root and (Path(root) / repo).exists():
        return str(Path(root) / repo)
    return repo
