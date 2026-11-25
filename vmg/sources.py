"""From the command line to a model adapter: presets, released checkpoints, or an adapter of your own.

    --model sd35m                                         a preset (vmg.models.REGISTRY); the pair is fixed
    --guidance cfg --model <diffusers repo id or dir>     CFG with one network: D1 the prompt, D0 the empty prompt
    --guidance cfg-pair --model <cond> --reference <uncond>     CFG with two networks (EDM2's)
    --guidance autoguidance --model <strong> --reference <weak>  autoguidance (Karras et al. 2024)
    --model mypkg.mymodule:MyAdapter                      any vmg.models.base.Model subclass

cfg loads the checkpoint through the adapter of its diffusers pipeline family (FAMILIES below; each was checked
against its released pipeline, see `--smoke`), with its own step count and weights. cfg-pair and autoguidance
take two EDM or EDM2 pickles: a path, a URL, or an EDM2 post-hoc reconstruction by name (e.g.
edm2-img64-s-1073741-0.045, fetched from NVIDIA). Anything else is an adapter: implement levels, rows, data, noise
and denoise_pair (vmg/models/base.py) and pass it as module:Class.

A folder of images, `--dataset folder:<dir>`, calibrates any of them (vmg/data/folder.py).
"""
import importlib
import inspect
import json
import re
from pathlib import Path

from .data import folder
from .models import base

KINDS = ("auto", "cfg", "cfg-pair", "autoguidance")
# auto (the default): a preset's own pair; with no --reference, cfg; with one, cfg-pair when the reference network is
# unconditional and autoguidance when it takes the same conditioning (read from the two pickles when they load)

# diffusers pipeline class -> (adapter module, adapter class) whose code runs that pipeline's network call
FAMILIES = {
    "StableDiffusion3Pipeline": ("vmg.models.sd", "SD35Medium"),
    "StableDiffusionPipeline": ("vmg.models.sd", "SD15"),
    "Flux2KleinPipeline": ("vmg.models.sd", "Flux2Klein"),
    "ChromaPipeline": ("vmg.models.t2i", "Chroma"),
    "OvisImagePipeline": ("vmg.models.t2i", "Ovis"),
    "ZImagePipeline": ("vmg.models.t2i", "ZImage"),
    "CogView4Pipeline": ("vmg.models.t2i", "CogView4"),
    "SanaPipeline": ("vmg.models.t2i", "Sana"),
}
PLUGIN = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")


def key_of(*names):
    """A run-folder name for checkpoints given by repo id, path or URL."""
    out = []
    for n in names:
        n = str(n).rstrip("/")
        n = n.rsplit("/", 1)[-1] if (n.startswith(("http://", "https://")) or Path(n).exists()) else n
        out.append(re.sub(r"[^A-Za-z0-9._-]+", "--", Path(n).stem if n.endswith((".pkl", ".pt")) else n))
    return "__".join(out)


def resolve(a):
    """The Model an argparse namespace of vmg.cli.parser names (see the module docstring)."""
    m = a.model
    if PLUGIN.fullmatch(m) and m not in base.REGISTRY:
        model = plugin(m)
    elif m in base.REGISTRY:
        assert not getattr(a, "reference", None), f"{m} is a preset: its reference model is fixed ({m!r})"
        assert not getattr(a, "steps", None), (f"{m} is a preset with the released step count; to calibrate another "
                                               f"one, load the checkpoint with --guidance and --steps")
        model = base.get(m)
    else:
        kind = a.guidance or "auto"
        if kind == "auto":
            kind = "pair" if a.reference else "cfg"
        if kind == "cfg":
            assert not a.reference, "--guidance cfg makes D0 from the empty prompt: drop --reference"
            model = diffusers_cfg(m, steps=a.steps, published_w=a.published_w, size=a.size)
        else:
            assert a.reference, f"--guidance {kind} needs --reference, the model D0"
            model = edm_pair(m, a.reference, "auto" if kind == "pair" else kind, a.dataset, steps=a.steps,
                             published_w=a.published_w)
    if a.dataset and folder.is_folder(a.dataset) and a.dataset not in model.spec.datasets:
        model.with_spec(datasets={a.dataset: folder.split(a.dataset), **model.spec.datasets})
        if getattr(a, "size", None):
            model.size = int(a.size)
    return model


def plugin(spec):
    mod, cls = spec.split(":")
    obj = getattr(importlib.import_module(mod), cls)
    assert isinstance(obj, type) and issubclass(obj, base.Model), f"{spec} is not a vmg.models.base.Model subclass"
    return obj()


# --- diffusers: CFG with the empty prompt ----------------------------------------------------------------------
def model_index(ckpt):
    """The pipeline's model_index.json (a local directory, $VMG_WEIGHTS/<repo>, or the hub)."""
    from .models import hub
    p = Path(hub.path(ckpt))
    hint = (f"{ckpt!r} is not a preset and has no readable diffusers model_index.json: for two networks pass "
            f"--reference (cfg-pair / autoguidance), for anything else an adapter as module:Class")
    if p.is_dir():
        if not (p / "model_index.json").exists():
            raise SystemExit(hint)
        return json.loads((p / "model_index.json").read_text())
    try:
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(ckpt, "model_index.json")).read_text())
    except Exception as e:                       # not on the hub, no network, not a repo id
        raise SystemExit(f"{hint} ({type(e).__name__})")


def scheduler_config(ckpt):
    from .models import hub
    p = Path(hub.path(ckpt))
    if p.is_dir():
        f = p / "scheduler" / "scheduler_config.json"
        return json.loads(f.read_text()) if f.exists() else {}
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(ckpt, "scheduler_config.json", subfolder="scheduler")).read_text())


def pipeline_defaults(cls_name):
    """(num_inference_steps, guidance_scale) defaults of a diffusers pipeline's __call__."""
    import diffusers
    sig = inspect.signature(getattr(diffusers, cls_name).__call__).parameters
    get = lambda k: sig[k].default if k in sig and sig[k].default is not inspect.Parameter.empty else None  # noqa
    return get("num_inference_steps"), get("guidance_scale")


def diffusers_cfg(ckpt, steps=None, published_w=None, size=None):
    """A released text-to-image checkpoint through the adapter of its pipeline family (FAMILIES)."""
    cls_name = model_index(ckpt)["_class_name"]
    if cls_name not in FAMILIES:
        raise SystemExit(f"{ckpt}: no adapter for {cls_name} (have: {', '.join(FAMILIES)}). Write one as a "
                         f"vmg.models.base.Model subclass and pass it as --model module:Class")
    if cls_name == "StableDiffusionPipeline":
        pt = scheduler_config(ckpt).get("prediction_type", "epsilon")
        assert pt == "epsilon", f"{ckpt} predicts {pt!r}; the SD 1.x adapter handles epsilon-prediction only"
    mod, cls = FAMILIES[cls_name]
    model = getattr(importlib.import_module(mod), cls)()
    d_steps, d_w = pipeline_defaults(cls_name)
    steps = int(steps or d_steps or model.spec.steps)
    if published_w is None:
        w, note = float("nan"), f"none given (diffusers' default guidance_scale is {d_w})"
    else:
        w, note = float(published_w), "given on the command line"
    model.with_spec(key=key_of(ckpt) + ("" if steps == model.spec.steps else f"-s{steps}"),
                    name=str(ckpt), checkpoint=str(ckpt), steps=steps, calib_levels=steps, published_w=w,
                    published_note=note, sampler=f"{model.spec.sampler} (the {cls_name} default)")
    if size:
        model.size = int(size)
    return model


# --- EDM / EDM2 pickles: two networks ---------------------------------------------------------------------------
def local_pickle(p):
    """A pickle given as a path (absolute, or relative to $VMG_WEIGHTS), a URL (downloaded once into
    $VMG_WEIGHTS/url/), or an EDM2 post-hoc reconstruction by name (fetched from NVIDIA's CDN)."""
    from .models import _edm
    s = str(p)
    if s.startswith(("http://", "https://")):
        import os
        import urllib.request
        from . import runs
        dst = Path(os.environ.get("VMG_WEIGHTS", runs.root() / "_weights")) / "url" / s.rsplit("/", 1)[-1].split("?")[0]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            urllib.request.urlretrieve(s, tmp)
            tmp.replace(dst)
        return str(dst)
    if Path(s).expanduser().exists():
        return str(Path(s).expanduser().resolve())
    if re.fullmatch(r"edm2-img\d+-[\w-]+-\d+-\d\.\d+", s):
        rel = _edm.posthoc(s)
        _edm.fetch(rel)
        return rel
    return s                                                   # relative to $VMG_WEIGHTS (vmg.models._edm.weight)


def edm_pair(net, guide, kind, dataset, steps=None, published_w=None):
    from .models import edm2
    assert dataset, "an EDM pair needs --dataset (imagenet64, imagenet512, cifar10 or folder:<dir>)"
    n, g = local_pickle(net), local_pickle(guide)
    cls = edm2.generic_pair(n, g, dataset, kind, key=key_of(net, guide), steps=steps or 32,
                            published_w=float("nan") if published_w is None else float(published_w))
    return cls()
