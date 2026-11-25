"""MDM (Tevet et al., ICLR 2023), the KIT-ML text-to-motion checkpoint kit_trans_enc_512, through its own code.

MDM predicts x0 directly and guides it affinely, out_u + s (out_c - out_u) (utils/sampler_util.py,
ClassifierFreeSampleModel), so the published s = 2.5 is the paper's w; D0 drops the text (y['uncond'], which
zeroes the CLIP embedding). The sampler is ancestral DDPM over all T = 1000 cosine-schedule timesteps
(p_sample_loop, t = 999 ... 0); x_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps by MDM's own q_sample, EDM sigma
sqrt((1 - a) / a). The moments are taken at 50 of the 1000 levels, evenly spaced in sampler order with both ends.

A motion is a [251, 1, 196] frame-feature array, Z-normalised and zero-padded past its length; flat() makes
each real frame one sample of 251 features, so padded frames never enter the moments.

Evaluation is MDM's own (eval/eval_humanml.py, wo_mm): one replication per evaluation row, seed 10 + row,
scored by MDM's T2M evaluator. Its limit is 1,000 generated motions, but KIT-ML's test loader holds 22 full
batches of 32, so a replication generates and scores 704. Generation and scoring are one loop in MDM, so
sample() writes each replication's metrics (rep<row>.json). MultiModality is MDM's mm_short instead: on the
dataset "kitml-mm", row r is one replication at seed 10 + r that also generates 100 prompts 30 times each, and
sample() writes mm<row>.json.
"""
import json
import os
import re
import sys
import types
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from .. import guidance as G
from ..data import kitml
from .base import Model, Spec, Split


class Weighted(nn.Module):
    """ClassifierFreeSampleModel with the weight read from w_by_t[timestep] instead of y['scale']: the same two
    network calls and the same arithmetic, so a constant 2.5 reproduces it bit for bit (checked in smoke())."""

    def __init__(self, model, w_by_t):
        super().__init__()
        self.model = model
        assert self.model.cond_mask_prob > 0
        self.register_buffer("w_by_t", torch.as_tensor(np.asarray(w_by_t), dtype=torch.float32))
        for k in ("rot2xyz", "translation", "njoints", "nfeats", "data_rep", "cond_mode", "encode_text"):
            setattr(self, k, getattr(model, k))

    def forward(self, x, timesteps, y=None):
        y_uncond = deepcopy(y)
        y_uncond["uncond"] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (self.w_by_t[timesteps].view(-1, 1, 1, 1) * (out - out_uncond))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["model"], name)


def _clip_cache():
    """Point clip.load at $VMG_CLIP_CACHE (a pre-fetched ViT-B/32) when set; MDM calls clip.load(name, ...)."""
    root = os.environ.get("VMG_CLIP_CACHE")
    if not root:
        return
    import clip
    if getattr(clip.load, "_vmg", False):
        return
    orig = clip.load

    def load(name, *a, **kw):
        kw.setdefault("download_root", root)
        return orig(name, *a, **kw)

    load._vmg = True
    clip.load = load


def _wandb_stub():
    """MDM's eval imports train/train_platforms.py, whose WandBPlatform class body runs `import wandb`; the eval never
    logs to it. An inert module stands in when wandb is not installed."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        sys.modules["wandb"] = types.ModuleType("wandb")


class MDM(Model):
    spec = Spec(key="mdm", name="MDM", modality="motion", checkpoint="kit_trans_enc_512", reference="text dropped",
                sampler="DDPM", steps=1000, output="KIT-ML pose sequence", published_w=2.5,
                datasets={"kitml": Split(calib="KIT-ML validation, 288 motions x 3 noise draws (864)", n_calib=864,
                                         eval="KIT-ML test, 20 replications of 1,000 motions", n_eval=20,
                                         reference="the test set", seed="seeds 10-29"),
                          "kitml-mm": Split(calib="as kitml", n_calib=864,
                                            eval="KIT-ML test, MultiModality: 5 replications of 100 prompts x 30",
                                            n_eval=5, reference="none", seed="seeds 10-14")},
                calib_levels=50)
    batch = 64

    def __init__(self):
        self.model = None

    def load(self):
        """(model, diffusion) exactly as MDM's eval builds them from the checkpoint's own args.json: every option at
        its default, overwritten by the saved value (the 2022 checkpoint predates options the code now reads)."""
        if self.model is not None:
            return
        from argparse import ArgumentParser, Namespace
        _clip_cache()
        path = str(kitml.root() / kitml.CKPT)
        with kitml.mdm_env():
            from utils import parser_util as P
            from utils.model_util import create_model_and_diffusion, load_saved_model
            parser = ArgumentParser()
            for add in (P.add_base_options, P.add_data_options, P.add_model_options, P.add_diffusion_options):
                add(parser)
            args = parser.parse_args([])
            saved = json.load(open(os.path.join(os.path.dirname(path), "args.json")))
            for k in list(vars(args)):
                if k in saved:
                    setattr(args, k, saved[k])
                elif k == "unconstrained" and "cond_mode" in saved:
                    args.unconstrained = saved["cond_mode"] == "no_cond"
            args.use_ema = saved.get("use_ema", False)
            args.model_path = path
            args = P.apply_rules(args)
            model, diffusion = create_model_and_diffusion(args, Namespace(dataset=Namespace(num_actions=1)))
            load_saved_model(model, path, use_avg=args.use_ema)
        assert args.dataset == "kit" and diffusion.num_timesteps == self.spec.steps, (args.dataset, diffusion.num_timesteps)
        model.to("cuda")                 # MDM.to moves its SMPL layer too, and returns None
        model.eval()
        self.model, self.diffusion, self.args = model, diffusion, args

    def levels(self):
        self.load()
        a = np.asarray(self.diffusion.alphas_cumprod, dtype=np.float64)
        t = np.arange(len(a))[::-1].copy()
        return dict(sigma=np.sqrt((1 - a[t]) / a[t]), t=t.astype(np.float64))

    def calib_levels(self):
        return [int(i) for i in np.unique(np.round(np.linspace(0, self.spec.steps - 1, self.spec.calib_levels)))]

    def rows(self, dataset="kitml", split="calib"):
        assert dataset in ("kitml", "kitml-mm"), dataset
        return kitml.rows(split, mm=dataset == "kitml-mm")

    def _y(self, rows):
        """MDM's own collate of the calibration rows, so mask / lengths / text are exactly the trainer's."""
        from data_loaders.tensors import collate
        x0, lengths, caps, toks = kitml.calib_set()
        b = collate([{"inp": torch.from_numpy(x0[r]), "text": caps[r], "tokens": toks[r], "lengths": int(lengths[r])}
                     for r in rows])
        y = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b[1]["y"].items()}
        return b[0].cuda(), y

    @torch.no_grad()
    def data(self, dataset, rows):
        self.load()
        with kitml.mdm_env():
            x0, y = self._y(rows)
        return x0.float(), y

    def flat(self, x, cond):
        X = x[:, :, 0, :].transpose(1, 2)                            # [B, 196 frames, 251 features]
        L = cond["lengths"].to(x.device)
        return X, torch.arange(X.shape[1], device=x.device)[None] < L[:, None]

    def _t(self, k, B, device):
        return torch.full((B,), int(self.levels()["t"][k]), device=device, dtype=torch.long)

    def noise(self, x0, k, eps):
        return self.diffusion.q_sample(x0, self._t(k, x0.shape[0], x0.device), noise=eps.to(x0.device))

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        tt = self._t(k, x.shape[0], x.device)
        yu = deepcopy(cond)
        yu["uncond"] = True
        return self.model(x, tt, cond).float(), self.model(x, tt, yu).float()

    @torch.no_grad()
    def smoke(self):
        from utils.sampler_util import ClassifierFreeSampleModel
        self.load()
        w = self.spec.published_w
        x0, y = self.data("kitml", [0, 1])
        B = x0.shape[0]
        lv = self.levels()
        keep = {0, 1, 1000, 1001, 1998, 1999}                       # (cond, uncond) calls of steps 0, 500, 999
        ts, cap = [], {}

        def hook(mod, args, kw, out):
            i = len(ts)
            ts.append(int(args[1][0]))
            if i in keep:
                cap[i] = (args[0].clone(), out.clone())

        yy = deepcopy(y)
        yy["scale"] = torch.ones(B, device="cuda") * w
        h = self.model.register_forward_hook(hook, with_kwargs=True)
        try:
            torch.manual_seed(0)
            ref = self.diffusion.p_sample_loop(ClassifierFreeSampleModel(self.model), x0.shape, clip_denoised=False,
                                               model_kwargs={"y": yy}, progress=False)
        finally:
            h.remove()
        assert ts == list(np.repeat(lv["t"].astype(int), 2)), "p_sample_loop does not visit levels() in order"
        for k in (0, 500, 999):
            (xc, oc), (xu, ou) = cap[2 * k], cap[2 * k + 1]
            assert torch.equal(xc, xu), "the two branches saw different inputs"
            Dc, Du = self.denoise_pair(xc, k, y)
            assert torch.equal(Dc, oc.float()) and torch.equal(Du, ou.float()), f"network outputs at step {k}"
        torch.manual_seed(0)
        ours = self.diffusion.p_sample_loop(Weighted(self.model, np.full(self.spec.steps, w)).to("cuda"), x0.shape,
                                            clip_denoised=False, model_kwargs={"y": yy}, progress=False)
        assert torch.equal(ours, ref), "Weighted at a constant is not ClassifierFreeSampleModel"
        print(f"{self.spec.name} smoke: 2000 network calls at levels() in order; both branches identical at steps 0, "
              f"500, 999; Weighted at constant {w} bitwise ClassifierFreeSampleModel over the full 1000-step sampler")

    def sample(self, dataset, rows, w, outdir):
        """One replication of MDM's evaluation per row (seed 10 + row): wo_mm -> rep<row>.json, or on "kitml-mm"
        mm_short -> mm<row>.json, each with its metrics.
        Every arm runs the same seeds, so arms share the eval loader's captions, lengths and noise."""
        self.load()
        ws = None if w is None else G.weights(w, self.levels())
        mm = dataset == "kitml-mm"
        for r in rows:
            out = outdir / f"{'mm' if mm else 'rep'}{r:02d}.json"
            if not out.exists():
                self._replication(kitml.EVAL_SEED0 + int(r), ws, out, mm)

    def _replication(self, seed, ws, out, mm=False):
        """One replication of eval_humanml.py: wo_mm, or mm_short (mm=True: also 100 prompts x 30 generations,
        MultiModality over 10 draws)."""
        n_mm, n_rep, n_times = (100, 30, 10) if mm else (0, 0, 0)
        _wandb_stub()
        _clip_cache()
        argv = sys.argv
        with kitml.mdm_env():
            sys.argv = ["eval_humanml", "--model_path", str(kitml.root() / kitml.CKPT), "--eval_mode", "mm_short" if mm else "wo_mm",
                        "--guidance_param", str(self.spec.published_w), "--seed", str(seed)]
            try:
                import eval.eval_humanml as EH
                from data_loaders.get_data import get_dataset_loader
                from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
                from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
                from diffusion import logger
                from utils import dist_util
                from utils.fixseed import fixseed
                from utils.parser_util import evaluation_parser
                from utils.sampler_util import ClassifierFreeSampleModel
                args = evaluation_parser()
            finally:
                sys.argv = argv
            fixseed(args.seed)
            args.batch_size = 32                                  # R-precision assumes 32
            dist_util.setup_dist(args.device)
            logger.configure()
            gt = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split="test",
                                    hml_mode="gt")
            gen = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split="test",
                                     hml_mode="eval", fixed_len=args.context_len + args.pred_len, pred_len=args.pred_len,
                                     device=dist_util.dev(), autoregressive=args.autoregressive)
            # the same network as self.model, built by the eval's own path so its RNG use is MDM's
            from utils.model_util import create_model_and_diffusion, load_saved_model
            model, diffusion = create_model_and_diffusion(args, gen)
            load_saved_model(model, args.model_path, use_avg=args.use_ema)
            model = ClassifierFreeSampleModel(model) if ws is None else Weighted(model, np.asarray(ws)[::-1].copy())
            model.to(dist_util.dev())
            model.eval()
            loaders = {"vald": lambda: get_mdm_loader(
                args, model=model, diffusion=diffusion, batch_size=args.batch_size, ground_truth_loader=gen,
                mm_num_samples=n_mm, mm_num_repeats=n_rep, max_motion_length=gt.dataset.opt.max_motion_length,
                num_samples_limit=1000, scale=args.guidance_param)}
            wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            log = f"{out}.log"
            EH.evaluation(wrapper, gt, loaders, log, 1, 300, n_times, run_mm=mm)
        res = dict(seed=seed, mm=mm, weights="published" if ws is None else np.asarray(ws).tolist(),
                   device=torch.cuda.get_device_name(), metrics=_parse(open(log).read()))
        tmp = f"{out}.tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, out)


def _parse(text):
    """{'ground truth' | 'vald': {metric: value, or [top 1, top 2, top 3] for R_precision}} from MDM's eval log."""
    out = {}
    text = text.split("!!! DONE !!!")[0]                       # the summary block that follows repeats the values
    for who, key, rest in re.findall(r"---> \[(ground truth|vald)\] ([A-Za-z_ ]+?): (.*)", text):
        if key == "R_precision":
            v = [float(x) for x in re.findall(r"\(top \d\): ([-\d.e+]+)", rest)]
        else:
            v = float(rest.split()[0])
        out.setdefault(who, {})[key] = v
    return out
