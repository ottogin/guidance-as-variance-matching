"""The other text-to-image pairs of Fig 2: moments only, never sampled in the paper.

SD 1.x-shaped UNets (SD 1.4, SD 1.3, CommonCanvas, SD 2.1-base, BK-SDM). Each is measured on SD 1.5's 50 PNDM levels
and in SD 1.5's latents, as the DOCCI pass behind the paper did: all five share SD 1.5's alphas_cumprod and eps-
prediction, and their VAE encoders are SD 1.5's (CommonCanvas and SD 2.1 ship SD 2's VAE, whose encoder tensors
are byte-identical to SD 1.5's; BK-SDM ships SD 1.5's own file). SD 1.3 and 1.4 are branches of SD 1.2 and use
SD 1.5's (byte-identical) text encoder; the other three bring their own. D0 is each model's empty prompt.

Cross-model pairs (cross-*). D1 and D0 are two different models' conditional predictions for the same prompt,
each through its own text encoder, on one shared x_t: guidance D0 + w (D1 - D0) moves model D1 away from model D0.

SD 3 Medium is measured on SD 3.5 Medium's 40-level grid, although its model card
samples 28 steps; both grids are the same shift-3 schedule, from s = 1 to s = 0.0089.

Flow families (Chroma, Ovis-Image, Z-Image, CogView4, Sana) on DOCCI's whole frame at 1152 x 864 (Sana also on
MJHQ-30K at 1024^2). Each adapter rebuilds the released pipeline's own transformer call from (x_s, level,
text), and `smoke` checks it against the pipeline's step-0 calls input by input and output by output. D0 is the
pipeline's own negative branch (the empty prompt). Two conventions differ from the rest: Z-Image's
guidance_scale g is the excess form pos + g (pos - neg), so the paper's w is g + 1, and its transformer predicts
minus the velocity at time 1 - s; CogView4 pads text to the longest prompt in a batch and has no attention mask,
so a caption's conditioning depends on its batch -- the calibration pass pads every caption to the longest
calibration caption, as the pass behind the paper did.
"""
import copy

import numpy as np
import torch

from .. import runs
from ..data import folder
from . import hub
from .base import Model, Spec, Split
from .sd import DOCCI, SD15, SD3Family, _stop_after_first_step, calib_rows, captions

DOCCI1024 = {"docci": Split(calib="DOCCI test photos 0-1,023, full description", n_calib=1024)}
MJHQ1024 = {"mjhq": Split(calib="MJHQ-30K images 0-1,023 at 1024^2", n_calib=1024)}
COCO_CROSS = {"coco": Split(calib="COCO val2014 photos 0-5,704", n_calib=5705)}
NAN = float("nan")


# ------------------------------------------------------------------ helpers

def _record(module):
    """Forward hook recording (args, kwargs, output) of every call, tensors and lists of tensors cloned."""
    calls = []

    def cl(v):
        if torch.is_tensor(v):
            return v.detach().clone()
        if isinstance(v, (list, tuple)):
            return type(v)(cl(u) for u in v)
        return v

    def hook(m, args, kw, out):
        o = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
        calls.append((cl(args), {k: cl(v) for k, v in kw.items() if torch.is_tensor(v) or isinstance(v, (list, tuple))},
                      cl(o)))
    return calls, module.register_forward_hook(hook, with_kwargs=True)


def _same(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(u, v) for u, v in zip(a, b))
    return a == b


def _take(c, idx):
    """Rows `idx` of a conditioning: a tensor, a tuple of batch-first tensors, or a list (one entry per row)."""
    if isinstance(c, list):
        return [c[i] for i in idx]
    if isinstance(c, tuple):
        return tuple(_take(v, idx) for v in c)
    return c[list(idx)]


def _cat(cs):
    if isinstance(cs[0], list):
        return [x for c in cs for x in c]
    if isinstance(cs[0], tuple):
        return tuple(_cat([c[j] for c in cs]) for j in range(len(cs[0])))
    return torch.cat(cs)


def _alphas(config):
    """alphas_cumprod from a scheduler config's betas (the configs ship different scheduler classes)."""
    n = int(config.get("num_train_timesteps", 1000))
    b0, b1 = float(config.get("beta_start", 1e-4)), float(config.get("beta_end", 0.02))
    kind = config.get("beta_schedule", "linear")
    b = np.linspace(b0 ** 0.5, b1 ** 0.5, n) ** 2 if kind == "scaled_linear" else np.linspace(b0, b1, n)
    assert kind in ("scaled_linear", "linear"), kind
    return np.cumprod(1.0 - b)


# ------------------------------------------------------------------ SD 1.x-shaped UNets

class SD1x(SD15):
    """A UNet measured on SD 1.5's levels and latents (see the module docstring); subclasses set `spec`."""

    own_text = True          # its own text encoder and tokenizer (False: SD 1.5's, byte-identical)
    unet_variant = None      # "fp16" for the CompVis checkpoints, which are fetched as their fp16 UNet only
    revision = None          # a Hub commit, for checkpoints whose repository is not the publisher's own

    def load(self):
        if self.pipe is not None:
            return
        from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        base, own = hub.path(SD15.spec.checkpoint), hub.path(self.spec.checkpoint)
        vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32)
        sch = PNDMScheduler.from_pretrained(base, subfolder="scheduler")
        kw = dict(vae=vae, scheduler=sch, safety_checker=None, feature_extractor=None, requires_safety_checker=False,
                  torch_dtype=self.dtype)
        rev = dict(revision=self.revision) if self.revision and own == self.spec.checkpoint else {}   # not a mirror
        if self.own_text:
            pipe = StableDiffusionPipeline.from_pretrained(own, **kw, **rev)
            own_sch = type(sch).load_config(own, subfolder="scheduler", **rev)
            assert np.array_equal(_alphas(own_sch), _alphas(sch.config)), f"{self.spec.name}: another noise schedule"
        else:
            unet = UNet2DConditionModel.from_pretrained(own, subfolder="unet", variant=self.unet_variant,
                                                        torch_dtype=self.dtype, **rev)
            pipe = StableDiffusionPipeline.from_pretrained(base, unet=unet, **kw)
        self.pipe = pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unet.eval().requires_grad_(False)
        sch = self.pipe.scheduler
        sch.set_timesteps(self.spec.steps)
        t = np.unique(sch.timesteps.cpu().numpy())[::-1].copy()
        a = sch.alphas_cumprod[torch.from_numpy(t)].double()
        self.sqrt_a, self.sqrt_1ma = a.sqrt().numpy(), (1 - a).sqrt().numpy()
        self.sig, self.ts = ((1 - a) / a).sqrt().numpy(), t
        self.null = self.encode([""])


def _sd1x_spec(key, name, checkpoint, note):
    return Spec(key=key, name=name, modality="image", checkpoint=checkpoint, reference="empty prompt",
                sampler="PNDM (SD 1.5's levels)", steps=50, output="512^2 image", published_w=7.5,
                published_note=note, datasets={"docci": DOCCI["docci"]}, calib_levels=50)


class SD14(SD1x):
    own_text, unet_variant = False, "fp16"
    spec = _sd1x_spec("sd14", "SD 1.4", "CompVis/stable-diffusion-v1-4", "the StableDiffusionPipeline default")


class SD13(SD1x):
    own_text, unet_variant = False, "fp16"
    spec = _sd1x_spec("sd13", "SD 1.3", "CompVis/stable-diffusion-v1-3", "the StableDiffusionPipeline default")


class CommonCanvas(SD1x):
    spec = _sd1x_spec("commoncanvas", "CommonCanvas", "common-canvas/CommonCanvas-S-C",
                      "the StableDiffusionPipeline default (as SD 1.5's)")


class SD21(SD1x):
    # Stability's own repository is gone; a third-party copy, pinned to the commit the paper's runs used
    revision = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
    spec = _sd1x_spec("sd21", "SD 2.1", "Manojb/stable-diffusion-2-1-base",
                      "a mirror of stabilityai/stable-diffusion-2-1-base; the StableDiffusionPipeline default")


class BKSDM(SD1x):
    spec = _sd1x_spec("bksdm", "BK-SDM", "nota-ai/bk-sdm-base", "distilled from SD 1.5; the pipeline default")


# ------------------------------------------------------------------ cross-model pairs

class _Cross(SD15):
    """D1 = model A's conditional prediction, D0 = model B's, same prompt, one x_t (SD 1.5's latents and levels).

    Every member here runs on the same pipeline apart from its UNet and text encoder: SD 1.5's PNDM scheduler at
    50 steps, SD 1.5's VAE and 480^2 on COCO -- as the cross runs behind the paper did (a member that ships no
    VAE was rendered through SD 1.5's pipeline with its own UNet and text encoder swapped in). So a pair and its
    reverse (cross-cc-sd21 for cross-sd21-cc) differ only in which model hosts the pipeline call and in w <-> 1 - w;
    the paper sampled w > 0 on A's pipeline and w <= 0 on B's. Latents and decoding go through one float32 copy
    of SD 1.5's VAE, so every pair and orientation shares x0 and the decoder.
    """

    a = b = None             # member classes
    batch = 16

    def __init__(self):
        super().__init__()
        self.A, self.B = self.a(), self.b()

    def load(self):
        if self.pipe is not None:
            return
        self.A.load()
        self.B.load()
        assert np.array_equal(self.A.sqrt_a, self.B.sqrt_a) and np.array_equal(self.A.ts, self.B.ts), "levels differ"
        from diffusers import AutoencoderKL
        self.vae32 = AutoencoderKL.from_pretrained(hub.path(SD15.spec.checkpoint), subfolder="vae",
                                                   torch_dtype=torch.float32).to("cuda").eval().requires_grad_(False)
        self.pipe = self.A.pipe              # levels and sampling are A's
        self.sqrt_a, self.sqrt_1ma, self.sig, self.ts = self.A.sqrt_a, self.A.sqrt_1ma, self.A.sig, self.A.ts

    @torch.no_grad()
    def latents(self, images, chunk=8):
        sc = self.vae32.config.scaling_factor
        return torch.cat([self.vae32.encode(x.to("cuda", torch.float32)).latent_dist.mode() * sc for x in images.split(chunk)])

    @torch.no_grad()
    def decode(self, z):
        x = self.vae32.decode(z.to(torch.float32) / self.vae32.config.scaling_factor, return_dict=False)[0]
        return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def images(self, dataset, rows):
        return self.A.images(dataset, rows)

    def data(self, dataset, rows):
        self.load()
        return self.latents(self.images(dataset, rows)), (self.A.text(rows, dataset), self.B.text(rows, dataset))

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        d = lambda m, emb: (x - float(self.sqrt_1ma[k]) * m.eps(x, k, emb).to(torch.float32)) / float(self.sqrt_a[k])
        return d(self.A, cond[0]), d(self.B, cond[1])

    def smoke(self):
        self.load()
        self.A.smoke()
        self.B.smoke()
        print(f"{self.spec.name}: both members match their pipelines; levels and alphas identical")

    @property
    def generate(self):      # smoke.py's CFG-hook check does not apply (hasattr() is False); see mixed_latents
        raise AttributeError("a cross pair has no CFG branch to hook")

    @torch.no_grad()
    def mixed_latents(self, dataset, rows, ws):
        """Final latents of D_B + w (D_A - D_B) for `rows`, at one weight per level `ws` (any real w).

        Model A's own pipeline runs with its CFG off and its UNet's output replaced by eps_B + w (eps_A - eps_B),
        B reading the same caption through its own text encoder. With one alphas_
        cumprod D is the same affine map of eps for both, so this is the guided denoiser. At w = 1 B is never
        called (A's pipeline exactly); at w = 0 the output is eps_B (B alone on A's pipeline)."""
        from .. import guidance as G
        assert dataset == "coco", "cross pairs sample on COCO (fig:cross-pairs)"
        self.load()
        _, _, weight_at = G.diffusers_hook(ws, self.levels()["t"], floor=-np.inf)
        unet_a, unet_b = self.A.pipe.unet, self.B.pipe.unet
        fwd, emb = unet_a.forward, {}

        def mixed(sample, timestep, encoder_hidden_states=None, **kw):
            out = fwd(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kw)
            w = weight_at(timestep)
            if w == 1.0:
                return out
            ea = out[0] if isinstance(out, tuple) else out.sample
            eb = unet_b(sample, timestep, encoder_hidden_states=emb["b"], return_dict=False)[0]
            return ((eb.float() + w * (ea.float() - eb.float())).to(ea.dtype),)

        unet_a.forward = mixed          # an instance attribute: nn.Module.__call__ finds it first
        try:
            caps = captions(dataset, rows)
            emb["b"] = self.B.encode(caps)
            return self.A.pipe(prompt=caps, height=self.A.size, width=self.A.size, num_inference_steps=self.spec.steps,
                               guidance_scale=1.0, latents=self.A.initial_noise(rows).to("cuda", self.dtype),
                               output_type="latent").images
        finally:
            del unet_a.forward

    def sample(self, dataset, rows, w, outdir, batch=8):
        """PNGs <row>.png of D_B + w (D_A - D_B), decoded with SD 1.5's VAE in float32; prompt i (COCO's longest
        caption) and initial noise i are shared by every arm and both orientations. `w` is a constant or one weight
        per level; there is no published one. fig:cross-pairs: --dataset coco --n 1000 (rows 5,705-6,704)."""

        from .. import guidance as G
        assert w is not None, "a cross pair has no published weight: pass --w or --weights"
        self.load()
        ws = G.weights(w, self.levels())
        for b0 in range(0, len(rows), batch):
            rb = [r for r in rows[b0:b0 + batch] if not (outdir / f"{r:05d}.png").exists()]
            if rb:
                for r, im in zip(rb, self.decode(self.mixed_latents(dataset, rb, ws))):
                    runs.save_image(im, outdir / f"{r:05d}.png")


def _cross_spec(key, a, b):
    return Spec(key=key, name=f"{a.spec.name} <- {b.spec.name}", modality="image",
                checkpoint=f"{a.spec.checkpoint} + {b.spec.checkpoint}", reference=f"{b.spec.name}, same prompt",
                sampler="PNDM (SD 1.5's levels)", steps=50, output="512^2 image (DOCCI), 480^2 (COCO)",
                published_w=NAN, published_note="no published weight: cross-model guidance",
                datasets={"docci": DOCCI["docci"], **COCO_CROSS}, calib_levels=50)


class CrossSD15CC(_Cross):
    a, b = SD15, CommonCanvas
    spec = _cross_spec("cross-sd15-cc", SD15, CommonCanvas)


class CrossSD15SD21(_Cross):
    a, b = SD15, SD21
    spec = _cross_spec("cross-sd15-sd21", SD15, SD21)


class CrossSD21CC(_Cross):
    a, b = SD21, CommonCanvas
    spec = _cross_spec("cross-sd21-cc", SD21, CommonCanvas)


# the reverse orientations: fig:cross-pairs samples its w <= 0 points on B's pipeline, at 1 - w on the reverse pair
class CrossCCSD21(_Cross):
    a, b = CommonCanvas, SD21
    spec = _cross_spec("cross-cc-sd21", CommonCanvas, SD21)


class CrossSD21SD15(_Cross):
    a, b = SD21, SD15
    spec = _cross_spec("cross-sd21-sd15", SD21, SD15)


# ------------------------------------------------------------------ SD 3 Medium

class SD3Medium(SD3Family):
    grid_steps = 40
    spec = Spec(key="sd3m", name="SD 3 Medium", modality="image",
                checkpoint="stabilityai/stable-diffusion-3-medium-diffusers", reference="empty prompt",
                sampler="flow-matching Euler", steps=28, output="1152x864 image", published_w=7.0,
                published_note="model card: 7.0 at 28 steps; calibrated on SD 3.5 Medium's 40-level grid",
                datasets={"docci": DOCCI["docci"]}, calib_levels=40)


# ------------------------------------------------------------------ flow families

class _Flow(Model):
    """Rectified flow x_s = (1 - s) x0 + s eps, D = x_s - s v; subclasses rebuild the pipeline's network call."""

    pipeline = vae_class = ""
    dtype = torch.bfloat16
    batch = 4
    row_align = 16           # text is encoded in blocks of 16 rows aligned to row 0
    frame = (1152, 864)      # (width, height) on DOCCI
    pipe_kw = {}
    seed = 0
    smoke_tol = 3e-2         # relative L2 between our separate branch calls and the pipeline's batched ones

    def __init__(self):
        self.pipe = None
        self._block = (None, None)

    # --- loading and levels ------------------------------------------------------------------------------
    def load(self):
        if self.pipe is not None:
            return
        import diffusers
        path = hub.path(self.spec.checkpoint)
        vae = getattr(diffusers, self.vae_class).from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
        self.P = getattr(diffusers, self.pipeline)
        self.pipe = self.P.from_pretrained(path, vae=vae, torch_dtype=self.dtype).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.net = self.pipe.transformer.eval().requires_grad_(False)
        self.s, self.ts = self.grid(*self.render(self.spec.default_dataset))
        self.null = self.encode([""])

    def render(self, dataset):
        if folder.is_folder(dataset):
            return (self.size, self.size) if getattr(self, "size", None) else self.frame
        return (1024, 1024) if dataset == "mjhq" else self.frame

    def grid(self, W, H):
        """(s [S] float64, t [S] float32): the sampler's levels at a W x H render, exactly as the pipeline sets them."""
        raise NotImplementedError

    def levels(self):
        self.load()
        with np.errstate(divide="ignore"):
            return dict(sigma=np.where(self.s < 1, self.s / (1 - self.s), np.inf), t=self.ts.astype(np.float64))

    def pipeline_w(self, w):
        return w

    # --- data ----------------------------------------------------------------------------------------------
    def rows(self, dataset, split="calib"):
        return calib_rows(self.spec, dataset, split)

    def captions(self, dataset, ids):
        return captions(dataset, ids)

    def images(self, dataset, rows):
        if folder.is_folder(dataset):
            return folder.images(dataset, rows, self.render(dataset))
        if dataset == "mjhq":
            from ..data import mjhq
            return mjhq.images(rows, 1024)
        from ..data import docci
        assert dataset in ("docci", "docci-short"), dataset
        return docci.images(rows, "frame")

    @torch.no_grad()
    def latents(self, images, chunk=4):
        vae = self.pipe.vae
        shift, scale = vae.config.get("shift_factor") or 0.0, vae.config.scaling_factor
        return torch.cat([(vae.encode(x.to("cuda", torch.float32)).latent_dist.mode() - shift) * scale
                          for x in images.split(chunk)])

    def encode_block(self, dataset, prompts):
        return self.encode(prompts)

    def text(self, rows, dataset):
        out = []
        for r in rows:
            b0 = r - r % self.row_align
            if self._block[0] != (dataset, b0):
                ids = range(b0, b0 + self.row_align)
                self._block = ((dataset, b0), self.encode_block(dataset, self.captions(dataset, ids)))
            out.append(_take(self._block[1], [r - b0]))
        return _cat(out)

    def data(self, dataset, rows):
        self.load()
        s, t = self.grid(*self.render(dataset))
        assert np.array_equal(s, self.s) and np.array_equal(t, self.ts), f"{dataset}: the level grid depends on the size"
        return self.latents(self.images(dataset, rows)), self.text(rows, dataset)

    # --- the two predictions -------------------------------------------------------------------------------
    def noise(self, x0, k, eps):
        s = torch.tensor(float(self.s[k]), device=x0.device)
        return (1 - s) * x0 + s * eps.to(x0.device)

    def call(self, x, k, cond):
        """(args, kwargs) of the pipeline's transformer call on latents x [B, C, h, w] at level k."""
        raise NotImplementedError

    def velocity_of(self, out, x):
        """The velocity [B, C, h, w] float32 from the network's output."""
        return out.float()

    def x_of(self, args, kw):
        """The latents [B, C, h, w] of a recorded pipeline call."""
        raise NotImplementedError

    def text_rows(self, cond):
        """One text tensor per row of a conditioning, to tell a recorded call's branches apart."""
        return list((cond[0] if isinstance(cond, tuple) else cond).unbind(0)) if not isinstance(cond, list) else cond

    def text_of(self, args, kw):
        raise NotImplementedError

    def velocity(self, x, k, cond):
        a, kw = self.call(x, k, cond)
        out = self.net(*a, **kw)
        return self.velocity_of(out[0] if isinstance(out, tuple) else out, x)

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        s = torch.tensor(float(self.s[k]), device=x.device)
        null = _take(self.null, [0] * len(x))
        return x - s * self.velocity(x, k, cond), x - s * self.velocity(x, k, null)

    # --- the check -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def smoke(self):
        """Run the pipeline's first step; its level grid is ours, every transformer call it makes is the call we
        build from (x, level, text) argument by argument, and the network reproduces its output."""
        self.load()
        ds = self.spec.default_dataset
        W, H = self.render(ds)
        cap = self.captions(ds, [0])
        calls, h = _record(self.net)
        try:
            self.pipe(prompt=list(cap), height=H, width=W, num_inference_steps=self.spec.steps,   # Z-Image edits it
                      guidance_scale=self.pipeline_w(self.spec.published_w), generator=torch.Generator("cpu").manual_seed(0),
                      output_type="latent", callback_on_step_end=_stop_after_first_step, **self.pipe_kw)
        finally:
            h.remove()
        sch = self.pipe.scheduler
        n = self.spec.steps
        assert np.array_equal(torch.as_tensor(sch.timesteps)[:n].float().cpu().numpy(), self.ts), "timestep grid"
        assert np.array_equal(torch.as_tensor(sch.sigmas)[:n].double().cpu().numpy(), self.s), "sigma grid"
        c = self.encode(cap)
        ours_rows, null_rows = self.text_rows(c), self.text_rows(self.null)
        kinds, vs, xs = [], {}, None
        assert calls, "no transformer call captured"
        for args, kw, out in calls:
            x = self.x_of(args, kw)
            rows = self.text_of(args, kw)
            kind = ["cond" if _same(r, ours_rows[0]) else "null" if _same(r, null_rows[0]) else "?" for r in rows]
            kinds += kind
            assert "?" not in kind, f"a branch's text is neither the caption's nor the empty prompt's: {kind}"
            cb = _cat([c if k == "cond" else self.null for k in kind])
            a, kw2 = self.call(x, 0, cb)
            assert len(a) == len(args) and all(_same(u, v) for u, v in zip(a, args)), "positional arguments differ"
            for key, v in kw2.items():
                if torch.is_tensor(v) or isinstance(v, (list, tuple)):
                    assert key in kw and _same(v, kw[key]), f"argument {key!r} differs from the pipeline's"
            again = self.net(*a, **kw2)
            assert _same(again[0] if isinstance(again, tuple) else again, out), "network output on the pipeline's inputs"
            v = self.velocity_of(out, x)
            for j, k in enumerate(kind):
                vs[k] = v[j:j + 1]
            xs = x[:1].float()
        assert sorted(kinds) == ["cond", "null"], f"branches seen at step 0: {kinds}"
        s, w = float(self.s[0]), self.spec.published_w
        p1, p0 = xs - s * vs["cond"], xs - s * vs["null"]           # the pipeline's own two predictions
        ref = p0 + w * (p1 - p0)
        d1, d0 = self.denoise_pair(xs, 0, c)
        err = lambda a, b: float((a - b).float().norm() / b.float().norm())      # noqa: E731  relative L2
        e = max(err(d1, p1), err(d0, p0))
        print(f"{self.spec.name} smoke: {len(calls)} call(s) at step 0, branches {kinds}; grid, every argument and the "
              f"network output identical; our separate calls within {e:.1e} (relative L2) of the pipeline's batched branches "
              f"(guided at w = {w:g}: {err(d0 + w * (d1 - d0), ref):.1e})")
        # bf16 batching noise only (a max-element ratio reached 8e-2 on some H200s); the guided prediction multiplies
        # it by about 2w - 1, so test each branch
        assert e < self.smoke_tol, "separate branch calls disagree with the pipeline beyond bf16 batching noise"

    def sample(self, *a, **kw):
        raise NotImplementedError(f"{self.spec.name} is a moments-only pair of Fig 2")


def _flow_spec(key, name, checkpoint, sampler, steps, w, datasets, note=""):
    return Spec(key=key, name=name, modality="image", checkpoint=checkpoint, reference="empty prompt",
                sampler=sampler, steps=steps, output="1152x864 image", published_w=w, published_note=note,
                datasets=datasets, calib_levels=steps)


class _Packed(_Flow):
    """FLUX-style transformers on 2 x 2 packed latent patches (Chroma, Ovis-Image)."""

    vae_class = "AutoencoderKL"

    def _flux_grid(self, W, H, module, flow_sigmas_off=False):
        import importlib
        m = importlib.import_module(module)
        sch = copy.deepcopy(self.pipe.scheduler)
        n = self.spec.steps
        sigmas = np.linspace(1.0, 1 / n, n)
        if flow_sigmas_off and getattr(sch.config, "use_flow_sigmas", False):
            sigmas = None
        seq = (H // 16) * (W // 16)                       # packed tokens: VAE 8x, then 2 x 2 patches
        mu = m.calculate_shift(seq, sch.config.get("base_image_seq_len", 256), sch.config.get("max_image_seq_len", 4096),
                               sch.config.get("base_shift", 0.5), sch.config.get("max_shift", 1.15))
        m.retrieve_timesteps(sch, n, "cpu", sigmas=sigmas, mu=mu)
        return torch.as_tensor(sch.sigmas)[:n].double().numpy(), torch.as_tensor(sch.timesteps)[:n].float().numpy()

    def _packed(self, x):
        b, c, h, w = x.shape
        xp = self.P._pack_latents(x.to(self.dtype), b, c, h, w)
        ids = self.P._prepare_latent_image_ids(b, h // 2, w // 2, x.device, self.dtype)
        return xp, ids

    def _t(self, k, b, device):
        # the pipeline casts the timestep to the latents' dtype before dividing by 1000
        return torch.full((b,), float(self.ts[k]), device=device, dtype=torch.float32).to(self.dtype) / 1000

    def x_of(self, args, kw):
        hs = kw["hidden_states"]
        W, H = self.render(self.spec.default_dataset)
        return self.P._unpack_latents(hs, H, W, self.pipe.vae_scale_factor)

    def velocity_of(self, out, x):
        h, w = x.shape[-2:]
        return self.P._unpack_latents(out, h * self.pipe.vae_scale_factor, w * self.pipe.vae_scale_factor,
                                      self.pipe.vae_scale_factor).float()

    def text_of(self, args, kw):
        return list(kw["encoder_hidden_states"].unbind(0))


class Chroma(_Packed):
    """Chroma1-HD: FLUX's architecture de-distilled back to affine CFG; one T5-XXL whose mask is conditioning."""

    pipeline = "ChromaPipeline"
    pipe_kw = dict(max_sequence_length=512)
    spec = _flow_spec("chroma", "Chroma1-HD", "lodestones/Chroma1-HD", "flow-matching Euler", 35, 5.0, DOCCI1024)

    def grid(self, W, H):
        return self._flux_grid(W, H, "diffusers.pipelines.chroma.pipeline_chroma")

    @torch.no_grad()
    def encode(self, prompts):
        return self.pipe._get_t5_prompt_embeds(prompt=list(prompts), max_sequence_length=512, device="cuda")

    def call(self, x, k, cond):
        emb, mask = cond
        xp, ids = self._packed(x)
        txt = torch.zeros(emb.shape[1], 3).to(device=x.device, dtype=self.dtype)
        att = self.pipe._prepare_attention_mask(batch_size=len(x), sequence_length=xp.shape[1], dtype=self.dtype,
                                                attention_mask=mask)
        return (), dict(hidden_states=xp, timestep=self._t(k, len(x), x.device), encoder_hidden_states=emb, txt_ids=txt,
                        img_ids=ids, attention_mask=att, return_dict=False)


class Ovis(_Packed):
    """Ovis-Image-7B: FLUX packing behind a chat-templated LLM encoder; text tokens get rotary positions."""

    pipeline = "OvisImagePipeline"
    pipe_kw = dict(max_sequence_length=256)
    spec = _flow_spec("ovis", "Ovis-Image 7B", "AIDC-AI/Ovis-Image-7B", "flow-matching Euler", 50, 5.0, DOCCI1024)

    def grid(self, W, H):
        return self._flux_grid(W, H, "diffusers.pipelines.ovis_image.pipeline_ovis_image", flow_sigmas_off=True)

    @torch.no_grad()
    def encode(self, prompts):
        return self.pipe._get_ovis_prompt_embeds(prompt=list(prompts), max_sequence_length=256, device="cuda")

    def call(self, x, k, emb):
        xp, ids = self._packed(x)
        txt = torch.zeros(emb.shape[1], 3)
        txt[..., 1] = txt[..., 1] + torch.arange(emb.shape[1])[None, :]
        txt[..., 2] = txt[..., 2] + torch.arange(emb.shape[1])[None, :]
        txt = txt.to(device=x.device, dtype=self.dtype)
        return (), dict(hidden_states=xp, timestep=self._t(k, len(x), x.device), encoder_hidden_states=emb, txt_ids=txt,
                        img_ids=ids, return_dict=False)


class ZImage(_Flow):
    """Z-Image: a Qwen3 encoder with ragged per-prompt text, guidance in the excess form, and -v at time 1 - s."""

    pipeline, vae_class = "ZImagePipeline", "AutoencoderKL"
    pipe_kw = dict(max_sequence_length=512)
    # the pipeline's two-prompt batch has ragged text, so it runs the masked attention kernel and one prompt the
    # unmasked one; in bf16 they drift apart by about 3% (relative L2) over the 30 layers
    smoke_tol = 6e-2
    spec = _flow_spec("zimage", "Z-Image", "Tongyi-MAI/Z-Image", "flow-matching Euler", 50, 6.0, DOCCI1024,
                      note="the pipeline's guidance_scale 5.0 is the excess coefficient g of pos + g (pos - neg): w = g + 1")

    def pipeline_w(self, w):
        return w - 1.0

    def grid(self, W, H):
        from diffusers.pipelines.z_image import pipeline_z_image as m
        sch = copy.deepcopy(self.pipe.scheduler)
        n, sf = self.spec.steps, self.pipe.vae_scale_factor
        h, w = 2 * (H // (sf * 2)), 2 * (W // (sf * 2))
        mu = m.calculate_shift((h // 2) * (w // 2), sch.config.get("base_image_seq_len", 256),
                               sch.config.get("max_image_seq_len", 4096), sch.config.get("base_shift", 0.5),
                               sch.config.get("max_shift", 1.15))
        m.retrieve_timesteps(sch, n, "cpu", sigmas=m.get_default_z_image_sigmas(n), mu=mu)
        return torch.as_tensor(sch.sigmas)[:n].double().numpy(), torch.as_tensor(sch.timesteps)[:n].float().numpy()

    @torch.no_grad()
    def encode(self, prompts):
        return self.pipe._encode_prompt(prompt=list(prompts), device="cuda", max_sequence_length=512)   # it edits the list

    def call(self, x, k, caps):
        t = torch.full((len(x),), float(self.ts[k]), device=x.device, dtype=torch.float32)
        return (list(x.to(self.dtype).unsqueeze(2).unbind(dim=0)), (1000 - t) / 1000, list(caps)), dict(return_dict=False)

    def velocity_of(self, out, x):
        return -torch.stack([o.float() for o in out], dim=0).squeeze(2)

    def x_of(self, args, kw):
        return torch.stack(list(args[0])).squeeze(2)

    def text_of(self, args, kw):
        return list(args[2])


class CogView4(_Flow):
    """CogView4-6B: GLM-4 text without an attention mask, SDXL-style size conditioning, a size-dependent grid."""

    pipeline, vae_class = "CogView4Pipeline", "AutoencoderKL"
    pipe_kw = dict(max_sequence_length=1024)
    max_seq = 1024
    spec = _flow_spec("cogview4", "CogView4-6B", "zai-org/CogView4-6B", "flow-matching Euler", 50, 5.0, DOCCI1024)

    def __init__(self):
        super().__init__()
        self._pad = {}

    def grid(self, W, H):
        from diffusers.pipelines.cogview4 import pipeline_cogview4 as m
        sch = copy.deepcopy(self.pipe.scheduler)
        n, sf = self.spec.steps, self.pipe.vae_scale_factor
        seq = ((H // sf) * (W // sf)) // (self.net.config.patch_size ** 2)
        t = np.linspace(sch.config.num_train_timesteps, 1.0, n).astype(np.int64).astype(np.float32)
        sig = t / sch.config.num_train_timesteps
        mu = m.calculate_shift(seq, sch.config.get("base_image_seq_len", 256), sch.config.get("base_shift", 0.25),
                               sch.config.get("max_shift", 0.75))
        m.retrieve_timesteps(sch, n, "cpu", t, sig, mu=mu)
        return torch.as_tensor(sch.sigmas)[:n].double().numpy(), torch.as_tensor(sch.timesteps)[:n].float().numpy()

    @torch.no_grad()
    def encode(self, prompts):
        """The pipeline's own encoding: padded to the longest prompt given, then left-padded to a multiple of 16."""
        return self.pipe._get_glm_embeds(prompt=list(prompts), max_sequence_length=self.max_seq, device="cuda")

    @torch.no_grad()
    def encode_block(self, dataset, prompts):
        """Every calibration caption padded to the longest one (left, the tokenizer's pad token), as one batch
        holding the whole set would be: the conditioning then does not depend on the shard or the batch."""
        tok = self.pipe.tokenizer
        if dataset not in self._pad:
            ids = tok(self.captions(dataset, self.rows(dataset, "calib")), padding="longest", max_length=self.max_seq,
                      truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
            self._pad[dataset] = ids.shape[1]
        L = self._pad[dataset]
        ids = tok(list(prompts), padding="max_length", max_length=L, truncation=True, add_special_tokens=True,
                  return_tensors="pt").input_ids
        pad = (16 - ids.shape[1] % 16) % 16
        if pad:
            ids = torch.cat([torch.full((len(ids), pad), tok.pad_token_id, dtype=ids.dtype), ids], dim=1)
        te = self.pipe.text_encoder
        return te(ids.to("cuda"), output_hidden_states=True).hidden_states[-2].to(te.dtype)

    def call(self, x, k, emb):
        W, H = x.shape[-1] * self.pipe.vae_scale_factor, x.shape[-2] * self.pipe.vae_scale_factor
        b = len(x)
        size = torch.tensor([(H, W)], dtype=emb.dtype, device=x.device).repeat(b, 1)
        crop = torch.tensor([(0, 0)], dtype=emb.dtype, device=x.device).repeat(b, 1)
        t = torch.full((b,), float(self.ts[k]), device=x.device, dtype=torch.float32)
        return (), dict(hidden_states=x.to(self.dtype), encoder_hidden_states=emb, timestep=t, original_size=size,
                        target_size=size.clone(), crop_coords=crop, return_dict=False)

    def x_of(self, args, kw):
        return kw["hidden_states"]

    def text_of(self, args, kw):
        return list(kw["encoder_hidden_states"].unbind(0))


class Sana(_Flow):
    """Sana 1.6B: DC-AE latents (32x, 32 channels), Gemma-2 text with an instruction template, DPM-Solver++ on flow."""

    pipeline, vae_class = "SanaPipeline", "AutoencoderDC"
    pipe_kw = dict(max_sequence_length=300)
    spec = _flow_spec("sana", "Sana 1.6B", "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
                      "DPM-Solver++ (flow)", 20, 4.5, {**DOCCI1024, **MJHQ1024})

    def grid(self, W, H):
        from diffusers.pipelines.sana import pipeline_sana as m
        sch = copy.deepcopy(self.pipe.scheduler)
        n = self.spec.steps
        m.retrieve_timesteps(sch, n, "cpu", None, None)
        return torch.as_tensor(sch.sigmas)[:n].double().numpy(), torch.as_tensor(sch.timesteps)[:n].float().numpy()

    @torch.no_grad()
    def latents(self, images, chunk=4):
        vae = self.pipe.vae               # deterministic: .latent, and no shift
        return torch.cat([vae.encode(x.to("cuda", torch.float32)).latent * vae.config.scaling_factor
                          for x in images.split(chunk)])

    @torch.no_grad()
    def encode(self, prompts):
        """Captions get the pipeline's default instruction template; the empty prompt is encoded the way the pipeline
        encodes its negative branch (no template, same length)."""
        import inspect
        dt = self.pipe.text_encoder.dtype
        if list(prompts) == [""]:
            e, m = self.pipe._get_gemma_prompt_embeds(prompt=[""], device="cuda", dtype=dt, clean_caption=False,
                                                      max_sequence_length=300, complex_human_instruction=False)
            return e.to(dtype=dt), m
        chi = inspect.signature(self.P.__call__).parameters["complex_human_instruction"].default
        e, m, _, _ = self.pipe.encode_prompt(prompt=list(prompts), do_classifier_free_guidance=False, device="cuda",
                                             max_sequence_length=300, complex_human_instruction=chi)
        return e, m

    def call(self, x, k, cond):
        emb, mask = cond
        t = torch.full((len(x),), float(self.ts[k]), device=x.device, dtype=torch.float32)
        return (x.to(self.dtype),), dict(encoder_hidden_states=emb.to(self.dtype), encoder_attention_mask=mask,
                                         timestep=t * self.net.config.timestep_scale, return_dict=False)

    def velocity_of(self, out, x):
        v = out.float()
        return v.chunk(2, dim=1)[0] if self.net.config.out_channels // 2 == x.shape[1] else v

    def x_of(self, args, kw):
        return args[0]

    def text_of(self, args, kw):
        return list(kw["encoder_hidden_states"].unbind(0))
