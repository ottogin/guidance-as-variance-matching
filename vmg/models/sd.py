"""Text-to-image models with diffusers pipelines: SD 3.5 Medium / Large, SD 1.5 and FLUX.2 [klein].

Besides COCO (the paper's main dataset), SD 1.5 and SD 3.5 Medium also calibrate on DOCCI and SD 3.5 Medium on
MJHQ-30K: the extra pairs of Fig 2 (vmg/data/docci.py; the other Fig 2 checkpoints are in t2i.py).

SD 3.5 (rectified flow, MMDiT). x_s = (1 - s) x0 + s eps with s the flow time of the released
FlowMatchEulerDiscreteScheduler grid (shift 3); the network predicts v = eps - x0 at timestep 1000 s, so
D = x_s - s v, affine in v, and diffusers' guidance_scale is the paper's w. The EDM-equivalent level is
sigma = s / (1 - s). D0 is the empty prompt, which is also what the pipeline encodes when no negative
prompt is given. Text is encoded with 256 T5 tokens, as the pass behind the paper did; the transformer runs
in bf16 on x_s cast to bf16, and D is formed in float32. Latents are the VAE's mode (a sample would add
encoder noise) in float32, normalised by the VAE's shift and scale.

Batching follows the stored pass exactly (forward batches of 8, text encoded in blocks of 16 aligned to
row 0), so a fresh pass can be compared with it photo by photo.
"""
import numpy as np
import torch

from .. import guidance as G
from .. import noise, runs
from ..data import coco, folder
from . import hub
from .base import Model, Spec, Split

COCO = {"coco": Split(calib="COCO val2014 photos 0-5,704", n_calib=coco.N_CALIB,
                      eval="photos 5,705-15,704", n_eval=coco.N_EVAL, reference="those 10,000 photos",
                      seed="per photo")}
# Fig 2 only (moments, never sampled): DOCCI's frozen 2,000 test photos, with the full description or its first
# sentence as the prompt, and MJHQ-30K's first 5,705 images at 480^2
DOCCI = {"docci": Split(calib="DOCCI test photos 0-1,999, full description", n_calib=2000),
         "docci-short": Split(calib="DOCCI test photos 0-1,999, first sentence", n_calib=2000)}
MJHQ480 = {"mjhq": Split(calib="MJHQ-30K images 0-5,704 at 480^2", n_calib=5705)}
TEXT_BLOCK = 16


def captions(dataset, ids):
    """The prompts of rows `ids` of a text-to-image dataset."""
    if folder.is_folder(dataset):
        return folder.captions(dataset, ids)
    if dataset == "coco":
        return coco.captions(ids)
    if dataset in ("docci", "docci-short"):
        from ..data import docci
        return docci.captions(ids, short=dataset == "docci-short")
    if dataset == "mjhq":
        from ..data import mjhq
        return mjhq.captions(ids)
    raise KeyError(f"no captions for dataset {dataset!r}")


def calib_rows(spec, dataset, split):
    """Row ids of `dataset` for a model whose Spec lists it (COCO and MJHQ keep their own evaluation slices)."""
    assert dataset in spec.datasets, f"{spec.name} does not calibrate on {dataset!r} (it has {list(spec.datasets)})"
    if folder.is_folder(dataset):
        return folder.rows(dataset, split)
    if dataset == "coco":
        return coco.rows(split)
    if dataset in ("docci", "docci-short"):
        from ..data import docci
        return docci.rows(split, spec.datasets[dataset].n_calib)
    from ..data import mjhq
    return list(range(spec.datasets[dataset].n_calib)) if split == "calib" else mjhq.rows("eval")


class Fp32State:
    """Keep a diffusers flow pipeline's latent in float32 between steps, for the length of a `with` block.

    Stock diffusers keeps the latent in the network's dtype: FlowMatchEulerDiscreteScheduler.step computes in
    float32 but casts its result back to the model output's dtype, so in bf16 every update is rounded onto a grid
    of spacing 2^-7 near 1. The runs behind the paper's SD 3.5 numbers avoided that; this is
    their wrapper, installed and removed again so the cached
    pipeline is left as it was:
      * prepare_latents upcasts the initial latent (so the starting noise is the same bf16 draw),
      * the transformer gets its input cast to its own dtype (a forward pre-hook),
      * the scheduler step is handed a float32 model output, so its closing cast leaves the state in float32.
    The networks still run in their own dtype. `seen` records the dtypes that reached the step (the gate)."""

    def __init__(self, pipe):
        self.pipe, self.seen, self.h = pipe, set(), None

    def __enter__(self):
        pipe, net = self.pipe, self.pipe.transformer
        prep, step = pipe.prepare_latents, pipe.scheduler.step

        def prepare_latents(*a, **k):
            out = prep(*a, **k)
            return (out[0].float(),) + tuple(out[1:]) if isinstance(out, tuple) else out.float()

        def cast_input(mod, args, kw):
            if "hidden_states" in kw:
                kw["hidden_states"] = kw["hidden_states"].to(net.dtype)
            elif args:
                args = (args[0].to(net.dtype),) + tuple(args[1:])
            return args, kw

        def fp32_step(model_output, timestep, sample, *a, **k):
            self.seen.add(str(sample.dtype))
            return step(model_output.float(), timestep, sample.float(), *a, **k)

        pipe.prepare_latents = prepare_latents
        pipe.scheduler.step = fp32_step
        self.h = net.register_forward_pre_hook(cast_input, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        del self.pipe.prepare_latents          # instance attributes: the class's own methods show through again
        del self.pipe.scheduler.step
        self.h.remove()
        return False


class SD3Family(Model):
    """Shared code for SD 3 / 3.5; subclasses set `spec`."""

    batch = 8
    row_align = TEXT_BLOCK
    dtype = torch.bfloat16
    size = coco.RENDER
    max_tokens = 256
    grid_steps = 0           # the calibration grid's step count when it is not the sampler's (SD 3 Medium)

    def __init__(self):
        self.pipe = None
        self._block = (None, None)

    # --- loading and levels ------------------------------------------------------------------------------
    def load(self):
        if self.pipe is not None:
            return
        from diffusers import AutoencoderKL, StableDiffusion3Pipeline
        path = hub.path(self.spec.checkpoint)
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)   # never half-rounded
        pipe = StableDiffusion3Pipeline.from_pretrained(path, vae=vae, torch_dtype=self.dtype)
        self.pipe = pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.transformer.eval().requires_grad_(False)
        sch = self.pipe.scheduler
        sch.set_timesteps(self.grid_steps or self.spec.steps)  # what the pipeline does when sigmas is None
        self.s = sch.sigmas[:-1].to(torch.float32).double().numpy()
        self.ts = sch.timesteps.to(torch.float32).numpy()
        self.null = self.encode([""])

    def levels(self):
        self.load()
        with np.errstate(divide="ignore"):
            return dict(sigma=np.where(self.s < 1, self.s / (1 - self.s), np.inf), t=self.ts.astype(np.float64))

    # --- data ----------------------------------------------------------------------------------------------
    def rows(self, dataset="coco", split="calib"):
        return calib_rows(self.spec, dataset, split)

    def captions(self, dataset, ids):
        return captions(dataset, ids)

    @torch.no_grad()
    def encode(self, prompts):
        e, _, p, _ = self.pipe.encode_prompt(prompt=list(prompts), prompt_2=None, prompt_3=None, device="cuda",
                                             do_classifier_free_guidance=False,
                                             max_sequence_length=self.max_tokens)
        return e, p

    def text(self, rows, dataset="coco"):
        """Prompt embeddings for `rows`, encoded in blocks of 16 aligned to row 0 (one block kept)."""
        out_e, out_p = [], []
        for r in rows:
            b0 = r - r % TEXT_BLOCK
            if self._block[0] != (dataset, b0):
                self._block = ((dataset, b0), self.encode(captions(dataset, range(b0, b0 + TEXT_BLOCK))))
            e, p = self._block[1]
            out_e.append(e[r - b0])
            out_p.append(p[r - b0])
        return torch.stack(out_e), torch.stack(out_p)

    @torch.no_grad()
    def latents(self, images, chunk=8):
        """VAE mode, normalised; encoded 8 images at a time, as the stored pass was (float32 at 480^2)."""
        vae = self.pipe.vae
        shift, scale = vae.config.shift_factor or 0.0, vae.config.scaling_factor
        return torch.cat([(vae.encode(x.to("cuda", torch.float32)).latent_dist.mode() - shift) * scale
                          for x in images.split(chunk)])

    def images(self, dataset, rows):
        """COCO and MJHQ as 480^2 squares; DOCCI's whole frame at 1152 x 864, as the DOCCI pass rendered it."""
        if folder.is_folder(dataset):
            return folder.images(dataset, rows, self.size)
        if dataset == "coco":
            return coco.images(rows, self.size)
        if dataset in ("docci", "docci-short"):
            from ..data import docci
            return docci.images(rows, "frame")
        from ..data import mjhq
        return mjhq.images(rows, self.size)

    def data(self, dataset, rows):
        self.load()
        return self.latents(self.images(dataset, rows)), self.text(rows, dataset)

    # --- the two predictions -------------------------------------------------------------------------------
    def noise(self, x0, k, eps):
        s = torch.tensor(float(self.s[k]), device=x0.device)
        return (1 - s) * x0 + s * eps.to(x0.device)

    def velocity(self, x, k, emb, pool):
        t = torch.tensor(self.ts[k], device=x.device, dtype=torch.float32).expand(len(x))
        return self.pipe.transformer(hidden_states=x.to(self.dtype), timestep=t, encoder_hidden_states=emb,
                                     pooled_projections=pool, return_dict=False)[0]

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        emb, pool = cond
        ne, npl = (v.expand(len(x), *v.shape[1:]) for v in self.null)
        s = torch.tensor(float(self.s[k]), device=x.device)
        v1 = self.velocity(x, k, emb, pool)
        v0 = self.velocity(x, k, ne, npl)
        return x - s * v1.to(torch.float32), x - s * v0.to(torch.float32)

    # --- checks and sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def smoke(self):
        """The pipeline's own first transformer call equals ours: same timestep, same text embeddings, same
        network output on its input, and its guided prediction equals D0 + w (D1 - D0)."""
        self.load()
        cap = captions(self.spec.default_dataset, [0])
        seen = []
        h = self.pipe.transformer.register_forward_hook(
            lambda m, a, kw, out: seen.append(({k: v.detach().clone() for k, v in kw.items() if torch.is_tensor(v)},
                                               out[0].detach().clone())), with_kwargs=True)

        def stop(pipe, i, t, kw):
            pipe._interrupt = True
            return kw
        try:
            self.pipe(prompt=cap, height=self.size, width=self.size, num_inference_steps=self.spec.steps,
                      guidance_scale=self.spec.published_w, max_sequence_length=self.max_tokens,
                      generator=torch.Generator("cuda").manual_seed(0), output_type="latent",
                      callback_on_step_end=stop)
        finally:
            h.remove()
        kw, out = seen[0]
        e, p = self.encode(cap)
        x = kw["hidden_states"]                      # [2, ...]: (uncond, cond), as the pipeline batches CFG
        assert torch.equal(kw["timestep"][0].float(), torch.tensor(self.ts[0], device=x.device)), "timestep"
        assert torch.equal(kw["encoder_hidden_states"][1], e[0]) and torch.equal(kw["pooled_projections"][1], p[0])
        assert torch.equal(kw["encoder_hidden_states"][0], self.null[0][0]), "the uncond branch is the empty prompt"
        ours = self.pipe.transformer(hidden_states=x, timestep=kw["timestep"],
                                     encoder_hidden_states=torch.cat([self.null[0], e]),
                                     pooled_projections=torch.cat([self.null[1], p]), return_dict=False)[0]
        assert torch.equal(ours, out), "network output on the pipeline's own batch"
        # our per-branch calls (batch 1 each, as the moments pass makes them) against the pipeline's batch of 2
        s = float(self.s[0])
        xs = x[:1].float()
        d1, d0 = self.denoise_pair(xs, 0, (e, p))
        vu, vc = out.float().chunk(2)
        ref = (xs - s * vu) + self.spec.published_w * ((xs - s * vc) - (xs - s * vu))
        got = d0 + self.spec.published_w * (d1 - d0)
        rel = float((got - ref).abs().max() / ref.abs().max())
        print(f"{self.spec.name} smoke: timestep, embeddings and network output identical; "
              f"guided prediction from separate branch calls within {rel:.1e} of the pipeline's (bf16 batching, "
              f"amplified by w)")
        assert rel < 1e-1, "separate branch calls disagree with the pipeline beyond bf16 batching noise"

    @torch.no_grad()
    def decode(self, z):
        vae = self.pipe.vae
        x = vae.decode(z.to(torch.float32) / vae.config.scaling_factor + (vae.config.shift_factor or 0.0),
                       return_dict=False)[0]
        return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def initial_noise(self, rows):
        c, h = self.pipe.transformer.config.in_channels, self.size // self.pipe.vae_scale_factor
        return torch.stack([torch.randn((c, h, h), generator=noise.generator(self.seed, r, 1)) for r in rows])

    GUIDE_NOMINAL = 2.0      # the pipeline's own scale while a guide is set: any value > 1 (see guide_hook)

    @torch.no_grad()
    def generate(self, prompts, z, w, trace=False, iso=None, guide=None, fp32_state=False):
        """Final latents of the released sampler from initial noise z, at weights w (one per level).

        iso = (eps [B, S, ...], std [S]) adds std[k] eps[:, k] to the clean prediction at level k (the noise
        control, see iso_hook). guide (vmg/baselines.py) replaces the guided prediction at every level, and w is
        then ignored (guide_hook). fp32_state keeps the latent in float32 between steps (Fp32State), as the runs
        behind the paper's SD 3.5 numbers did."""
        import contextlib
        lv = self.levels()
        if guide is not None:
            assert iso is None, "a guide and the noise control do not combine"
            guide.reset()
            w = np.full(len(lv["t"]), self.GUIDE_NOMINAL)
        w0, cb, _ = G.diffusers_hook(w, lv["t"])
        tr = G.WeightTrace(self.pipe, self.pipe.transformer) if trace else None
        hooks = []
        if iso:
            hooks.append(self.pipe.transformer.register_forward_hook(self.iso_hook(*iso), with_kwargs=True))
        if guide is not None:
            hooks.append(self.pipe.transformer.register_forward_hook(self.guide_hook(guide), with_kwargs=True))
        f32 = Fp32State(self.pipe) if fp32_state else contextlib.nullcontext()
        try:
            with f32:
                out = self.pipe(prompt=list(prompts), height=self.size, width=self.size,
                                num_inference_steps=self.spec.steps, guidance_scale=G.LiveScale(self.pipe, w0),
                                max_sequence_length=self.max_tokens, latents=z.to("cuda", self.dtype),
                                output_type="latent", callback_on_step_end=cb).images
        finally:
            if tr:
                tr.remove()
            for h in hooks:
                h.remove()
        if fp32_state:      # the gate: every scheduler step saw a float32 state
            assert f32.seen == {"torch.float32"}, f"fp32 state: the step saw {f32.seen}"
        return (out, tr.seen) if trace else out

    def guide_hook(self, guide, log=None):
        """Forward hook on the transformer that makes the pipeline step with guide(k, v_u, v_c) at level k.

        The pipeline batches (uncond, cond) and forms v_u' + g (v_c' - v_u'). The hook writes the guide's output
        into both halves, so that is guide(...) + g * 0: exactly the guide's output, whatever g, and the
        pipeline's own scale only has to stay above 1 (GUIDE_NOMINAL). `log` collects the level of every call."""
        ts = torch.as_tensor(self.ts, dtype=torch.float64)

        def hook(mod, args, kw, out):
            k = int(torch.argmin((ts - float(kw["timestep"].flatten()[0])).abs()))
            v = out[0]
            assert v.shape[0] % 2 == 0, v.shape
            vu, vc = v.chunk(2)
            g = guide(k, vu, vc).to(v.dtype)
            if log is not None:
                log.append(k)
            return (torch.cat([g, g]),) + tuple(out[1:])
        return hook

    DDPM_STREAMS = (12, 13, 14)     # noise.generator streams of vmg/ddpm.py's draws: reverse step, clean draw, rank-1

    def ddpm_noise(self, rows):
        """Per-row draws of the posterior-DDPM sampler, one per level: z and xi [B, S, C, H, W] (the reverse step's
        noise and the clean draw's), g [B, S] (the rank-1 weight). Every DDPM arm shares them row by row."""
        c, h = self.pipe.transformer.config.in_channels, self.size // self.pipe.vae_scale_factor
        S = len(self.s)
        sz, sx, sg = self.DDPM_STREAMS
        z, xi, g = ([torch.randn(shape, generator=noise.generator(self.seed, r, st)) for r in rows]
                    for st, shape in ((sz, (S, c, h, h)), (sx, (S, c, h, h)), (sg, (S,))))
        return torch.stack(z), torch.stack(xi), torch.stack(g)

    @torch.no_grad()
    def generate_loop(self, prompts, z, sampler=None, w=None, draws=None):
        """Final latents from our own copy of the released loop, with a float32 latent state.

        sampler None: the released Euler step with CFG at the constant weight w, which must equal
        generate(prompts, z, w, fp32_state=True) (the gate in reproduce/experiments/ddpm_posterior.py). Otherwise
        a vmg.ddpm.PosteriorDDPM, stepping with draws = ddpm_noise(rows). The text is encoded as the pipeline
        encodes it (the batch's prompts with the empty negative), the network sees (uncond, cond) in one batch
        of 2B in its own dtype, and both predictions D = x - s v are formed in float32."""
        from .. import ddpm as DD
        self.load()
        sch = self.pipe.scheduler
        sch.set_timesteps(self.spec.steps, device="cuda")
        sig, ts = sch.sigmas, sch.timesteps
        assert np.array_equal(sig[:-1].cpu().double().numpy(), self.s) and float(sig[-1]) == 0, "sigma grid"
        e, ne, p, npl = self.pipe.encode_prompt(prompt=list(prompts), prompt_2=None, prompt_3=None, device="cuda",
                                                do_classifier_free_guidance=True, max_sequence_length=self.max_tokens)
        emb, pool = torch.cat([ne, e]), torch.cat([npl, p])
        net = self.pipe.transformer
        x = z.to("cuda", self.dtype).float()        # the pipeline's start under Fp32State: the bf16 draw, upcast
        for k in range(len(ts)):
            xin = torch.cat([x, x]).to(self.dtype)
            v = net(hidden_states=xin, timestep=ts[k].expand(len(xin)), encoder_hidden_states=emb,
                    pooled_projections=pool, return_dict=False)[0]
            vu, vc = v.chunk(2)
            if sampler is None:
                x = DD.euler_cfg(x, sig[k], sig[k + 1], vu, vc, float(w))
                continue
            s = sig[k]
            d1, d0 = x - s * vc.float(), x - s * vu.float()
            zk, xk, gk = (t[:, k].to("cuda") for t in draws)
            x = sampler.step(x, float(sig[k]), float(sig[k + 1]), d0, d1, xk, gk, zk)
        return x

    def sample_ddpm(self, dataset, rows, sampler, outdir, batch=8):
        """PNGs <row>.png from the posterior-DDPM sampler (vmg/ddpm.py); prompts and initial noise as in sample."""
        self.load()
        for b0 in range(0, len(rows), batch):
            rb = [r for r in rows[b0:b0 + batch] if not (outdir / f"{r:05d}.png").exists()]
            if not rb:
                continue
            x = self.generate_loop(captions(dataset, rb), self.initial_noise(rb), sampler, draws=self.ddpm_noise(rb))
            for r, im in zip(rb, self.decode(x)):
                runs.save_image(im, outdir / f"{r:05d}.png")

    ISO_STREAM = 8          # noise.generator streams of the noise controls' draws
    SHAPE_STREAM = 9

    def iso_eps(self, rows):
        """[B, S, C, H, W] standard-normal draws of the noise control, one per row and level, seeded per row."""
        c, h = self.pipe.transformer.config.in_channels, self.size // self.pipe.vae_scale_factor
        S = len(self.s)
        return torch.stack([torch.randn((S, c, h, h), generator=noise.generator(self.seed, r, self.ISO_STREAM))
                            for r in rows])

    def calib_x0(self, dataset):
        """[N, C, H, W] float32 latents of the calibration rows, encoded once and cached in the run folder."""
        import os
        path = runs.run_dir(self.spec.key, dataset) / "x0-calib.npy"
        if not path.exists():
            self.load()
            rows = self.rows(dataset, "calib")
            x = torch.cat([self.latents(self.images(dataset, rows[i:i + 64])).cpu() for i in range(0, len(rows), 64)])
            tmp = path.with_name(f"x0-calib.tmp{os.getpid()}.npy")
            np.save(tmp, x.numpy())
            os.replace(tmp, path)
        return np.load(path)

    def shape_basis(self, dataset):
        """The N calibration latents [N, d], centred and scaled to ||X||_F^2 = d, so that g @ X for g ~ N(0, I_N)
        has covariance X^T X = Cov(x0) / (tr Cov(x0) / d): the data's empirical covariance (rank N - 1) at unit
        variance per coordinate."""
        if getattr(self, "_shape", (None,))[0] != dataset:
            X = torch.from_numpy(self.calib_x0(dataset)).to("cuda", torch.float64).flatten(1)
            X -= X.mean(0)
            X *= (X.shape[1] / X.pow(2).sum()).sqrt()           # now ||X||_F^2 = d, so E||g X||^2 = d
            self._shape = (dataset, X.float())
        return self._shape[1]

    def shape_eps(self, rows, dataset):
        """[B, S, C, H, W] Gaussian draws with the data's covariance at unit variance per coordinate, one per row
        and level, seeded per row (N standard normals each, combined over the N centred calibration latents)."""
        X = self.shape_basis(dataset)
        c, h = self.pipe.transformer.config.in_channels, self.size // self.pipe.vae_scale_factor
        S = len(self.s)
        g = torch.stack([torch.randn((S, X.shape[0]), generator=noise.generator(self.seed, r, self.SHAPE_STREAM))
                         for r in rows]).to("cuda")
        return (g @ X).reshape(len(rows), S, c, h, h)

    def iso_hook(self, eps, std, log=None):
        """Forward hook on the transformer that adds std[k] eps[:, k] to D = x_s - s v at level k, i.e.
        v -= std[k] eps[:, k] / s.

        It shifts every branch of a CFG call alike, so the pipeline's v0 + g (v1 - v0) carries the shift exactly
        once, whatever g. `log` collects the level of every call."""
        ts = torch.as_tensor(self.ts, dtype=torch.float64)

        def hook(mod, args, kw, out):
            k = int(torch.argmin((ts - float(kw["timestep"].flatten()[0])).abs()))
            v = out[0]
            B = eps.shape[0]
            assert v.shape[0] % B == 0, (v.shape, B)
            if log is not None:
                log.append(k)
            if std[k] == 0:
                return out
            dv = (float(std[k]) / float(self.s[k])) * eps[:, k].to(v.device, torch.float32)
            v = (v.float() - dv.repeat(v.shape[0] // B, *(1,) * (dv.dim() - 1))).to(v.dtype)
            return (v,) + tuple(out[1:])
        return hook

    def sample(self, dataset, rows, w, outdir, batch=8, noise_std=None, noise_shape="iso", noise_dataset=None,
               guide=None, fp32_state=False):
        """PNGs <row>.png for the evaluation rows; prompt i and initial noise i are shared by every arm.

        noise_std (one per level): add zero-mean Gaussian noise of that std per coordinate to the clean prediction
        at every level, isotropic (noise_shape "iso", sample.py --isonoise) or with the covariance of the
        calibration latents of noise_dataset ("data", --shapenoise). guide / fp32_state: see generate."""
        self.load()
        ws = G.weights(self.spec.published_w if w is None else w, self.levels())
        for b0 in range(0, len(rows), batch):
            rb = [r for r in rows[b0:b0 + batch] if not (outdir / f"{r:05d}.png").exists()]
            if not rb:
                continue
            iso = None
            if noise_std is not None:
                assert noise_shape in ("iso", "data"), noise_shape
                eps = self.iso_eps(rb).to("cuda") if noise_shape == "iso" else self.shape_eps(rb, noise_dataset or dataset)
                iso = (eps, np.asarray(noise_std, dtype=np.float64))
            # only the options in use, so subclasses whose generate() has none of them (SD 1.5) still sample
            kw = {k: v for k, v in dict(iso=iso, guide=guide, fp32_state=fp32_state).items() if v}
            ims = self.decode(self.generate(self.captions(dataset, rb), self.initial_noise(rb), ws, **kw))
            for r, im in zip(rb, ims):
                runs.save_image(im, outdir / f"{r:05d}.png")


class SD35Medium(SD3Family):
    spec = Spec(key="sd35m", name="SD 3.5 Medium", modality="image",
                checkpoint="stabilityai/stable-diffusion-3.5-medium", reference="empty prompt",
                sampler="flow-matching Euler", steps=40, output="480^2 image", published_w=4.5,
                datasets={**COCO, "docci": DOCCI["docci"], **MJHQ480}, calib_levels=40)


class SD35Large(SD3Family):
    batch = 4
    spec = Spec(key="sd35l", name="SD 3.5 Large", modality="image",
                checkpoint="stabilityai/stable-diffusion-3.5-large", reference="empty prompt",
                sampler="flow-matching Euler", steps=28, output="480^2 image", published_w=3.5,
                datasets=COCO, calib_levels=28)


def _capture(module):
    """Forward hook recording (args, tensor kwargs, output) of every call of `module`; returns (calls, handle)."""
    calls = []

    def hook(m, args, kw, out):
        o = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
        calls.append((tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args),
                      {k: v.detach().clone() for k, v in kw.items() if torch.is_tensor(v)}, o.detach().clone()))
    return calls, module.register_forward_hook(hook, with_kwargs=True)


def _stop_after_first_step(pipe, i, t, kw):
    pipe._interrupt = True
    return kw


class SD15(Model):
    """SD 1.5: UNet, eps-prediction on the DDPM scaled-linear schedule, PNDM sampler with 50 steps.

    x_t = sqrt(a) x0 + sqrt(1 - a) eps and D = (x_t - sqrt(1 - a) eps_hat) / sqrt(a), a = alphas_cumprod[t].
    PNDM visits one timestep twice; each distinct timestep is one level (50). fp16 network, as the model card
    runs it; COCO photos at 480 x 480 like the SD 3.5 models.
    """

    spec = Spec(key="sd15", name="SD 1.5", modality="image", checkpoint="stable-diffusion-v1-5/stable-diffusion-v1-5",
                reference="empty prompt", sampler="PNDM", steps=50, output="480^2 image", published_w=7.5,
                datasets={**COCO, **DOCCI}, calib_levels=50)
    batch = 32
    row_align = 64
    dtype = torch.float16
    size = coco.RENDER

    def __init__(self):
        self.pipe = None
        self._block = (None, None)

    def load(self):
        if self.pipe is not None:
            return
        from diffusers import AutoencoderKL, StableDiffusionPipeline
        path = hub.path(self.spec.checkpoint)
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)   # never half-rounded
        pipe = StableDiffusionPipeline.from_pretrained(path, vae=vae, torch_dtype=self.dtype, safety_checker=None,
                                                       requires_safety_checker=False)
        self.pipe = pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        sch = self.pipe.scheduler
        sch.set_timesteps(self.spec.steps)
        t = np.unique(sch.timesteps.cpu().numpy())[::-1].copy()
        a = sch.alphas_cumprod[torch.from_numpy(t)].double()
        self.sqrt_a, self.sqrt_1ma = a.sqrt().numpy(), (1 - a).sqrt().numpy()
        self.sig, self.ts = ((1 - a) / a).sqrt().numpy(), t
        self.null = self.encode([""])

    def levels(self):
        self.load()
        return dict(sigma=self.sig.copy(), t=self.ts.astype(np.float64))

    def rows(self, dataset="coco", split="calib"):
        return calib_rows(self.spec, dataset, split)

    def captions(self, dataset, ids):
        return captions(dataset, ids)

    @torch.no_grad()
    def encode(self, prompts):
        return self.pipe.encode_prompt(list(prompts), "cuda", 1, False)[0]

    def text(self, rows, dataset="coco"):
        out = []
        for r in rows:
            b0 = r - r % self.row_align
            if self._block[0] != (dataset, b0):
                self._block = ((dataset, b0), self.encode(captions(dataset, range(b0, b0 + self.row_align))))
            out.append(self._block[1][r - b0])
        return torch.stack(out)

    @torch.no_grad()
    def latents(self, images, chunk=8):
        vae = self.pipe.vae
        return torch.cat([vae.encode(x.to("cuda", torch.float32)).latent_dist.mode() * vae.config.scaling_factor
                          for x in images.split(chunk)])

    def images(self, dataset, rows):
        """COCO's 480^2 crops; on DOCCI the centre square at 512^2, the crop SD 1.x was trained on."""
        if folder.is_folder(dataset):
            return folder.images(dataset, rows, self.size)
        if dataset == "coco":
            return coco.images(rows, self.size)
        from ..data import docci
        assert dataset in ("docci", "docci-short"), dataset
        return docci.images(rows, "square")

    def data(self, dataset, rows):
        self.load()
        return self.latents(self.images(dataset, rows)), self.text(rows, dataset)

    def noise(self, x0, k, eps):
        return float(self.sqrt_a[k]) * x0 + float(self.sqrt_1ma[k]) * eps.to(x0.device)

    def eps(self, x, k, emb):
        t = torch.tensor(int(self.ts[k]), device=x.device)
        return self.pipe.unet(x.to(self.dtype), t, encoder_hidden_states=emb, return_dict=False)[0]

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        null = self.null.expand(len(x), *self.null.shape[1:])
        d = lambda e: (x - float(self.sqrt_1ma[k]) * e.to(torch.float32)) / float(self.sqrt_a[k])
        return d(self.eps(x, k, cond)), d(self.eps(x, k, null))

    @torch.no_grad()
    def smoke(self):
        self.load()
        cap = captions(self.spec.default_dataset, [0])
        calls, h = _capture(self.pipe.unet)
        try:
            self.pipe(prompt=cap, height=self.size, width=self.size, num_inference_steps=self.spec.steps,
                      guidance_scale=self.spec.published_w, generator=torch.Generator("cuda").manual_seed(0),
                      output_type="latent", callback_on_step_end=_stop_after_first_step)
        finally:
            h.remove()
        args, kw, out = calls[0]
        x, t = args[0], args[1]
        e = self.encode(cap)
        assert int(t) == int(self.ts[0]), f"timestep {t} vs {self.ts[0]}"
        assert torch.equal(kw["encoder_hidden_states"][0], self.null[0]) and torch.equal(kw["encoder_hidden_states"][1], e[0])
        ours = self.pipe.unet(x, t, encoder_hidden_states=torch.cat([self.null, e]), return_dict=False)[0]
        assert torch.equal(ours, out), "network output on the pipeline's own batch"
        print(f"{self.spec.name} smoke: timestep, embeddings and network output identical")

    @torch.no_grad()
    def decode(self, z):
        vae = self.pipe.vae
        x = vae.decode(z.to(torch.float32) / vae.config.scaling_factor, return_dict=False)[0]
        return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def initial_noise(self, rows):
        h = self.size // self.pipe.vae_scale_factor
        return torch.stack([torch.randn((self.pipe.unet.config.in_channels, h, h), generator=noise.generator(self.seed, r, 1))
                            for r in rows])

    @torch.no_grad()
    def generate(self, prompts, z, w, trace=False):
        w0, cb, _ = G.diffusers_hook(w, self.levels()["t"])
        tr = G.WeightTrace(self.pipe, self.pipe.unet) if trace else None
        try:
            out = self.pipe(prompt=list(prompts), height=self.size, width=self.size, num_inference_steps=self.spec.steps,
                            guidance_scale=G.LiveScale(self.pipe, w0), latents=z.to("cuda", self.dtype),
                            output_type="latent", callback_on_step_end=cb).images
        finally:
            if tr:
                tr.remove()
        return (out, tr.seen) if trace else out

    sample = SD3Family.sample


class Flux2Klein(Model):
    """FLUX.2 [klein] Base 4B: rectified flow on patchified, batch-normalised latents, MJHQ-30K at 1024^2.

    The sampler, the guidance line and the velocity all live in the pipeline's own latent space: the 32-channel
    VAE mode with each 2x2 patch folded into channels (128 at 16x), standardised by the VAE's batch-norm running
    statistics. Text is three hidden layers of Qwen3-4B padded to a fixed 512 tokens, so encoding is batch
    independent. The level grid has a resolution-dependent shift (compute_empirical_mu of the packed sequence
    length). The timestep reaches the network cast to bf16 and then divided by 1000, as in the pipeline.
    """

    spec = Spec(key="flux2-klein", name="FLUX.2 [klein] 4B", modality="image",
                checkpoint="black-forest-labs/FLUX.2-klein-base-4B", reference="empty prompt",
                sampler="flow-matching Euler", steps=50, output="1024^2 image", published_w=4.0,
                datasets={"mjhq": Split(calib="MJHQ-30K images 0-1,023", n_calib=1024, eval="images 5,705-15,704",
                                        n_eval=10000, reference="those 10,000 images", seed="per image")},
                calib_levels=50)
    batch = 4
    dtype = torch.bfloat16
    size = 1024
    max_seq = 512

    def __init__(self):
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        path = hub.path(self.spec.checkpoint)
        self.P = Flux2KleinPipeline
        self.pipe = Flux2KleinPipeline.from_pretrained(path, torch_dtype=self.dtype).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.vae32 = AutoencoderKLFlux2.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
        sch = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        steps = self.spec.steps
        seq = (self.size // 16) ** 2
        sch.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=compute_empirical_mu(image_seq_len=seq, num_steps=steps))
        self.s = sch.sigmas[:-1].to(torch.float32).double().numpy()
        self.ts = sch.timesteps.to(torch.float32).numpy()
        self.null = self.encode([""])

    def levels(self):
        self.load()
        with np.errstate(divide="ignore"):
            return dict(sigma=np.where(self.s < 1, self.s / (1 - self.s), np.inf), t=self.ts.astype(np.float64))

    def rows(self, dataset="mjhq", split="calib"):
        from ..data import mjhq
        if folder.is_folder(dataset):
            return folder.rows(dataset, split)
        assert dataset == "mjhq", dataset
        return mjhq.rows(split)

    @torch.no_grad()
    def encode(self, prompts):
        return self.pipe.encode_prompt(prompt=list(prompts), device="cuda", max_sequence_length=self.max_seq)[0]

    @torch.no_grad()
    def latents(self, images, chunk=4):
        vae = self.vae32
        z = self.P._patchify_latents(torch.cat([vae.encode(x.to("cuda", torch.float32)).latent_dist.mode()
                                                for x in images.split(chunk)]))
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(z)
        return (z - mean) / std

    def data(self, dataset, rows):
        from ..data import mjhq
        self.load()
        if folder.is_folder(dataset):
            return self.latents(folder.images(dataset, rows, self.size)), self.encode(folder.captions(dataset, rows))
        return self.latents(mjhq.images(rows, self.size)), self.encode(mjhq.captions(rows))

    def noise(self, x0, k, eps):
        s = torch.tensor(float(self.s[k]), device=x0.device)
        return (1 - s) * x0 + s * eps.to(x0.device)

    def velocity(self, x, k, emb):
        P, b, h = self.P, len(x), self.size // 16
        t = torch.full((b,), float(self.ts[k]), device=x.device, dtype=self.dtype) / 1000
        v = self.pipe.transformer(hidden_states=P._pack_latents(x.to(self.dtype)), timestep=t, guidance=None,
                                  encoder_hidden_states=emb, txt_ids=P._prepare_text_ids(emb).to(x.device),
                                  img_ids=P._prepare_latent_ids(x).to(x.device), return_dict=False)[0]
        return v.permute(0, 2, 1).reshape(b, -1, h, h)

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        s = torch.tensor(float(self.s[k]), device=x.device)
        null = self.null.expand(len(x), *self.null.shape[1:])
        return x - s * self.velocity(x, k, cond).float(), x - s * self.velocity(x, k, null).float()

    @torch.no_grad()
    def smoke(self):
        """Every network call of the pipeline's first step: its text is ours (cond or the empty prompt), its timestep
        is ours, and the network reproduces its output on the same inputs."""
        from ..data import mjhq
        self.load()
        cap = mjhq.captions([0])
        calls, h = _capture(self.pipe.transformer)
        try:
            self.pipe(prompt=cap, height=self.size, width=self.size, num_inference_steps=self.spec.steps,
                      guidance_scale=self.spec.published_w, generator=torch.Generator("cpu").manual_seed(0),
                      output_type="latent", callback_on_step_end=_stop_after_first_step)
        finally:
            h.remove()
        e = self.encode(cap)
        assert calls, "no transformer call captured"
        kinds = []
        for _, kw, out in calls:
            emb = kw["encoder_hidden_states"]
            for r in range(len(emb)):
                kinds.append("cond" if torch.equal(emb[r], e[0]) else "null" if torch.equal(emb[r], self.null[0]) else "?")
            t = torch.full((len(emb),), float(self.ts[0]), device=emb.device, dtype=self.dtype) / 1000
            assert torch.equal(kw["timestep"], t), f"timestep {kw['timestep']} vs {t}"
            again = self.pipe.transformer(**{k: kw[k] for k in ("hidden_states", "timestep", "encoder_hidden_states",
                                                                "txt_ids", "img_ids")}, guidance=None, return_dict=False)[0]
            assert torch.equal(again, out), "network output on the pipeline's own inputs"
        assert sorted(kinds) == ["cond", "null"], f"text branches seen: {kinds}"
        print(f"{self.spec.name} smoke: {len(calls)} call(s) at step 0, branches {kinds}; timestep and network output identical")

    def sample(self, dataset, rows, w, outdir):
        """One row at a time: row r's initial noise comes from its own generator, shared by every arm."""
        from ..data import mjhq
        self.load()
        ws = G.weights(self.spec.published_w if w is None else w, self.levels())
        for r in rows:
            path = outdir / f"{r:05d}.png"
            if path.exists():
                continue
            w0, cb, _ = G.diffusers_hook(ws, self.levels()["t"])
            im = self.pipe(prompt=mjhq.captions([r]), height=self.size, width=self.size,
                           num_inference_steps=self.spec.steps, guidance_scale=G.LiveScale(self.pipe, w0),
                           generator=noise.generator(self.seed, r, 1), callback_on_step_end=cb).images[0]
            runs.save_image(im, path)
