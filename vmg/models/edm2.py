"""NVIDIA's EDM2 (ImageNet-512 latents, ImageNet-64 pixels) and EDM (CIFAR-10) pairs: Fig 2's autoguidance and
class-CFG groups, and EDM2-S's classifier-free-guidance preset (`edm2-512`, sampled for tab:tuned-w).

Every pair is two released pickles combined the way edm2/generate_images.py does, D = D0.lerp(D1, w), so its
`guidance` is the paper's w: autoguidance guides a model with a smaller or shorter-trained copy of itself
(Karras et al. 2024), and EDM2's CFG guides the conditional model with a separately trained unconditional one
(EDM2 and EDM never drop labels, so there is no null class). The process is EDM's x = x0 + sigma eps, and the
levels are the Karras grid of EDM's deterministic Heun sampler (32 steps, 18 for CIFAR-10) with every level
the sampler evaluates; the guided sampler below reproduces edm_sampler bit for bit at a constant weight.

The moments pass runs the networks in float32 (force_fp32, as the passes behind Fig 2 did); sampling uses the
release's own precision (fp16 inside EDM2's UNet). Calibration data are in vmg/data/imagenet.py and
vmg/data/cifar10.py; checkpoints and the edm2 clone in vmg/models/_edm.
"""
from pathlib import Path

import numpy as np
import torch

from .. import guidance as G, runs
from .. import noise as N
from ..data import cifar10, folder, imagenet
from . import _edm
from ._edm import edm1, posthoc, raw, rebuilt
from .base import Model, Spec, Split

HEUN = "EDM Heun (2nd order), Karras sigma, rho 7"


class EDMHeun(Model):
    """The VE process and EDM's deterministic Heun sampler with one weight per level. Subclasses implement
    pair(x, t, labels, released, ref) -> (D1, D0), initial(seeds) -> (noise, labels) and decode(x) -> uint8."""

    steps = 32
    sigma_min, sigma_max, rho = 0.002, 80.0, 7.0
    state_dtype = torch.float32          # EDM2's edm_sampler; EDM (v1) runs its sampler in float64
    batch = 64
    sample_batch = 32

    def t_steps(self, device="cpu"):
        """The sampler's own t_steps (with the trailing 0), computed exactly as the released sampler does."""
        i = torch.arange(self.steps, dtype=self.state_dtype, device=device)
        a, b = self.sigma_max ** (1 / self.rho), self.sigma_min ** (1 / self.rho)
        t = (a + i / (self.steps - 1) * (b - a)) ** self.rho
        return torch.cat([t, torch.zeros_like(t[:1])])

    def levels(self):
        s = self.t_steps()[:-1].double().numpy()
        return dict(sigma=s, t=s.copy())

    def noise(self, x0, k, eps):
        return x0 + float(self.levels()["sigma"][k]) * eps.to(x0.device)

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        t = torch.full((len(x),), float(self.levels()["sigma"][k]), device=x.device, dtype=torch.float32)
        d1, d0 = self.pair(x, t, cond)
        return d1.float(), d0.float()

    @torch.no_grad()
    def heun(self, noise, labels, ws, record=None):
        """edm_sampler (S_churn = 0) guided at the weight of the level each network call evaluates. If given,
        record(i, D) sees the guided prediction at the state step i starts from (reproduce/experiments/wvar_edm2.py);
        it observes only, so the samples are unchanged."""
        dt, ts, N = self.state_dtype, self.t_steps(noise.device), self.steps

        def guided(x, t, i):
            d1, d0 = self.pair(x, t, labels, released=True, ref=ws[i] != 1)
            return d1.to(dt) if ws[i] == 1 else d0.to(dt).lerp(d1.to(dt), float(ws[i]))

        x_next = noise.to(dt) * ts[0]
        for i, (t_cur, t_next) in enumerate(zip(ts[:-1], ts[1:])):
            x_hat, t_hat = x_next, t_cur
            d_hat = guided(x_hat, t_hat, i)
            if record is not None:
                record(i, d_hat)
            d_cur = (x_hat - d_hat) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < N - 1:
                d_prime = (x_next - guided(x_next, t_next, i + 1)) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next

    def sample(self, dataset, rows, w, outdir):
        """PNG <row>.png per evaluation row; row r is seed r, which fixes both its noise and its class."""
        self.load()
        pub = self.spec.published_w
        ws = G.weights((pub if np.isfinite(pub) else 1.0) if w is None else w, self.levels())
        for b0 in range(0, len(rows), self.sample_batch):
            rb = [r for r in rows[b0:b0 + self.sample_batch] if not (outdir / f"{r:05d}.png").exists()]
            if rb:
                noise, labels = self.initial(rb)
                ims = self.decode(self.heun(noise, labels, ws))
                for r, im in zip(rb, ims.permute(0, 2, 3, 1).cpu().numpy()):
                    runs.save_image(im, outdir / f"{r:05d}.png")

    def initial(self, seeds):
        """generate_images.py: per seed, the noise and then the class from one generator on the device."""
        rnd = _edm.StackedRandomGenerator("cuda", seeds)
        noise = rnd.randn([len(seeds), self.channels, self.res, self.res], device="cuda")
        if not self.label_dim:
            return noise, None
        return noise, torch.eye(self.label_dim, device="cuda")[rnd.randint(self.label_dim, size=[len(seeds)],
                                                                            device="cuda")]

    def classes(self, seeds):
        """The class row (seed) r is generated with, by the same draws as initial(): CLIP's prompts use it."""
        self.load()
        _, labels = self.initial(list(seeds))
        return labels.argmax(1).tolist() if labels is not None else [None] * len(seeds)

    # --- checks ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def mse_check(self, x0, cond, at=(5.0, 1.0, 0.2)):
        """Both predictions beat the best scalar linear denoiser and D1 is no worse than D0 (8 rows): a wrong
        parameterisation (time, scale, label convention) fails this by a wide margin."""
        lv, V = self.levels()["sigma"], float(x0.var())
        out = []
        for s in at:
            k = int(np.argmin(np.abs(np.log(lv) - np.log(s))))
            eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(k)).to(x0.device)
            d1, d0 = self.denoise_pair(self.noise(x0, k, eps), k, cond)
            m1, m0 = float(((d1 - x0) ** 2).mean()), float(((d0 - x0) ** 2).mean())
            lin = V * lv[k] ** 2 / (V + lv[k] ** 2)
            assert m1 < lin and m0 < lin and m1 < 1.1 * m0, \
                f"sigma {lv[k]:.3g}: MSE D1 {m1:.4g}, D0 {m0:.4g}, linear {lin:.4g}"
            out.append(f"sigma {lv[k]:.3g}: {m1:.3g} / {m0:.3g} (linear {lin:.3g})")
        return "; ".join(out)


# --- EDM2 and EDM pickles ------------------------------------------------------------------------------------
IN512 = "imagenet512"
IN64 = "imagenet64"
C10 = "cifar10"
_N = {IN512: 10000, IN64: 10000, C10: 10000}


def _split(dataset, n):
    if dataset == C10:
        return Split(calib="CIFAR-10 test images (10,000)", n_calib=n, eval="seeds 0-49,999, a random class each",
                     n_eval=50000, reference="the 50,000 train images", seed="per seed")
    what = "every ImageNet val image" if n >= imagenet.N_VAL else f"ImageNet val, {n // 1000} per class"
    return Split(calib=f"{what} ({n:,})", n_calib=n, eval="seeds 0-9,999, a random class each (generate_images.py)",
                 n_eval=10000, reference="10,000 ImageNet val images", seed="per seed")


class EDMPair(EDMHeun):
    """Two pickles (net = D1, guide = D0) on one dataset. Subclasses set spec, net_ckpt, guide_ckpt."""

    fp32 = True              # the moments pass; sampling always runs at the release's own precision

    def __init__(self):
        self.net = None

    @property
    def dataset(self):
        return self.spec.default_dataset

    def load(self):
        if self.net is None:
            self.net, self.encoder = _edm.edm_pickle(self.net_ckpt)
            self.gnet, _ = _edm.edm_pickle(self.guide_ckpt)
            self.label_dim = int(self.net.label_dim)
            self.res, self.channels = int(self.net.img_resolution), int(self.net.img_channels)

    def rows(self, dataset=None, split="calib"):
        if folder.is_folder(dataset or self.dataset):
            return folder.rows(dataset or self.dataset, split)
        sp = self.spec.datasets[dataset or self.dataset]
        if split != "calib":
            return list(range(sp.n_eval))
        return list(range(sp.n_calib)) if self.dataset == C10 else imagenet.calib_rows(sp.n_calib)

    def data(self, dataset, rows):
        self.load()
        if folder.is_folder(dataset):
            return self.folder_data(dataset, rows)
        if self.dataset == C10:
            x0, lab = cifar10.pixels(rows), cifar10.class_of(rows)
        elif self.dataset == IN64:
            x0, lab = imagenet.pixels(rows), imagenet.class_of(rows, 64)
        else:
            x0 = imagenet.latents(512, rows, self.encoder.scale, self.encoder.bias, self.seed)
            lab = imagenet.class_of(rows, 512)
        nl = max(self.label_dim, int(self.gnet.label_dim))       # 0: an unconditional pair
        return x0.cuda(), torch.nn.functional.one_hot(lab, nl).float().cuda() if nl else None

    @torch.no_grad()
    def folder_data(self, dataset, rows):
        """(x0, one-hot labels) of a folder dataset (vmg/data/folder.py) in the space the networks denoise: EDM2's
        latents through the pickle's own VAE encoder (one posterior draw per row, as vmg.data.imagenet.latents),
        or pixels in [-1, 1] for pixel-space networks."""
        enc = self.encoder
        if enc is not None and hasattr(enc, "scale") and hasattr(enc, "encode_pixels"):      # StabilityVAEEncoder
            raw = enc.encode_pixels(folder.uint8(dataset, rows, self.res * 8).cuda()).float()   # [B, 8, h, w]
            mean, std = raw[:, :4], raw[:, 4:]
            eps = torch.stack([torch.randn(mean.shape[1:], generator=N.generator(self.seed, r, 1)) for r in rows])
            sc = torch.as_tensor(np.asarray(enc.scale, np.float32), device=mean.device).reshape(1, -1, 1, 1)
            bi = torch.as_tensor(np.asarray(enc.bias, np.float32), device=mean.device).reshape(1, -1, 1, 1)
            x0 = (mean + eps.to(mean.device) * std) * sc + bi
        else:
            x0 = (folder.uint8(dataset, rows, self.res).float() / 127.5 - 1).cuda()
        nl = max(self.label_dim, int(self.gnet.label_dim))
        if not nl:
            return x0, None
        lab = torch.as_tensor(folder.labels(dataset, rows))
        return x0, torch.nn.functional.one_hot(lab, nl).float().cuda()

    def pair(self, x, t, labels, released=False, ref=True):
        f = (not released) and self.fp32
        d1 = self.net(x, t, labels if self.label_dim else None, force_fp32=f)
        d0 = self.gnet(x, t, labels if self.gnet.label_dim else None, force_fp32=f) if ref else None
        return d1, d0

    def decode(self, x):
        if self.encoder is not None:
            return self.encoder.decode(x)                  # StabilityVAEEncoder (512) or StandardRGBEncoder (64)
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)

    @torch.no_grad()
    def smoke(self):
        """(1) D1, D0 are the networks' own calls, and float32 agrees with the release's precision; (2) the
        denoising-MSE sanity check; (3) the guided sampler equals generate_images.edm_sampler bit for bit at a
        constant weight (EDM2 only: EDM's sampler has no guidance)."""
        self.load()
        rows = self.rows(self.dataset)[:8]
        x0, cond = self.data(self.dataset, rows)
        k = self.steps // 2
        x = self.noise(x0, k, torch.randn(x0.shape, generator=torch.Generator().manual_seed(0)).to(x0.device))
        d1, d0 = self.denoise_pair(x, k, cond)
        t = torch.full((len(x),), float(self.levels()["sigma"][k]), device=x.device)
        assert torch.equal(d1, self.net(x, t, cond if self.label_dim else None, force_fp32=True).float())
        r1, r0 = (a.float() for a in self.pair(x, t, cond, released=True))
        rel = max(float((r1 - d1).norm() / d1.norm()), float((r0 - d0).norm() / d0.norm()))
        assert rel < 2e-2, f"release precision differs from float32 by {rel:.2e}"
        msg = [f"float32 vs release precision {rel:.1e}", self.mse_check(x0, cond)]
        if self.state_dtype == torch.float32:
            gen = _edm.import_file("vmg_edm2_generate_images", _edm.src("edm2") / "generate_images.py")
            pub = self.spec.published_w
            w = float(pub) if np.isfinite(pub) else 2.0
            noise, labels = self.initial([0, 1])
            theirs = gen.edm_sampler(self.net, noise, labels, gnet=self.gnet, guidance=w)
            ours = self.heun(noise, labels, np.full(self.steps, w))
            assert torch.equal(theirs, ours), f"sampler differs from edm_sampler by {(theirs - ours).abs().max()}"
            msg.append(f"Heun sampler at w {w:g} bitwise edm_sampler")
        print(f"{self.spec.name} smoke OK: " + "; ".join(msg))


def _pair(key, name, net, guide, dataset, published, note, n=None, batch=None, reference=None, **extra):
    """One registry entry: a subclass named after the key (dashes -> underscores)."""
    n = n or _N[dataset]
    res = {IN512: "512^2 image", IN64: "64^2 image", C10: "32^2 image"}[dataset]
    attrs = dict(net_ckpt=net, guide_ckpt=guide, **extra,
                 spec=Spec(key=key, name=name, modality="image", checkpoint=net.rsplit("/", 1)[1],
                           reference=reference or guide.rsplit("/", 1)[1], sampler=HEUN,
                           steps=extra.get("steps", 32), output=res, published_w=published, published_note=note,
                           datasets={dataset: _split(dataset, n)}, calib_levels=extra.get("steps", 32)))
    if batch:
        attrs["batch"] = batch
    cls = type(key.replace("-", "_"), (EDMPair,), attrs)
    cls.__module__ = __name__
    globals()[cls.__name__] = cls
    return cls


NAN = float("nan")
GEN = "edm2/generate_images.py preset"

# ImageNet-512, NVIDIA's autoguidance presets (generate_images.py = Table 1 of Karras et al. 2024)
_pair("edm2-512-ag-s-xs-fid", "EDM2-S <- XS autoguidance (FID), ImageNet-512",
      posthoc("edm2-img512-s-2147483-0.070"), posthoc("edm2-img512-xs-0134217-0.125"), IN512, 2.10,
      f"{GEN} edm2-img512-s-autog-fid")
_pair("edm2-512-ag-s-xs-dino", "EDM2-S <- XS autoguidance (FD_DINOv2), ImageNet-512",
      posthoc("edm2-img512-s-2147483-0.120"), posthoc("edm2-img512-xs-0134217-0.165"), IN512, 2.45,
      f"{GEN} edm2-img512-s-autog-dino")
_pair("edm2-512-ag-su-xsu-fid", "EDM2-S <- XS unconditional autoguidance (FID), ImageNet-512",
      posthoc("edm2-img512-s-uncond-2147483-0.070"), posthoc("edm2-img512-xs-uncond-0134217-0.110"), IN512, 2.85,
      f"{GEN} edm2-img512-s-uncond-autog-fid")
_pair("edm2-512-ag-su-xsu-dino", "EDM2-S <- XS unconditional autoguidance (FD_DINOv2), ImageNet-512",
      posthoc("edm2-img512-s-uncond-2147483-0.090"), posthoc("edm2-img512-xs-uncond-0134217-0.125"), IN512, 2.90,
      f"{GEN} edm2-img512-s-uncond-autog-dino")
_pair("edm2-512-ag-xxl-m-fid", "EDM2-XXL <- M autoguidance (FID), ImageNet-512",
      posthoc("edm2-img512-xxl-0939524-0.075"), posthoc("edm2-img512-m-0268435-0.155"), IN512, 2.05,
      f"{GEN} edm2-img512-xxl-autog-fid", batch=32)
_pair("edm2-512-ag-xxl-m-dino", "EDM2-XXL <- M autoguidance (FD_DINOv2), ImageNet-512",
      posthoc("edm2-img512-xxl-0939524-0.130"), posthoc("edm2-img512-m-0268435-0.205"), IN512, 2.30,
      f"{GEN} edm2-img512-xxl-autog-dino", batch=32)
# ImageNet-512, EDM2-S classifier-free guidance (generate_images.py edm2-img512-s-guid-fid and -dino)
_pair("edm2-512", "EDM2-S <- XS-uncond CFG (FID), ImageNet-512",
      posthoc("edm2-img512-s-2147483-0.025"), posthoc("edm2-img512-xs-uncond-2147483-0.025"), IN512, 1.40,
      f"{GEN} edm2-img512-s-guid-fid")
_pair("edm2-512-cfg-s-xsu-dino", "EDM2-S <- XS-uncond CFG (FD_DINOv2), ImageNet-512",
      posthoc("edm2-img512-s-2147483-0.085"), posthoc("edm2-img512-xs-uncond-2147483-0.085"), IN512, 1.90,
      f"{GEN} edm2-img512-s-guid-dino")
# ImageNet-512, EDM2-XXL classifier-free guidance (generate_images.py edm2-img512-xxl-guid-fid, FID 1.81)
_pair("edm2-512-cfg-xxl-xsu-fid", "EDM2-XXL <- XS-uncond CFG (FID), ImageNet-512",
      posthoc("edm2-img512-xxl-0939524-0.015"), posthoc("edm2-img512-xs-uncond-2147483-0.015"), IN512, 1.20,
      f"{GEN} edm2-img512-xxl-guid-fid", batch=32)

# ImageNet-64, NVIDIA's two autoguidance presets
S64 = posthoc("edm2-img64-s-1073741-0.045")
_pair("edm2-64-ag-s-xs-fid", "EDM2-S <- XS autoguidance (FID), ImageNet-64",
      S64, posthoc("edm2-img64-xs-0134217-0.110"), IN64, 1.70, f"{GEN} edm2-img64-s-autog-fid", batch=128)
_pair("edm2-64-ag-s-xs-dino", "EDM2-S <- XS autoguidance (FD_DINOv2), ImageNet-64",
      posthoc("edm2-img64-s-1073741-0.105"), posthoc("edm2-img64-xs-0134217-0.175"), IN64, 2.20,
      f"{GEN} edm2-img64-s-autog-dino", batch=128)
# the guide-length sweep: the FID preset's S guided by one EDM2-XS run at six training lengths, on all 50,000
# val images. The 0134217 guide is the preset's own (a 0.110 reconstruction); the others are raw 0.100 snapshots.
for _kimg in ("0033554", "0067108", "0134217", "0268435", "0536870", "2147483"):
    _g = posthoc("edm2-img64-xs-0134217-0.110") if _kimg == "0134217" else raw(f"edm2-img64-xs-{_kimg}-0.100")
    _pair(f"edm2-64-ag-s-xs-{_kimg}", f"EDM2-S <- XS at {int(_kimg) / 1024 ** 2:.0f}Mi images, ImageNet-64",
          S64, _g, IN64, 1.70 if _kimg == "0134217" else NAN,
          f"{GEN} edm2-img64-s-autog-fid" if _kimg == "0134217" else "no published weight for this guide",
          n=imagenet.N_VAL, batch=128)
# ImageNet-64, NVIDIA's classifier-free guidance (Karras et al. 2024, Table 1, FID row): EDM2-S guided by the
# unconditional EDM2-XS, the only unconditional 64 px run NVIDIA trained, both at post-hoc EMA 0.030. Neither
# EMA is released as a reconstruction, so both are rebuilt from the raw snapshots (_edm.rebuilt).
_pair("edm2-64-cfg-s-xsu-fid", "EDM2-S <- XS-uncond CFG (FID), ImageNet-64",
      rebuilt("edm2-img64-s-1073741-0.030"), rebuilt("edm2-img64-xs-uncond-2147483-0.030"), IN64, 1.15,
      "Karras et al. 2024 Table 1, ImageNet-64 CFG (FID 1.48)", batch=128)
# ImageNet-64 plain CFG: conditional XS guided by unconditional XS, same length and EMA (not an NVIDIA preset)
_pair("edm2-64-xs-cfg", "EDM2-XS <- XS-uncond CFG, ImageNet-64",
      raw("edm2-img64-xs-2147483-0.050"), raw("edm2-img64-xs-uncond-2147483-0.050"), IN64, NAN,
      "no published weight (NVIDIA tuned none for this pair)", batch=128)

# CIFAR-10, EDM config F: class-conditional VP guided by the unconditional VP (EDM samples unguided, w = 1)
_pair("cifar10-edm-vp", "EDM VP cond <- uncond CFG, CIFAR-10",
      edm1("edm-cifar10-32x32-cond-vp"), edm1("edm-cifar10-32x32-uncond-vp"), C10, NAN,
      "no published weight: EDM's sampler has no guidance (FID 1.79 at w = 1)", batch=256,
      steps=18, state_dtype=torch.float64)


class GenericPair(EDMPair):
    """Two EDM / EDM2 pickles named on the command line (compute_w.py --guidance cfg-pair | autoguidance).

    cfg-pair: D1 a class-conditional network, D0 a separately trained unconditional one (EDM2's CFG).
    autoguidance: D0 a weaker model of the same family with the same conditioning (Karras et al. 2024).
    auto: whichever of the two the networks' label spaces say (an unconditional reference: cfg-pair)."""

    kind = "autoguidance"

    def load(self):
        fresh = self.net is None
        super().load()
        if fresh:
            l1, l0 = int(self.net.label_dim), int(self.gnet.label_dim)
            if self.kind == "auto":
                self.kind = "cfg-pair" if l1 > 0 and l0 == 0 else "autoguidance"
                print(f"guidance: {self.kind} (label_dim {l1} for D1, {l0} for D0)")
            if self.kind == "cfg-pair":
                assert l1 > 0 and l0 == 0, (f"cfg-pair needs a conditional --model and an unconditional --reference "
                                            f"(label_dim {l1} and {l0}); two conditional models are autoguidance")
            else:
                assert l0 == l1, f"autoguidance needs both models on the same labels (label_dim {l1} vs {l0})"
            assert (self.res, self.channels) == (int(self.gnet.img_resolution), int(self.gnet.img_channels)), \
                "the two networks denoise different spaces"


def generic_pair(net, guide, dataset, kind, key, steps=32, published_w=NAN, batch=None):
    """A GenericPair class for two pickles (paths relative to $VMG_WEIGHTS, or absolute) on `dataset`."""
    if folder.is_folder(dataset):
        sp = folder.split(dataset)
    else:
        assert dataset in _N, f"EDM pairs calibrate on {', '.join(_N)} or folder:<dir>, not {dataset!r}"
        sp = _split(dataset, _N[dataset])
    attrs = dict(net_ckpt=str(net), guide_ckpt=str(guide), kind=kind, steps=int(steps),
                 spec=Spec(key=key, name=f"{Path(str(net)).stem} <- {Path(str(guide)).stem} ({kind})", modality="image",
                           checkpoint=str(net), reference=str(guide), sampler=HEUN, steps=int(steps), output="image",
                           published_w=published_w, published_note="given on the command line" if np.isfinite(published_w)
                           else "no published weight given", datasets={dataset: sp}, calib_levels=int(steps)))
    if batch:
        attrs["batch"] = batch
    return type("GenericPair_" + key.replace("-", "_"), (GenericPair,), attrs)


def checkpoints():
    return sorted({c for o in list(globals().values()) if isinstance(o, type) and issubclass(o, EDMPair)
                   and hasattr(o, "net_ckpt") for c in (o.net_ckpt, o.guide_ckpt)})   # not GenericPair
