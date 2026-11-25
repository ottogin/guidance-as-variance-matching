"""ImageNet-256 class-CFG models as EDM denoisers: DiT-XL/2, SiT-XL/2, U-ViT-L/2, U-ViT-H/2, MaskDiT and REPA's
SiT-XL/2 (Fig 2's class-CFG group).

Each trains with label dropout, so CFG guides D(x; sigma, c) with D(x; sigma, null) from ONE network; both
branches share the weights. All six denoise sd-vae-ft-mse latents times 0.18215 (4 x 32 x 32). Every model is
mapped onto EDM's x = x0 + sigma eps and evaluated on the levels of EDM's 32-step Heun grid, as the passes
behind Fig 2 did; each model's own parameterisation:

  dit      DDPM, linear betas 1e-4..0.02 over 1000 steps, eps-prediction (the learned-variance half dropped).
           x_t = x / sqrt(1 + sigma^2) at the fractional DDPM step whose sigma matches (log-sigma
           interpolation, clamped to [0, 999]). Null class 1000.
  sit      linear interpolant, data at t = 1: t = 1 / (1 + sigma), x_t = t x, x0 = x_t + (1 - t) v. Null 1000.
  uvit-l   continuous VP-SDE (beta 0.1..20), eps-prediction, fed as 999 t with alpha_bar(t) = 1 / (1 + sigma^2).
  uvit-h   the discrete Stable Diffusion schedule, eps-prediction, fed as the step n in [1, 1000]; it never
           trained above sigma = 14.615, so its grid tops out there instead of at 80. Null 1000.
  maskdit  EDM-preconditioned already: D = EDMPrecond(x, sigma, onehot); the null label is all zeros.
  repa     linear interpolant, data at t = 0: t = sigma / (1 + sigma), x_t = x / (1 + sigma), x0 = x_t - t v.

The networks come from each project's repository (vmg/models/_edm REPOS) and run in float32. The published
weights are each model's own, under its own sampler; here every model uses EDM's Heun.
"""
import numpy as np
import torch

from ..data import imagenet
from . import _edm
from .base import Spec, Split
from .edm2 import HEUN, EDMHeun

CLASSES, SCALE = 1000, 0.18215


def _ddpm_sigmas(betas):
    ab = np.cumprod(1 - np.asarray(betas, dtype=np.float64))
    return np.sqrt((1 - ab) / ab)


DIT_SIGMAS = _ddpm_sigmas(np.linspace(1e-4, 0.02, 1000))                               # step t = 0..999
UVIT_H_SIGMAS = _ddpm_sigmas(np.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, 1000) ** 2)     # step n = 1..1000


def _interp_timestep(sigma, table, t0):
    """Fractional timestep whose sigma matches, piecewise linear in log sigma, clamped to the table."""
    xp = torch.as_tensor(np.log(table), device=sigma.device, dtype=torch.float64)
    x = sigma.to(torch.float64).log().clamp(xp[0], xp[-1])
    i = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    return (t0 + (i - 1) + (x - xp[i - 1]) / (xp[i] - xp[i - 1])).to(torch.float32)


def uvit_l_time(sigma):
    """U-ViT-L's input time 999 t, with exp(-(0.1 t + 9.95 t^2)) = 1 / (1 + sigma^2)."""
    lg = torch.log1p(sigma.to(torch.float64) ** 2)
    return (999 * (torch.sqrt(0.01 + 39.8 * lg) - 0.1) / 19.9).to(torch.float32)


def denoise(kind, net, x, sigma, y):
    """D(x; sigma, y) for one family. y: int labels [B], with CLASSES meaning the null class."""
    x = x.to(torch.float32)
    sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(len(x))
    s = sigma.reshape(-1, 1, 1, 1)
    if kind == "maskdit":                                  # one-hot labels, all zeros for the null class
        oh = torch.zeros(len(x), CLASSES + 1, device=x.device).scatter_(1, y[:, None], 1.0)[:, :CLASSES]
        return net(x, sigma, oh)["x"]
    if kind == "sit":
        t = 1 / (1 + sigma)
        xt = x * t.reshape(-1, 1, 1, 1)
        return xt + (1 - t).reshape(-1, 1, 1, 1) * net(xt, t, y)[:, :x.shape[1]]
    if kind == "repa":
        t = sigma / (1 + sigma)
        xt = x / (1 + s)
        return xt - t.reshape(-1, 1, 1, 1) * net(xt, t, y)[0]
    if kind == "dit":
        t = _interp_timestep(sigma, DIT_SIGMAS, 0)
    elif kind == "uvit-h":
        t = _interp_timestep(sigma, UVIT_H_SIGMAS, 1)
    else:
        assert kind == "uvit-l", kind
        t = uvit_l_time(sigma)
    return x - s * net(x / (1 + s ** 2).sqrt(), t, y)[:, :4]


def _state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("ema", "model", "state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return sd


def build(kind, ckpt):
    """The network of one family on the CPU, its class from the project's own repository."""
    if kind == "dit":
        net = _edm.import_file("vmg_dit_models", _edm.src("DiT") / "models.py").DiT_XL_2(input_size=32,
                                                                                         num_classes=CLASSES)
    elif kind == "sit":
        net = _edm.import_file("vmg_sit_models", _edm.src("SiT") / "models.py").SiT_XL_2(input_size=32,
                                                                                         num_classes=CLASSES)
        miss = net.load_state_dict(_state(ckpt), strict=False)
        assert not [k for k in miss.missing_keys if "pos_embed" not in k], miss.missing_keys[:6]
        return net
    elif kind in ("uvit-l", "uvit-h"):
        _edm.on_path(_edm.src("U-ViT"))
        from libs.uvit import UViT
        common = dict(img_size=32, patch_size=2, in_chans=4, mlp_ratio=4, qkv_bias=False, mlp_time_embed=False,
                      num_classes=CLASSES + 1, use_checkpoint=False)
        net = (UViT(embed_dim=1024, depth=20, num_heads=16, **common) if kind == "uvit-l"
               else UViT(embed_dim=1152, depth=28, num_heads=16, conv=False, **common))
    elif kind == "repa":
        net = _edm.import_file("vmg_repa_sit", _edm.src("REPA") / "models" / "sit.py").SiT_XL_2(
            input_size=32, num_classes=CLASSES, use_cfg=True, z_dims=[768], encoder_depth=8, qk_norm=False,
            fused_attn=True)
    else:
        assert kind == "maskdit", kind
        net = _edm.import_file("vmg_maskdit", _edm.src("MaskDiT") / "models" / "maskdit.py").EDMPrecond(
            img_resolution=32, img_channels=4, num_classes=CLASSES, model_type="DiT-XL/2", use_decoder=True,
            mae_loss_coef=0.1, pad_cls_token=False)
    net.load_state_dict(_state(ckpt), strict=True)
    return net


IN256 = "imagenet256"
SPLIT = {IN256: Split(calib="ImageNet val, 10 per class (10,000)", n_calib=10000,
                      eval="seeds 0-9,999, a random class each", n_eval=10000,
                      reference="10,000 ImageNet val images", seed="per seed")}


class CFG256(EDMHeun):
    """One ImageNet-256 network and its null class. Subclasses set spec, kind, ckpt."""

    batch = 128
    sample_batch = 64
    label_dim, res, channels = CLASSES, 32, 4

    def __init__(self):
        self.net = self.vae = None
        if self.kind == "uvit-h":
            self.sigma_max = float(UVIT_H_SIGMAS[-1])

    def load(self):
        if self.net is None:
            self.net = build(self.kind, _edm.weight(self.ckpt)).to("cuda").eval().requires_grad_(False)

    def rows(self, dataset=IN256, split="calib"):
        sp = self.spec.datasets[dataset]
        return imagenet.calib_rows(sp.n_calib) if split == "calib" else list(range(sp.n_eval))

    def data(self, dataset, rows):
        x0 = imagenet.latents(256, rows, [SCALE] * 4, [0.0] * 4, self.seed)
        return x0.cuda(), imagenet.class_of(rows, 256).cuda()

    def pair(self, x, t, labels, released=False, ref=True):
        y = labels.argmax(1) if labels.dim() == 2 else labels           # the sampler passes one-hot labels
        d1 = denoise(self.kind, self.net, x, t, y)
        d0 = denoise(self.kind, self.net, x, t, torch.full_like(y, CLASSES)) if ref else None
        return d1, d0

    def decode(self, x):
        """sd-vae-ft-mse at latents / 0.18215, rounded as DiT's sample.py saves its images."""
        if self.vae is None:
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(_edm.hub.path(_edm.SD_VAE)).to("cuda").eval()
        with torch.no_grad():
            im = torch.cat([self.vae.decode(b / SCALE).sample for b in x.float().split(16)])
        return (im * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    @torch.no_grad()
    def smoke(self):
        """(1) the time conversion lands on the model's own grid; (2) the denoising-MSE sanity check;
        (3) DiT only: D1 equals pred_xstart of DiT's own diffusion code at its integer steps."""
        self.load()
        msg = []
        if self.kind in ("dit", "uvit-h"):
            table, t0 = (DIT_SIGMAS, 0) if self.kind == "dit" else (UVIT_H_SIGMAS, 1)
            got = _interp_timestep(torch.tensor(table[[10, 500, 990]]), table, t0)
            assert torch.allclose(got, torch.tensor([10.0, 500, 990]) + t0, atol=1e-3), got
            msg.append("own discrete steps hit exactly")
        if self.kind == "uvit-l":
            s = torch.tensor([0.05, 1.0, 20.0], dtype=torch.float64)
            tt = uvit_l_time(s).double() / 999
            assert torch.allclose(torch.exp(-(0.1 * tt + 9.95 * tt ** 2)), 1 / (1 + s ** 2), rtol=1e-4)
            msg.append("alpha_bar(t) = 1 / (1 + sigma^2)")
        rows = self.rows(IN256)[:8]
        x0, y = self.data(IN256, rows)
        msg.append(self.mse_check(x0, y))
        if self.kind == "dit":
            _edm.on_path(_edm.src("DiT"))
            from diffusion import create_diffusion
            diff = create_diffusion(timestep_respacing="")
            for i in (100, 600):
                s = float(DIT_SIGMAS[i])
                x = x0 + s * torch.randn(x0.shape, generator=torch.Generator().manual_seed(i)).to(x0.device)
                ours = denoise("dit", self.net, x, torch.tensor(s), y)
                tt = torch.full((len(x),), i, device=x.device, dtype=torch.long)
                theirs = diff.p_mean_variance(self.net, x / (1 + s ** 2) ** 0.5, tt, clip_denoised=False,
                                              model_kwargs=dict(y=y))["pred_xstart"]
                rel = float((ours - theirs).norm() / theirs.norm())
                assert rel < 1e-4, f"DiT step {i}: D differs from its own pred_xstart by {rel:.2e}"
            msg.append("D = DiT's own pred_xstart at steps 100, 600")
        print(f"{self.spec.name} smoke OK: " + "; ".join(msg))


KINDS = {   # key: (kind, checkpoint, name, published w, note)
    "dit-xl2": ("dit", "dit/DiT-XL-2-256x256.pt", "DiT-XL/2", 1.50,
                "DiT's cfg-scale 1.5 (250 DDPM steps; it guides 3 of 4 latent channels)"),
    "sit-xl2": ("sit", "sit/SiT-XL-2-256.pt", "SiT-XL/2", 1.50, "SiT's cfg-scale 1.5 (SDE, 250 steps)"),
    "maskdit": ("maskdit", "maskdit/imagenet256_with_guidance-ema.pt", "MaskDiT", 1.50,
                "MaskDiT's 1.5 (EDM Heun, 40 steps, all 4 channels)"),
    "repa-sit-xl2": ("repa", "repa/repa-sit-xl2-256-4m.pt", "REPA SiT-XL/2", 1.35,
                     "REPA's constant weight (Table 9, FID 1.80); its headline 1.8 applies only for t <= 0.7"),
    "uvit-l2": ("uvit-l", "uvit/imagenet256_uvit_large.pth", "U-ViT-L/2", 1.40,
                "U-ViT's scale 0.4 on eps_c - eps_u, i.e. w = 1.4 (DPM-Solver, 50 steps)"),
    "uvit-h2": ("uvit-h", "uvit/imagenet256_uvit_huge.pth", "U-ViT-H/2", 1.40,
                "U-ViT's scale 0.4 on eps_c - eps_u, i.e. w = 1.4 (DPM-Solver, 50 steps)"),
}
for _key, (_kind, _ckpt, _name, _pub, _note) in KINDS.items():
    _cls = type(_key.replace("-", "_"), (CFG256,), dict(
        kind=_kind, ckpt=_ckpt,
        spec=Spec(key=_key, name=f"{_name} CFG, ImageNet-256", modality="image", checkpoint=_ckpt.rsplit("/", 1)[1],
                  reference="null class (the same network)", sampler=HEUN, steps=32, output="256^2 image",
                  published_w=_pub, published_note=_note, datasets=SPLIT, calib_levels=32)))
    _cls.__module__ = __name__
    globals()[_cls.__name__] = _cls


def checkpoints():
    return sorted(c for _, c, *_ in KINDS.values())
