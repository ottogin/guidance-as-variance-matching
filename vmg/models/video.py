"""Video models: Wan2.1-T2V-1.3B and CogVideoX-2B on OpenVid-1M, SVD img2vid and Latte-XL/2 on UCF-101.

Wan2.1 (flow matching, UniPC, 50 steps, shift 3, 81 frames 832 x 480). x_t = (1 - s) x0 + s eps, v = eps - x0,
D = x_t - s v. D0 is Wan's default (Chinese) negative prompt, which the official code substitutes for an empty
one; diffusers combines velocities u + g (c - u), so w = g; the published weight is the code default 5. Latents
are the VAE mean, normalised by the VAE's per-channel latents_mean / latents_std.

CogVideoX-2B (v-prediction, DDIM, 50 steps, 49 frames 720 x 480, fp16). x_t = sqrt(a) x0 + sqrt(1 - a) eps and
D = sqrt(a) x_t - sqrt(1 - a) v; D0 is the empty prompt; published w = 6 (model card), dynamic CFG off. Latents
are a VAE sample (as the pipeline's own encoder path draws), scaled by the VAE's scaling factor.

SVD img2vid (14 frames 1024 x 576, diffusers defaults: 25 Euler steps on Karras sigmas 700 -> 0.002,
v-prediction, fps 7, motion bucket 127, noise augmentation 0.02). x = x0 + sigma eps (EDM); the network sees
x / sqrt(1 + sigma^2) next to the conditioning image's latents, and D = c_skip x + c_out v. D0 zeroes both the
CLIP image embedding and the image latents (the pipeline's unconditional branch). The published weight is per
FRAME, linspace(1, 3, 14), constant over sigma; our schedules are one weight per level for every frame.

Latte-XL/2 (maxin-cn/Latte ucf101.pt, EMA; configs/ucf101/ucf101_sample.yaml: 16 frames 256^2, DDPM with 250
respaced steps of DiT's 1000-step linear schedule, fp16, cfg 7.0 on eps of all 4 latent channels, null class
101). Its code is the Latte repository (VMG_LATTE_REPO, commit f8980c5). x0 = x_t / sqrt(ab) - sigma eps is affine
in eps with the same coefficients for both branches, so the eps weight is the paper's w.
"""
import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .. import guidance as G
from .. import noise, runs
from ..data import openvid, ucf101
from . import hub
from .base import Model, Spec, Split

OPENVID = {"openvid": Split(calib="OpenVid-1M, 256 clips from parts 30-70", n_calib=openvid.N_CALIB,
                            eval="1,024 captions from parts 80-140", n_eval=openvid.N_EVAL,
                            reference="their 1,024 clips", seed="caption i, seed i")}
WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
           "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
           "三条腿，背景人很多，倒着走")


def _capture(module, k):
    calls = []

    def hook(m, args, kw, out):
        if len(calls) < k:
            o = out[0] if isinstance(out, tuple) else out.sample
            calls.append(({n: v.detach().clone() for n, v in kw.items() if torch.is_tensor(v)}, o.detach().clone()))
    return calls, module.register_forward_hook(hook, with_kwargs=True)


def _stop(pipe, i, t, kw):
    pipe._interrupt = True
    return kw


class _OpenVidModel(Model):
    batch = 1

    def __init__(self):
        self.pipe = None
        self.metric_ref = f"video-{self.spec.key}"   # each model's own real clips (its fps, frames and size)

    @staticmethod
    def sample_file(outdir, r):
        return outdir / f"{r:04d}.mp4"

    def rows(self, dataset="openvid", split="calib"):
        assert dataset == "openvid", dataset
        return openvid.rows(split)

    def sample(self, dataset, rows, w, outdir):
        """MP4 per evaluation caption; caption i and initial noise i are shared by every arm."""
        self.load()
        ws = G.weights(self.spec.published_w if w is None else w, self.levels())
        caps = openvid.captions("eval", rows)
        for r, cap in zip(rows, caps):
            path = self.sample_file(outdir, r)
            if path.exists():
                continue
            w0, cb, _ = G.diffusers_hook(ws, self.levels()["t"])
            frames = self.pipe(prompt=cap, height=self.H, width=self.W, num_frames=self.NF,
                               num_inference_steps=self.spec.steps, guidance_scale=G.LiveScale(self.pipe, w0),
                               generator=noise.generator(self.seed, r, 1), output_type="np", callback_on_step_end=cb,
                               **self.pipe_kwargs()).frames[0]
            openvid.save_mp4(frames, path, self.FPS)


class Wan(_OpenVidModel):
    spec = Spec(key="wan", name="Wan2.1 1.3B", modality="video", checkpoint="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                reference="default negative prompt", sampler="UniPC, shift 3", steps=50, output="81 frames, 832x480",
                published_w=5.0, datasets=OPENVID, calib_levels=50)
    H, W, NF, FPS, SHIFT = 480, 832, 81, 16, 3.0

    def load(self):
        if self.pipe is not None:
            return
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
        path = hub.path(self.spec.checkpoint)
        vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(path, vae=vae, torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=self.SHIFT)
        self.pipe = pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.scheduler.set_timesteps(self.spec.steps, device="cuda")
        self.ts = self.pipe.scheduler.timesteps.clone()
        self.sig = self.pipe.scheduler.sigmas[:self.spec.steps].clone().float()
        c = self.pipe.vae.config
        self.lmean = torch.tensor(c.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
        self.lstd = torch.tensor(c.latents_std, device="cuda").view(1, -1, 1, 1, 1)

    def pipe_kwargs(self):
        return dict(negative_prompt=WAN_NEG)

    def levels(self):
        self.load()
        s = self.sig.double().cpu().numpy()
        with np.errstate(divide="ignore"):
            return dict(sigma=np.where(s < 1, s / (1 - s), np.inf), t=self.ts.double().cpu().numpy())

    def text(self, captions):
        pe, ne = self.pipe.encode_prompt(prompt=list(captions), negative_prompt=[WAN_NEG] * len(captions),
                                         do_classifier_free_guidance=True, num_videos_per_prompt=1,
                                         max_sequence_length=512, device="cuda")
        dt = self.pipe.transformer.dtype
        return pe.to(dt), ne.to(dt)

    @torch.no_grad()
    def data(self, dataset, rows):
        self.load()
        vids = torch.stack([openvid.read_clip("calib", r, self.FPS, self.NF, self.H, self.W) for r in rows]).cuda()
        mu = self.pipe.vae.encode(vids).latent_dist.mode()
        return ((mu.float() - self.lmean) / self.lstd).float(), self.text(openvid.captions("calib", rows))

    def noise(self, x0, k, eps):
        s = self.sig[k]
        return (1 - s) * x0 + s * eps.to(x0.device)

    def net_pair(self, x, k, cond):
        pe, ne = cond
        xin = x.to(self.pipe.transformer.dtype)
        t = self.ts[k].expand(x.shape[0])
        run = lambda e: self.pipe.transformer(hidden_states=xin, timestep=t, encoder_hidden_states=e, return_dict=False)[0]
        return run(pe), run(ne)

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        s = self.sig[k]
        return tuple(x.float() - s * v.float() for v in self.net_pair(x, k, cond))

    @torch.no_grad()
    def smoke(self):
        self.load()
        cap = openvid.captions("calib", [0])
        calls, h = _capture(self.pipe.transformer, 4)
        try:
            self.pipe(prompt=cap[0], negative_prompt=WAN_NEG, height=self.H, width=self.W, num_frames=self.NF,
                      num_inference_steps=self.spec.steps, guidance_scale=self.spec.published_w,
                      generator=torch.Generator("cuda").manual_seed(0), output_type="latent", callback_on_step_end=_stop)
        finally:
            h.remove()
        assert len(calls) >= 2, f"{len(calls)} transformer calls"
        (kc, oc), (ku, ou) = calls[:2]
        pe, ne = self.text(cap)
        assert torch.equal(kc["timestep"], self.ts[0].expand(1)), "timestep"
        assert torch.equal(kc["encoder_hidden_states"], pe) and torch.equal(ku["encoder_hidden_states"], ne), "text"
        vc, vu = self.net_pair(kc["hidden_states"].float(), 0, (pe, ne))
        assert torch.equal(vc, oc) and torch.equal(vu, ou), "network outputs at step 0"
        print(f"{self.spec.name} smoke: timestep, both text branches and network outputs identical")


class CogVideoX(_OpenVidModel):
    spec = Spec(key="cogvideox", name="CogVideoX-2B", modality="video", checkpoint="zai-org/CogVideoX-2b",
                reference="empty prompt", sampler="DDIM", steps=50, output="49 frames, 720x480", published_w=6.0,
                datasets=OPENVID, calib_levels=50)
    H, W, NF, FPS = 480, 720, 49, 8

    def load(self):
        if self.pipe is not None:
            return
        from diffusers import CogVideoXPipeline
        self.pipe = CogVideoXPipeline.from_pretrained(hub.path(self.spec.checkpoint), torch_dtype=torch.float16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.scheduler.set_timesteps(self.spec.steps, device="cuda")
        self.ts = self.pipe.scheduler.timesteps.clone()
        self.ac = self.pipe.scheduler.alphas_cumprod.to("cuda", torch.float32)
        self.rot = None
        if self.pipe.transformer.config.use_rotary_positional_embeddings:
            self.rot = self.pipe._prepare_rotary_positional_embeddings(self.H, self.W, (self.NF - 1) // 4 + 1,
                                                                       torch.device("cuda"))

    def pipe_kwargs(self):
        return dict(use_dynamic_cfg=False)

    def levels(self):
        self.load()
        a = self.ac[self.ts].double().cpu().numpy()
        return dict(sigma=np.sqrt((1 - a) / np.maximum(a, 1e-30)), t=self.ts.double().cpu().numpy())

    def text(self, captions):
        return self.pipe.encode_prompt(prompt=list(captions), negative_prompt=[""] * len(captions),
                                       do_classifier_free_guidance=True, num_videos_per_prompt=1,
                                       max_sequence_length=226, device="cuda", dtype=torch.float16)

    @torch.no_grad()
    def data(self, dataset, rows):
        self.load()
        vids = torch.stack([openvid.read_clip("calib", r, self.FPS, self.NF, self.H, self.W) for r in rows]).cuda().half()
        g = noise.generator(self.seed, rows[0], 2)                  # the VAE sample's own randomness, per row
        lat = self.pipe.vae.encode(vids).latent_dist.sample(generator=g)            # [B, C, F, h, w]
        x0 = (lat.permute(0, 2, 1, 3, 4) * self.pipe.vae.config.scaling_factor).float()
        return x0, self.text(openvid.captions("calib", rows))

    def noise(self, x0, k, eps):
        a = self.ac[self.ts[k]]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps.to(x0.device)

    def net_pair(self, x, k, cond):
        """One batched forward, [uncond, cond], exactly as the pipeline stacks it; returns (v_cond, v_uncond)."""
        pe, ne = cond
        xin, b = x.half(), x.shape[0]
        out = self.pipe.transformer(hidden_states=torch.cat([xin, xin]), encoder_hidden_states=torch.cat([ne, pe]),
                                    timestep=self.ts[k].expand(2 * b), image_rotary_emb=self.rot,
                                    return_dict=False)[0].float()
        vu, vc = out.chunk(2)
        return vc, vu

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        a = self.ac[self.ts[k]]
        xs = x.half().float()                    # the sample the DDIM step sees is the fp16 latent
        return tuple(a.sqrt() * xs - (1 - a).sqrt() * v for v in self.net_pair(x, k, cond))

    @torch.no_grad()
    def smoke(self):
        self.load()
        cap = openvid.captions("calib", [0])
        calls, h = _capture(self.pipe.transformer, 2)
        try:
            self.pipe(prompt=cap[0], height=self.H, width=self.W, num_frames=self.NF, num_inference_steps=self.spec.steps,
                      guidance_scale=self.spec.published_w, use_dynamic_cfg=False,
                      generator=torch.Generator("cuda").manual_seed(0), output_type="latent", callback_on_step_end=_stop)
        finally:
            h.remove()
        kw, o = calls[0]
        pe, ne = self.text(cap)
        assert torch.equal(kw["timestep"], self.ts[0].expand(2)), "timestep"
        assert torch.equal(kw["encoder_hidden_states"], torch.cat([ne, pe])), "text"
        vc, vu = self.net_pair(kw["hidden_states"][:1].float(), 0, (pe, ne))
        ou, oc = o.float().chunk(2)
        assert torch.equal(vc, oc) and torch.equal(vu, ou), "network outputs at step 0"
        print(f"{self.spec.name} smoke: timestep, text and network outputs identical")


# --- UCF-101: SVD img2vid and Latte-XL/2 --------------------------------------------------------------------

class _Stop(Exception):
    """Raised from a forward hook to end a pipeline call after the network calls a smoke test needs."""


def _first_call(module):
    """Capture (args, kwargs, output) of the first call of `module`, then stop the pipeline."""
    got = {}

    def hook(m, args, kw, out):
        got.update(args=[a.detach().clone() if torch.is_tensor(a) else a for a in args],
                   kw={n: v.detach().clone() for n, v in kw.items() if torch.is_tensor(v)},
                   out=(out[0] if isinstance(out, tuple) else out.sample).detach().clone())
        raise _Stop
    return got, module.register_forward_hook(hook, with_kwargs=True)


SVD_DATA = {"ucf101": Split(calib="UCF-101, 1,024 clips (SVD rows 2,048-3,071)", n_calib=ucf101.SVD_CALIB,
                            eval="2,048 other clips, each conditioned on its first frame", n_eval=ucf101.SVD_EVAL,
                            reference="those 2,048 clips", seed="per clip")}


class SVD(Model):
    spec = Spec(key="svd", name="SVD", modality="video", checkpoint="stabilityai/stable-video-diffusion-img2vid",
                reference="zeroed image conditioning", sampler="Euler, Karras sigma", steps=25,
                output="14 frames, 1024x576", published_w=float("nan"),
                published_note="per-frame ramp linspace(1, 3, 14), constant over sigma (diffusers default)",
                datasets=SVD_DATA, calib_levels=25)
    batch = 1                        # the conditioning (image embedding, image latents) is per clip
    H, W, NF, FPS, MOTION, AUG, WMIN, WMAX = 576, 1024, 14, 7, 127, 0.02, 1.0, 3.0
    FLOOR = 1.0 + 2 ** -10           # the smallest fp16 above 1: the pipeline's fp16 weight must stay > 1 (CFG on)

    def __init__(self):
        self.pipe = None
        self.metric_ref = "video-svd"

    @staticmethod
    def sample_file(outdir, r):
        return outdir / f"{r:04d}.npz"

    def load(self):
        if self.pipe is not None:
            return
        from diffusers import StableVideoDiffusionPipeline
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(hub.path(self.spec.checkpoint), torch_dtype=torch.float16,
                                                                 variant="fp16").to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self._grid()

    def _grid(self):
        """sig: the scheduler's float32 sigmas, on its own device; ts: its timesteps (0.25 log sigma)."""
        sch = self.pipe.scheduler
        sch.set_timesteps(self.spec.steps, device="cuda")
        self.sig, self.ts = sch.sigmas[:self.spec.steps].clone(), sch.timesteps.clone()

    def levels(self):
        self.load()
        return dict(sigma=self.sig.double().cpu().numpy(), t=self.ts.double().cpu().numpy())

    def rows(self, dataset="ucf101", split="calib"):
        assert dataset == "ucf101", dataset
        return ucf101.rows("svd", split)

    @staticmethod
    def frames(clip):
        """uint8 [14, 180, 320, 3] -> 14 PIL frames at 1024 x 576 (bicubic, the resize diffusers' docs use)."""
        from PIL import Image
        return [Image.fromarray(f).resize((SVD.W, SVD.H), Image.BICUBIC) for f in clip]

    @torch.no_grad()
    def condition(self, img, gen):
        """The pipeline's own conditioning for one image, both branches (uncond first): CLIP image embedding, VAE
        latents of the noise-augmented image, added time ids. `gen` draws the augmentation noise."""
        from diffusers.utils.torch_utils import randn_tensor
        p = self.pipe
        emb = p._encode_image(img, "cuda", 1, True)
        im = p.video_processor.preprocess(img, height=self.H, width=self.W).to("cuda")
        im = im + self.AUG * randn_tensor(im.shape, generator=gen, device=torch.device("cuda"), dtype=im.dtype)
        up = p.vae.dtype == torch.float16 and p.vae.config.force_upcast
        if up:
            p.vae.to(dtype=torch.float32)
        lat = p._encode_vae_image(im, device="cuda", num_videos_per_prompt=1, do_classifier_free_guidance=True).to(emb.dtype)
        if up:
            p.vae.to(dtype=torch.float16)
        lat = lat.unsqueeze(1).repeat(1, self.NF, 1, 1, 1)
        ids = p._get_add_time_ids(self.FPS - 1, self.MOTION, self.AUG, emb.dtype, 1, 1, True).to("cuda")
        return emb, lat, ids

    @torch.no_grad()
    def encode(self, frames, gen):
        """14 PIL frames -> x0 [14, 4, 72, 128]: a posterior sample of the fp32 encoder, times the scaling factor."""
        x = self.pipe.video_processor.preprocess(frames, height=self.H, width=self.W).to("cuda")
        vae, dt = self.pipe.vae, self.pipe.vae.dtype
        vae.to(torch.float32)
        d = vae.encode(x.float()).latent_dist
        z = d.mean + d.std * torch.randn(d.mean.shape, generator=gen).to("cuda")
        vae.to(dt)
        return z * vae.config.scaling_factor

    @torch.no_grad()
    def data(self, dataset, rows):
        self.load()
        assert len(rows) == 1, "SVD calibrates one clip per batch"
        r = rows[0]
        fr = self.frames(ucf101.svd_clip(r))
        cond = self.condition(fr[0], noise.generator(self.seed, r, 2))
        return self.encode(fr, noise.generator(self.seed, r, 3)).unsqueeze(0).float(), cond

    def noise(self, x0, k, eps):
        return x0 + float(self.sig[k]) * eps.to(x0.device)

    def scale(self, x, k):
        """The scheduler's scale_model_input on the fp16 latents: x / sqrt(sigma^2 + 1)."""
        return x.half() / ((self.sig[k] ** 2 + 1) ** 0.5)

    def net(self, xin, k, cond):
        """One batched unet call [uncond, cond] on scaled latents [2, 14, 4, 72, 128], as the pipeline makes it."""
        emb, lat, ids = cond
        return self.pipe.unet(torch.cat([xin, lat], dim=2), self.ts[k], encoder_hidden_states=emb, added_time_ids=ids,
                              return_dict=False)[0]

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        assert x.shape[0] == 1, x.shape
        xx = torch.cat([x, x])
        v = self.net(self.scale(xx, k), k, cond).float()
        s = self.sig[k].to(x.device)
        D = v * (-s / (s ** 2 + 1) ** 0.5) + xx.half().float() / (s ** 2 + 1)     # Euler's step upcasts the sample
        return D[1:], D[:1]

    @torch.no_grad()
    def smoke(self):
        self.load()
        img = self.frames(ucf101.svd_clip(self.rows("ucf101", "calib")[0]))[0]
        z = torch.randn((1, self.NF, 4, self.H // 8, self.W // 8), generator=torch.Generator().manual_seed(0)).half().cuda()
        got, h = _first_call(self.pipe.unet)
        try:
            self.pipe(img, height=self.H, width=self.W, num_frames=self.NF, num_inference_steps=self.spec.steps, fps=self.FPS,
                      motion_bucket_id=self.MOTION, noise_aug_strength=self.AUG, min_guidance_scale=self.WMIN,
                      max_guidance_scale=self.WMAX, generator=torch.Generator("cuda").manual_seed(7), latents=z.clone(),
                      output_type="latent")
        except _Stop:
            pass
        finally:
            h.remove()
        self._grid()
        cond = self.condition(img, torch.Generator("cuda").manual_seed(7))
        emb, lat, ids = cond
        xin = got["args"][0]
        assert torch.equal(got["kw"]["encoder_hidden_states"], emb), "image embedding"
        assert torch.equal(xin[:, :, 4:], lat), "image latents"
        assert torch.equal(got["kw"]["added_time_ids"], ids), "added time ids"
        assert torch.equal(torch.as_tensor(got["args"][1]).float(), self.ts[0].float()), "timestep"
        x_lat = z * self.pipe.scheduler.init_noise_sigma            # what prepare_latents hands the loop
        assert torch.equal(self.scale(torch.cat([x_lat, x_lat]), 0), xin[:, :, :4]), "scaled model input"
        assert torch.equal(self.net(xin[:, :, :4], 0, cond), got["out"]), "network output at step 0"
        print(f"{self.spec.name} smoke: image embedding, image latents, time ids, timestep, scaled input and network "
              f"output identical to the pipeline's first unet call")

    @torch.no_grad()
    def sample(self, dataset, rows, w, outdir):
        """uint8 [14, 576, 1024, 3] per evaluation clip in <row>.npz (lossless: the paper's FVD read uncompressed
        frames), conditioned on its first frame; clip k's generator is shared by every arm. w None is the published
        per-frame ramp; otherwise one weight per level for every frame."""
        self.load()
        lv = self.levels()
        ws = G.weights(w, lv)
        for r in rows:
            path = self.sample_file(outdir, r)
            if path.exists():
                continue
            img = self.frames(ucf101.svd_clip(r)[:1])[0]
            kw = dict(height=self.H, width=self.W, num_frames=self.NF, num_inference_steps=self.spec.steps, fps=self.FPS,
                      motion_bucket_id=self.MOTION, noise_aug_strength=self.AUG, decode_chunk_size=self.NF,
                      generator=noise.generator(self.seed, r, 1), output_type="pt")
            if ws is None:
                kw.update(min_guidance_scale=self.WMIN, max_guidance_scale=self.WMAX)
            else:
                _, _, weight_at = G.diffusers_hook(ws, lv["t"], floor=self.FLOOR)

                def cb(pipe, i, t, ckw):          # the weight of step i + 1, on every frame
                    ts = pipe.scheduler.timesteps
                    pipe._guidance_scale = torch.full_like(pipe._guidance_scale,
                                                           weight_at(ts[i + 1] if i + 1 < len(ts) else ts[-1]))
                    return ckw
                w0 = weight_at(lv["t"][0])
                kw.update(min_guidance_scale=w0, max_guidance_scale=w0, callback_on_step_end=cb)
            fr = self.pipe(img, **kw).frames[0]                                 # [14, 3, H, W] in [0, 1]
            u8 = (fr.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            tmp = outdir / f"{r:04d}.tmp.npz"
            np.savez_compressed(tmp, video=u8)
            os.replace(tmp, path)


class SVDFrame1(SVD):
    """w_MSE on latent frame 1 alone (fig:wmse-sigma): the svd calibration pass with only that frame's coordinates
    kept. Moments only; the paper never samples it."""
    spec = dataclasses.replace(SVD.spec, key="svd-frame1", name="SVD (frame 1)")

    def flat(self, x, cond):
        return x[:, :1].reshape(x.shape[0], 1, -1), None


LATTE_DATA = {"ucf101": Split(calib="UCF-101, 3,320 clips (rows 10,000-13,319)", n_calib=ucf101.N - ucf101.LATTE_EVAL,
                              eval="10,000 class-conditional clips (row k takes clip k's class)", n_eval=ucf101.LATTE_EVAL,
                              reference="all 13,320 clips", seed="per row")}
LATTE_COMMIT = "f8980c5"


def _ddpm_sigmas(betas):
    ab = np.cumprod(1 - np.asarray(betas, dtype=np.float64))
    return np.sqrt((1 - ab) / ab)


DIT_SIGMAS = _ddpm_sigmas(np.linspace(1e-4, 0.02, 1000))          # Latte = DiT's linear schedule, EDM sigma per t


def _latte_code():
    """(models/latte.py, the diffusion package) from the Latte repository, loaded by path under private names:
    the repo's top-level `models`, `datasets` and `utils` would otherwise shadow installed packages."""
    repo = Path(os.environ.get("VMG_LATTE_REPO", Path(os.environ.get("VMG_DATA", runs.REPO / "data")) / "repos" / "Latte"))
    assert (repo / "models" / "latte.py").exists(), (
        f"no Latte code at {repo}: git clone https://github.com/Vchitect/Latte {repo} && git -C {repo} checkout "
        f"{LATTE_COMMIT}, or set VMG_LATTE_REPO")
    mods = []
    for name, path, pkg in (("vmg_latte_model", repo / "models" / "latte.py", None),
                            ("vmg_latte_diffusion", repo / "diffusion" / "__init__.py", [str(repo / "diffusion")])):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=pkg)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        mods.append(sys.modules[name])
    return mods


class _LatteGuide:
    """Latte's forward_with_cfg with the weight looked up from the ORIGINAL timestep the respacing wrapper hands the
    model, recording what it used. A constant table is the published call with that cfg_scale."""

    def __init__(self, net, w_by_t):
        self.net, self.w, self.seen = net, w_by_t, []

    def __call__(self, x, t, y=None, use_fp16=True):
        ti = int(t[0])
        assert bool((t == ti).all())
        self.seen.append((ti, float(self.w[ti])))
        return self.net.forward_with_cfg(x, t, y=y, cfg_scale=float(self.w[ti]), use_fp16=use_fp16)


class Latte(Model):
    spec = Spec(key="latte", name="Latte-XL/2", modality="video", checkpoint="maxin-cn/Latte", reference="null class",
                sampler="DDPM", steps=250, output="16 frames, 256x256", published_w=7.0,
                published_note="configs/ucf101/ucf101_sample.yaml cfg_scale; the paper never states one",
                datasets=LATTE_DATA, calib_levels=51)
    batch = 16
    row_align = 16                   # the VAE sample is per row, but keep the batches identical across shard counts
    FRAMES, SIZE, NULL, CLASSES, SCALE = 16, 256, 101, 101, 0.18215
    SAMPLE_BATCH = 8

    def __init__(self):
        self.net = None
        self._diff = self._lv = None
        self.metric_ref = "video-latte"

    @staticmethod
    def sample_file(outdir, r):
        return outdir / f"{r:05d}.npz"

    def diffusion(self):
        if self._diff is None:
            self._diff = _latte_code()[1].create_diffusion(str(self.spec.steps))
        return self._diff

    def load(self):
        if self.net is not None:
            return
        from diffusers.models import AutoencoderKL
        mod, _ = _latte_code()
        net = mod.Latte_models["Latte-XL/2"](input_size=self.SIZE // 8, num_classes=self.CLASSES, num_frames=self.FRAMES,
                                            learn_sigma=True, extras=2)
        p = hub.path(self.spec.checkpoint)
        if os.path.isdir(p):
            ck_path = Path(p) / "ucf101.pt"
        else:
            from huggingface_hub import hf_hub_download
            ck_path = hf_hub_download(self.spec.checkpoint, "ucf101.pt")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])       # sample.py's find_model
        self.net = net.to("cuda").eval().requires_grad_(False).to(torch.float16)
        self.vae = AutoencoderKL.from_pretrained(p, subfolder="vae").to("cuda").eval().requires_grad_(False)

    def levels(self):
        """All 250 sampler steps, noisiest first: the original timestep of each respaced step and its EDM sigma."""
        if self._lv is None:
            ts = np.asarray(self.diffusion().timestep_map, dtype=np.int64)[::-1].copy()
            self._lv = dict(sigma=DIT_SIGMAS[ts], t=ts.astype(np.float64))
        return self._lv

    def calib_levels(self):
        """Every 5th sampler step plus the last: 51 of 250."""
        return list(np.unique(np.r_[np.arange(0, self.spec.steps, 5), self.spec.steps - 1]).tolist())

    def rows(self, dataset="ucf101", split="calib"):
        assert dataset == "ucf101", dataset
        return ucf101.rows("latte", split)

    def pixels(self, clips_u8):
        """uint8 [B, 16, 240, 320, 3] -> Latte's training transform, [B, 16, 3, 256, 256] in [-1, 1]: ToTensorVideo,
        UCFCenterCropVideo(256) (bilinear scale of the short edge, centre crop with Python round), Normalize(0.5)."""
        import torch.nn.functional as F
        x = torch.as_tensor(clips_u8, device="cuda").permute(0, 1, 4, 2, 3).float() / 255.0
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = F.interpolate(x, scale_factor=self.SIZE / min(x.shape[-2:]), mode="bilinear", align_corners=False)
        h, w = x.shape[-2:]
        i, j = int(round((h - self.SIZE) / 2.0)), int(round((w - self.SIZE) / 2.0))
        x = x[..., i:i + self.SIZE, j:j + self.SIZE]
        return ((x - 0.5) / 0.5).reshape(B, T, 3, self.SIZE, self.SIZE)

    @torch.no_grad()
    def data(self, dataset, rows):
        """x0 [B, 16, 4, 32, 32]: a posterior SAMPLE of the fp32 encoder (as Latte's train.py draws it) x 0.18215."""
        self.load()
        pix = self.pixels(np.stack([ucf101.latte_clip(r) for r in rows]))
        out = []
        for r, p in zip(rows, pix):
            d = self.vae.encode(p.float()).latent_dist
            out.append(d.mean + d.std * torch.randn(d.mean.shape, generator=noise.generator(self.seed, r, 3)).cuda())
        return torch.stack(out) * self.SCALE, torch.as_tensor(ucf101.labels(rows), device="cuda")

    def _sab(self, k):
        s = float(DIT_SIGMAS[int(self.levels()["t"][k])])
        return s, 1.0 / np.sqrt(1.0 + s * s)

    def noise(self, x0, k, eps):
        s, a = self._sab(k)                       # sqrt(ab) = a, sqrt(1 - ab) = s a
        return a * (x0 + s * eps.to(x0.device))

    def net_pair(self, x, k, y):
        """eps of (cond, null) through one batched call cat([x_t, x_t]), labels cat([y, 101]), as forward_with_cfg."""
        t = int(self.levels()["t"][k])
        B = x.shape[0]
        eps = self.net(torch.cat([x, x]).to(torch.float16), torch.full((2 * B,), t, device=x.device, dtype=torch.long),
                       y=torch.cat([y, torch.full_like(y, self.NULL)]), use_fp16=True)[:, :, :4]
        return eps[:B], eps[B:]

    @torch.no_grad()
    def denoise_pair(self, x, k, y):
        s, a = self._sab(k)
        return tuple(x.float() / a - s * e.float() for e in self.net_pair(x, k, y))

    def _loop(self, z, y, w_by_t, noise_fn):
        """sample_ddp.py's DDPM loop (p_mean_variance of the released diffusion, then mean + exp(log var / 2) noise)
        over cat([z, z]) with labels cat([y, 101]); noise_fn(x) draws each step's noise. Returns the cond half."""
        diff = self.diffusion()
        guide = _LatteGuide(self.net, w_by_t)
        img = torch.cat([z, z])
        yy = torch.cat([y, torch.full_like(y, self.NULL)])
        for i in range(diff.num_timesteps)[::-1]:
            t = torch.tensor([i] * img.shape[0], device=img.device)
            out = diff.p_mean_variance(guide, img, t, clip_denoised=False, model_kwargs=dict(y=yy, use_fp16=True))
            nz = (t != 0).float().view(-1, *([1] * (img.ndim - 1)))
            img = out["mean"] + nz * torch.exp(0.5 * out["log_variance"]) * noise_fn(img)
        return img.chunk(2)[0], guide.seen

    @torch.no_grad()
    def decode(self, lat):
        """[B, 16, 4, 32, 32] -> uint8 [B, 16, 256, 256, 3]: the fp16 VAE per frame and sample_ddp.py's rounding."""
        b, f = lat.shape[:2]
        self.vae.to(torch.float16)
        fr = torch.cat([self.vae.decode(c / self.SCALE).sample for c in lat.to(torch.float16).reshape(b * f, 4, 32, 32).split(64)])
        self.vae.to(torch.float32)
        u8 = ((fr.float() * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        return u8.reshape(b, f, 3, self.SIZE, self.SIZE).permute(0, 1, 3, 4, 2).contiguous()

    @torch.no_grad()
    def smoke(self):
        self.load()
        diff = self.diffusion()
        lv = self.levels()
        assert len(lv["t"]) == 250 and lv["t"][0] == 999 and lv["t"][-1] == 0
        for i in (0, 125, 249):          # our sigma grid is the respaced diffusion's own
            ab = float(diff.alphas_cumprod[249 - i])
            assert abs(np.sqrt((1 - ab) / ab) - lv["sigma"][i]) < 1e-9 * max(1.0, lv["sigma"][i]), i
        # 1. our network pair is bitwise the call forward_with_cfg makes, and its combination is the DDPM's pred_xstart
        x0, y = self.data("ucf101", self.rows("ucf101", "calib")[:2])
        worst = 0.0
        for k in (0, 60, 125, 200, 249):
            x = self.noise(x0, k, noise.eps_batch(self.seed, [0, 1], k, x0.shape[1:], device="cuda"))
            calls, fwd = [], self.net.forward
            self.net.forward = lambda *a, **kw: (lambda o: calls.append(o.clone()) or o)(fwd(*a, **kw))
            try:                         # forward_with_cfg calls self.forward directly, so record it there
                guide = _LatteGuide(self.net, np.full(1000, self.spec.published_w))
                out = diff.p_mean_variance(guide, torch.cat([x, x]), torch.full((4,), 249 - k, device="cuda"),
                                           clip_denoised=False,
                                           model_kwargs=dict(y=torch.cat([y, torch.full_like(y, 101)]), use_fp16=True))
            finally:
                del self.net.forward
            assert guide.seen == [(int(lv["t"][k]), self.spec.published_w)], guide.seen
            ec, eu = self.net_pair(x, k, y)
            assert len(calls) == 1 and torch.equal(torch.cat([ec, eu]), calls[0][:, :, :4]), f"network output at step {k}"
            Dc, Du = self.denoise_pair(x, k, y)
            ref = out["pred_xstart"][:2].float()
            worst = max(worst, float((Du + self.spec.published_w * (Dc - Du) - ref).abs().max() / ref.abs().max()))
        assert worst < 2e-2, worst                        # the gap is the fp16 CFG arithmetic inside forward_with_cfg
        # 2. our sampling loop at the published constant is the released p_sample_loop, bitwise
        z = torch.randn((1, 16, 4, 32, 32), generator=torch.Generator().manual_seed(0)).to("cuda", torch.float16)
        yl = torch.tensor([3], device="cuda")
        torch.manual_seed(1)
        ours, seen = self._loop(z, yl, np.full(1000, 7.0), torch.randn_like)
        torch.manual_seed(1)
        ref = diff.p_sample_loop(self.net.forward_with_cfg, (2, 16, 4, 32, 32), torch.cat([z, z]), clip_denoised=False,
                                 model_kwargs=dict(y=torch.cat([yl, torch.full_like(yl, 101)]), cfg_scale=7.0, use_fp16=True),
                                 progress=False, device="cuda").chunk(2)[0]
        assert torch.equal(ours, ref), "sampling loop differs from p_sample_loop"
        assert [t for t, _ in seen] == lv["t"].astype(int).tolist(), "the guide did not see every step in order"
        print(f"{self.spec.name} smoke: sigma grid = the respaced DDPM's; network outputs bitwise forward_with_cfg's, "
              f"guided D within {worst:.1e} of pred_xstart (fp16 CFG); our loop at 7.0 = p_sample_loop bitwise")

    @torch.no_grad()
    def sample(self, dataset, rows, w, outdir):
        """uint8 [16, 256, 256, 3] per evaluation row in <row>.npz (lossless frames, as the paper's FVD read them).
        Row k is class k's label with initial and per-step noise from row k's own generator, so arms are paired."""
        self.load()
        lv = self.levels()
        ws = G.weights(self.spec.published_w if w is None else w, lv)
        w_by_t = np.full(1000, np.nan)
        w_by_t[lv["t"].astype(int)] = ws
        todo = [r for r in rows if not self.sample_file(outdir, r).exists()]
        for b0 in range(0, len(todo), self.SAMPLE_BATCH):
            rb = todo[b0:b0 + self.SAMPLE_BATCH]
            gens = [noise.generator(self.seed, r, 1) for r in rb]
            draw = lambda: torch.stack([torch.randn((16, 4, 32, 32), generator=g) for g in gens])       # noqa: E731
            z = draw().to("cuda", torch.float16)
            noise_fn = lambda x: (lambda n: torch.cat([n, n]))(draw().to(x.device, x.dtype))         # noqa: E731
            lat, _ = self._loop(z, torch.as_tensor(ucf101.labels(rb), device="cuda"), w_by_t, noise_fn)
            for r, v in zip(rb, self.decode(lat).cpu().numpy()):
                tmp = outdir / f"{r:05d}.tmp.npz"
                np.savez_compressed(tmp, video=v)
                os.replace(tmp, self.sample_file(outdir, r))
