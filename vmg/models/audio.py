"""Text-to-audio models: AudioLDM 2 on AudioCaps and Stable Audio Open on Song Describer.

AudioLDM 2 (cvssp/audioldm2 = AudioLDM 2-Full, the 346M T-UNet). DDIM, 200 steps, eta 0, epsilon prediction,
"leading" spacing (t = 996, 991, ..., 1), fp32 as the authors' repo runs, w = 3.5, 10.24 s = a [8, 256, 16]
latent. D0 is the empty prompt through the same conditioning path (CLAP + FLAN-T5 -> GPT-2 "language of audio"
tokens). The pipeline multiplies by its local guidance_scale and has no step callback, so sampling runs a copy
of its loop that changes only the weight per step; the smoke test holds that copy to the pipeline. x0 is the
VAE posterior SAMPLE (x scaling factor) of the authors' log-mel of the real clip (16 kHz, mean removed, peak
0.5, padded to 163,840 samples, Tacotron STFT 1024/160/1024, 64 mels 0-8 kHz, log clamp 1e-5).

Stable Audio Open 1.0 (stable-audio-tools 0.0.19). The model card's call: DPM-Solver++ (3M SDE), 100 steps,
sigma 500 -> 0.3 log-uniform, cfg 7, fp32 with TF32 off. k-diffusion's VDenoiser: x = x0 + sigma eps,
D = c_skip x + c_out v. D0 zeroes the whole cross-attention sequence (T5 tokens and the timing tokens) and keeps
the global timing embedding. x0 is the Oobleck VAE posterior SAMPLE of 47 s of 44.1 kHz stereo, zero-padded to
the 2,097,152-sample window; moments cover the first 1,012 latent frames, which hold only real audio.
"""
import json
from functools import lru_cache

import numpy as np
import torch

from .. import guidance as G
from .. import noise
from ..data import audiocaps, songdescriber
from . import hub
from .base import Model, Spec, Split

# AudioLDM 2's real-audio path: the authors' read_wav_file / wav_to_fbank (audioldm2/utilities/audio)

SR16, N_WAV, N_FRAMES = 16000, 1024 * 160, 1024


def _hz_to_mel(f):
    """librosa.hz_to_mel, Slaney scale (htk=False)."""
    f = np.asanyarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    m = f / f_sp
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-30) / min_log_hz) / logstep, m)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_basis(sr=SR16, n_fft=1024, n_mels=64, fmin=0.0, fmax=8000.0):
    """librosa.filters.mel(sr=, n_fft=, n_mels=, fmin=, fmax=) with its defaults (Slaney scale and norm), float32."""
    w = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    fft = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft)
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w


@lru_cache(maxsize=1)
def _stft_bases(n_fft=1024, win=1024):
    """The Tacotron STFT's windowed Fourier basis (conv1d form) and the mel matrix."""
    from scipy.signal import get_window
    fb = np.fft.fft(np.eye(n_fft))
    cut = n_fft // 2 + 1
    basis = torch.FloatTensor(np.vstack([np.real(fb[:cut]), np.imag(fb[:cut])])[:, None, :])
    w = get_window("hann", win, fftbins=True)                 # win == n_fft: pad_center is a no-op
    basis *= torch.from_numpy(w).float()
    return basis.float(), torch.from_numpy(mel_basis()).float()


def fbank(wav_i16, sr):
    """[1024, 64] log-mel of one int16 clip, exactly as the AudioLDM 2 repo prepares real audio."""
    import torch.nn.functional as F
    import torchaudio
    w = torch.from_numpy(np.asarray(wav_i16, dtype=np.float32) / 32768.0)[None]     # torchaudio.load
    w = torchaudio.functional.resample(w, orig_freq=sr, new_freq=SR16).numpy()[0]
    w = w - np.mean(w)                                                             # normalize_wav
    w = w / (np.max(np.abs(w)) + 1e-8) * 0.5
    if len(w) < N_WAV:                                                             # pad_wav
        p = np.zeros((1, N_WAV))
        p[:, :len(w)] = w
        w = p
    else:
        w = w[None, :N_WAV]
    assert np.max(np.abs(w)) > 0, "silent clip"
    w = 0.5 * (w / np.max(np.abs(w)))
    y = torch.clip(torch.FloatTensor(w[0]).unsqueeze(0), -1, 1)                    # get_mel_from_wav
    basis, mel = _stft_bases()
    y = F.pad(y.view(1, 1, 1, -1), (512, 512, 0, 0), mode="reflect").squeeze(1)
    ft = F.conv1d(y, basis, stride=160, padding=0)
    mag = torch.sqrt(ft[:, :513] ** 2 + ft[:, 513:] ** 2)
    fb = torch.log(torch.clamp(torch.matmul(mel, mag), min=1e-5))[0].T             # [frames, 64]
    n = fb.shape[0]                                                                # _pad_spec
    return F.pad(fb, (0, 0, 0, N_FRAMES - n)) if n < N_FRAMES else fb[:N_FRAMES]


def to_pcm16(x):
    """What soundfile writes for float audio: x 0x7FFF, rounded."""
    return np.rint(np.clip(np.asarray(x, np.float32), -1, 1) * 32767).astype(np.int16)


def write_wav(path, pcm16, sr):
    import os
    import wave
    tmp = f"{path}.{os.getpid()}.tmp"
    with wave.open(tmp, "wb") as f:
        f.setnchannels(1 if pcm16.ndim == 1 else pcm16.shape[0])
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(np.ascontiguousarray(pcm16.T if pcm16.ndim > 1 else pcm16).tobytes())
    os.replace(tmp, path)


class AudioLDM2(Model):
    spec = Spec(key="audioldm2", name="AudioLDM 2", modality="audio", checkpoint="cvssp/audioldm2",
                reference="empty prompt", sampler="DDIM", steps=200, output="10.24 s clip", published_w=3.5,
                datasets={"audiocaps": Split(calib="AudioCaps validation, 2,223 captions of 445 clips", n_calib=2223,
                                             eval="all 4,411 test captions (884 clips)", n_eval=4411,
                                             reference="the test clips", seed="per caption")},
                calib_levels=200)
    batch = 16

    def __init__(self):
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return
        from diffusers import AudioLDM2Pipeline
        from transformers import GPT2LMHeadModel
        path = hub.path(self.spec.checkpoint)
        # fp32 pinned: text_encoder/config.json says float64 and transformers 5 loads the config's dtype. The
        # pipeline names its language model GPT2Model, which in transformers 5 lost the generation helper its LOA
        # loop calls; GPT2LMHeadModel wraps the same weights (the head is unused).
        lm = GPT2LMHeadModel.from_pretrained(path, subfolder="language_model", dtype=torch.float32)
        pipe = AudioLDM2Pipeline.from_pretrained(path, language_model=lm, torch_dtype=torch.float32)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to("cuda")
        self.pipe.scheduler.set_timesteps(self.spec.steps, device="cuda")
        self.ts = self.pipe.scheduler.timesteps.clone()
        self.ac = self.pipe.scheduler.alphas_cumprod.to("cuda", torch.float32)
        up = np.prod(pipe.vocoder.config.upsample_rates) / pipe.vocoder.config.sampling_rate
        secs = pipe.unet.config.sample_size * pipe.vae_scale_factor * up
        self.height, self.keep = int(secs / up), int(secs * pipe.vocoder.config.sampling_rate)
        assert (self.height, self.keep) == (N_FRAMES, N_WAV), (self.height, self.keep)

    def levels(self):
        self.load()
        a = self.ac[self.ts].double().cpu().numpy()
        return dict(sigma=np.sqrt((1 - a) / a), t=self.ts.double().cpu().numpy())

    def rows(self, dataset="audiocaps", split="calib"):
        assert dataset == "audiocaps", dataset
        return audiocaps.rows(split)

    def captions(self, dataset, rows, split="eval"):
        return audiocaps.captions(split, rows)

    @torch.no_grad()
    def condition(self, prompts):
        """Both branches, each prompt encoded ALONE as a one-prompt pipeline call encodes it (batched encoding
        would shift the GPT-2 tokens, which follow the padded T5 sequence), T5 sequences zero-padded to one length
        with mask 0 (the cross-attention gives them weight exactly 0). Uncond rows first, as the pipeline stacks."""
        import torch.nn.functional as F
        pe, am, ge = [], [], []
        for p in prompts:
            e, m, g = self.pipe.encode_prompt(p, "cuda", 1, True)
            pe.append(e)
            am.append(m)
            ge.append(g)
        L = max(e.shape[1] for e in pe)
        pe = [F.pad(e, (0, 0, 0, L - e.shape[1])) for e in pe]
        am = [F.pad(m, (0, L - m.shape[1])) for m in am]
        cat = lambda xs: torch.cat([x[:1] for x in xs] + [x[1:] for x in xs])    # noqa: E731
        return dict(pe=cat(pe), am=cat(am), ge=cat(ge))

    def _eps(self, inp, t, cond):
        return self.pipe.unet(inp, t, encoder_hidden_states=cond["ge"], encoder_hidden_states_1=cond["pe"],
                              encoder_attention_mask_1=cond["am"], return_dict=False)[0]

    @lru_cache(maxsize=512)
    def _fbank(self, clip):
        return fbank(*audiocaps.wav("calib", clip))

    @torch.no_grad()
    def data(self, dataset, rows):
        self.load()
        fb = torch.stack([self._fbank(c) for c in audiocaps.clips("calib", rows)]).cuda()
        d = self.pipe.vae.encode(fb[:, None].float()).latent_dist
        z = torch.stack([torch.randn(d.mean.shape[1:], generator=noise.generator(self.seed, r, 2)) for r in rows])
        x0 = (d.mean + d.std * z.to(d.mean.device)) * self.pipe.vae.config.scaling_factor
        return x0.float(), self.condition(audiocaps.captions("calib", rows))

    def noise(self, x0, k, eps):
        a = self.ac[self.ts[k]]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps.to(x0.device)

    def net_pair(self, x, k, cond):
        """(eps_cond, eps_uncond) from the one batched call the sampler makes, [x, x] with uncond first."""
        eu, ec = self._eps(torch.cat([x, x]), self.ts[k], cond).float().chunk(2)
        return ec, eu

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        a = self.ac[self.ts[k]]
        return tuple((x - (1 - a).sqrt() * e) / a.sqrt() for e in self.net_pair(x, k, cond))

    @torch.no_grad()
    def loop(self, prompts, gens, w_steps, cond=None):
        """The pipeline's denoising loop with the weight of step i from w_steps[i]. One generator per row, so a
        row's noise does not depend on its batch. Returns float32 waveforms [B, 163840] at 16 kHz (cpu)."""
        p = self.pipe
        cond = cond or self.condition(prompts)
        p.scheduler.set_timesteps(self.spec.steps, device="cuda")
        lat = p.prepare_latents(len(prompts), p.unet.config.in_channels, self.height, cond["pe"].dtype, "cuda", gens)
        extra = p.prepare_extra_step_kwargs(gens, 0.0)
        for i, t in enumerate(p.scheduler.timesteps):
            u, c = self._eps(p.scheduler.scale_model_input(torch.cat([lat] * 2), t), t, cond).chunk(2)
            lat = p.scheduler.step(u + float(w_steps[i]) * (c - u), t, lat, **extra).prev_sample
        mel = p.vae.decode(1 / p.vae.config.scaling_factor * lat).sample
        return p.mel_spectrogram_to_waveform(mel)[:, :self.keep].float().cpu()

    def sample(self, dataset, rows, w, outdir, batch=16):
        """A 16 kHz PCM16 wav per test caption; caption r uses initial noise seed 1_000_003 + r in every arm."""
        self.load()
        ws = G.weights(self.spec.published_w if w is None else w, self.levels())
        todo = [r for r in rows if not (outdir / f"{r:04d}.wav").exists()]
        for b0 in range(0, len(todo), batch):
            rb = todo[b0:b0 + batch]
            gens = [torch.Generator("cuda").manual_seed(1_000_003 + int(r)) for r in rb]
            wav = self.loop(audiocaps.captions("eval", rb), gens, ws)
            for r, x in zip(rb, wav):
                write_wav(outdir / f"{r:04d}.wav", to_pcm16(x.numpy()), SR16)

    @torch.no_grad()
    def smoke(self):
        self.load()
        lv = self.levels()
        assert len(lv["t"]) == 200 and lv["t"][0] == 996 and lv["t"][-1] == 1, lv["t"][[0, -1]]
        cap = audiocaps.captions("eval", [0])
        CHECK = (0, 100, 180)

        def trace(run):
            """run() with the unet's inputs and output recorded at the CHECK calls."""
            seen, n = {}, [0]

            def pre(mod, args, kw):
                if n[0] in CHECK:
                    seen[n[0]] = dict(x=args[0].clone(), t=args[1],
                                      kw={k: v.clone() for k, v in kw.items() if torch.is_tensor(v)})

            def post(mod, args, kw, out):
                if n[0] in CHECK:
                    seen[n[0]]["eps"] = out[0].clone()
                n[0] += 1

            h1 = self.pipe.unet.register_forward_pre_hook(pre, with_kwargs=True)
            h2 = self.pipe.unet.register_forward_hook(post, with_kwargs=True)
            try:
                out = run()
            finally:
                h1.remove()
                h2.remove()
            assert n[0] == self.spec.steps, f"{n[0]} unet calls"
            return out, seen

        w = self.spec.published_w
        pipe = lambda: self.pipe(cap[0], num_inference_steps=self.spec.steps, guidance_scale=w,  # noqa: E731
                                 generator=[torch.Generator("cuda").manual_seed(7)], output_type="np").audios
        ref, seen = trace(pipe)
        cond = self.condition(cap)
        kw = seen[0]["kw"]
        for ours, k in (("ge", "encoder_hidden_states"), ("pe", "encoder_hidden_states_1"),
                        ("am", "encoder_attention_mask_1")):
            assert torch.equal(cond[ours], kw[k]), f"conditioning {ours}"
        for i in CHECK:
            assert torch.equal(seen[i]["t"], self.ts[i]), f"timestep at call {i}"
            x = seen[i]["x"][:1]
            assert torch.equal(seen[i]["x"][1:], x), "the two branch inputs differ"
            ec, eu = self.net_pair(x, i, cond)
            ou, oc = seen[i]["eps"].float().chunk(2)
            assert torch.equal(ec, oc) and torch.equal(eu, ou), f"network outputs at call {i}"
        floor = float(np.abs(pipe() - ref).max())                # the pipeline against itself
        ours, mine = trace(lambda: self.loop(cap, [torch.Generator("cuda").manual_seed(7)], np.full(200, w), cond))
        gap = float(np.abs(ours.numpy() - ref).max())
        dx = {i: float((mine[i]["x"] - seen[i]["x"]).abs().max()) for i in CHECK}
        msg = (f"our loop at constant {w} vs the pipeline: waveform max |diff| {gap:.1e}, unet input max |diff| at "
               f"calls {dx}; the pipeline against itself {floor:.1e} (rms {float(np.sqrt((ref ** 2).mean())):.3f})")
        assert dx[0] == 0 and gap <= max(1e-6, 3 * floor), msg
        print(f"{self.spec.name} smoke: timesteps, conditioning and network outputs at calls {CHECK} identical; {msg}")


# Stable Audio Open

SAO_SR, SAO_WIN, SAO_SEC = 44100, 2097152, 47
NREAL = SAO_SEC * SAO_SR // 2048          # 1012 latent frames of real audio (downsampling ratio 2048)
SAO_SMIN, SAO_SMAX, SAO_RHO, SAO_SAMPLER = 0.3, 500.0, 1.0, "dpmpp-3m-sde"


class _Scheduled(torch.nn.Module):
    """The published DiTWrapper with cfg_scale replaced by sched[k] at the k-th network call of a sampler run."""

    def __init__(self, inner, sched, t):
        super().__init__()
        self.inner, self.sched, self.t, self.i = inner, [float(w) for w in sched], t, 0

    def forward(self, x, t, cfg_scale=None, **kw):
        k = self.i
        assert k < len(self.sched), f"sampler made more than {len(self.sched)} network calls"
        assert float((t[0] - self.t[k]).abs()) < 1e-6, f"call {k}: t {float(t[0])} != level t {float(self.t[k])}"
        self.i += 1
        return self.inner(x, t, cfg_scale=self.sched[k], **kw)


class StableAudioOpen(Model):
    spec = Spec(key="sao", name="Stable Audio Open", modality="audio", checkpoint="stabilityai/stable-audio-open-1.0",
                reference="zeroed text conditioning", sampler="DPM-Solver++ (3M SDE)", steps=100,
                output="47 s stereo clip", published_w=7.0,
                datasets={"songdescriber": Split(calib="Song Describer, 356 captions of 260 tracks (708 clips)",
                                                 n_calib=708, eval="586 other captions without singing", n_eval=586,
                                                 reference="their tracks", seed="caption id")},
                calib_levels=100)
    batch = 4

    def __init__(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import os

        import k_diffusion as K
        # generate_diffusion_cond sets exactly these before sampling; the moments use them too
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False
        path = hub.path(self.spec.checkpoint)
        if os.path.isdir(path):              # get_pretrained_model, from a local copy of the repository
            from stable_audio_tools.models.factory import create_model_from_config
            from stable_audio_tools.models.utils import load_ckpt_state_dict
            cfg = json.load(open(os.path.join(path, "model_config.json")))
            model = create_model_from_config(cfg)
            model.load_state_dict(load_ckpt_state_dict(os.path.join(path, "model.safetensors")))
        else:
            from stable_audio_tools import get_pretrained_model
            model, cfg = get_pretrained_model(path)
        self.model, self.cfg = model.to("cuda"), cfg              # as the model card: no .half(), no .eval()
        self.dit = self.model.model.model                         # ConditionedDiffusionModelWrapper -> DiTWrapper -> DiT
        self.dtype = next(self.model.model.parameters()).dtype
        self.vd = K.external.VDenoiser(self.model.model)
        self.sig = K.sampling.get_sigmas_polyexponential(self.spec.steps, SAO_SMIN, SAO_SMAX, SAO_RHO, device="cuda")[:-1]
        self.t = self.vd.sigma_to_t(self.sig)
        assert cfg["sample_rate"] == SAO_SR and cfg["sample_size"] == SAO_WIN and self.model.diffusion_objective == "v"
        assert self.model.pretransform.downsampling_ratio == 2048 and self.model.io_channels == 64

    def levels(self):
        self.load()
        return dict(sigma=self.sig.double().cpu().numpy(), t=self.t.double().cpu().numpy())

    def rows(self, dataset="songdescriber", split="calib"):
        assert dataset == "songdescriber", dataset
        return songdescriber.rows(split)

    def captions(self, dataset, rows, split="eval"):
        return songdescriber.captions(split, rows)

    def cond(self, captions):
        """Conditioning inputs exactly as generate_diffusion_cond builds them."""
        c = [{"prompt": p, "seconds_start": 0, "seconds_total": SAO_SEC} for p in captions]
        ci = self.model.get_conditioning_inputs(self.model.conditioner(c, "cuda"))
        ci = {k: (v.type(self.dtype) if v is not None else v) for k, v in ci.items()}
        assert ci["input_concat_cond"] is None and ci["prepend_cond"] is None
        return ci

    def _v_pair(self, xin, t, cond):
        """(v_cond, v_uncond) from ONE batched DiT pass, built as DiffusionTransformer.forward's CFG branch does."""
        dt = self.dtype
        xin, t = xin.to(dt), t.to(dt)
        ca, ge = cond["cross_attn_cond"].to(dt), cond["global_cond"].to(dt)
        out = self.dit._forward(torch.cat([xin, xin], dim=0), torch.cat([t, t], dim=0),
                                cross_attn_cond=torch.cat([ca, torch.zeros_like(ca)], dim=0),
                                cross_attn_cond_mask=None, mask=None, input_concat_cond=None,
                                global_embed=torch.cat([ge, ge], dim=0), prepend_cond=None, prepend_cond_mask=None,
                                return_info=False)
        return torch.chunk(out, 2, dim=0)

    def _scalings(self, x, k):
        import k_diffusion as K
        sig = self.sig[k] * x.new_ones([x.shape[0]])            # = sigmas[i] * s_in in the sampler
        c_skip, c_out, c_in = [K.utils.append_dims(s, x.ndim) for s in self.vd.get_scalings(sig)]
        return c_skip, c_out, c_in, self.vd.sigma_to_t(sig)

    def noise(self, x0, k, eps):
        return x0 + eps.to(x0.device) * self.sig[k]

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        c_skip, c_out, c_in, t = self._scalings(x, k)
        vc, vu = self._v_pair(x * c_in, t, cond)
        return (vc * c_out + x * c_skip).float(), (vu * c_out + x * c_skip).float()

    def flat(self, x, cond):
        return x[:, :, :NREAL].reshape(x.shape[0], 1, -1), None

    def encode(self, audio, z):
        """Posterior samples as pretransform.encode draws them (vae_sample), with the standard-normal draws z given."""
        pt = self.model.pretransform
        ae = pt.model
        assert ae.pretransform is None and not pt.model_half and not pt.chunked
        out = []
        for i in range(audio.shape[0]):
            mean, scale = ae.encoder(audio[i:i + 1]).chunk(2, dim=1)
            out.append(z[i:i + 1].to(mean.device, mean.dtype) * (torch.nn.functional.softplus(scale) + 1e-4) + mean)
        return torch.cat(out, dim=0) / pt.scale

    @torch.no_grad()
    def data(self, dataset, rows):
        self.load()
        audio = torch.stack([songdescriber.clip("calib", r) for r in rows]).cuda()
        z = torch.stack([torch.randn(64, SAO_WIN // 2048, generator=noise.generator(self.seed, r, 2)) for r in rows])
        x0 = self.encode(audio, z).float()
        return x0, self.cond(songdescriber.captions("calib", rows))

    def run_sampler(self, caption, seed, sched=None, **kw):
        """Latents [1, 64, 1024] from the published call; `sched` (one w per level) swaps cfg_scale per network call.
        (Not named generate(): smoke.py's hook check is for diffusers pipelines, and this one is checked in smoke().)"""
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        args = dict(steps=self.spec.steps, cfg_scale=self.spec.published_w, sample_size=SAO_WIN, sigma_min=SAO_SMIN,
                    sigma_max=SAO_SMAX, sampler_type=SAO_SAMPLER, device="cuda", seed=int(seed), return_latents=True,
                    batch_size=1, conditioning=[{"prompt": caption, "seconds_start": 0, "seconds_total": SAO_SEC}], **kw)
        if sched is None:
            return generate_diffusion_cond(self.model, **args)
        assert len(sched) == self.spec.steps
        inner = self.model.model
        self.model.model = _Scheduled(inner, sched, self.t)
        try:
            out = generate_diffusion_cond(self.model, **args)
            assert self.model.model.i == self.spec.steps, f"{self.model.model.i} network calls"
        finally:
            self.model.model = inner
        return out

    def sample(self, dataset, rows, w, outdir):
        """Per eval caption <row>.npy (latents [64, 1024]) and <row>.wav (decode(): the 47 s stereo wav that
        stable-audio-metrics scores); seed = caption id in every arm."""
        import os
        self.load()
        sched = None if w is None else G.weights(w, self.levels())
        for r in rows:
            p, wav = outdir / f"{r:04d}.npy", outdir / f"{r:04d}.wav"
            if not p.exists():
                ev = songdescriber.eval_row(r)
                lat = self.run_sampler(ev["caption"], seed=int(ev["caption_id"]), sched=sched)
                tmp = f"{p}.{os.getpid()}.tmp.npy"
                np.save(tmp, lat[0].float().cpu().numpy())
                assert np.load(tmp).shape == (64, SAO_WIN // 2048), f"{tmp}: short write?"
                os.replace(tmp, p)
            if not wav.exists():
                self.decode(np.load(p), wav)

    @torch.no_grad()
    def decode(self, lat, path):
        """The model card's post-processing: decode, peak-normalise, int16, trimmed to 47 s; a stereo wav."""
        a = self.model.pretransform.decode(torch.as_tensor(lat)[None].cuda().to(self.dtype))[0].float()
        a = a.div(a.abs().max()).clamp(-1, 1).mul(32767).to(torch.int16).cpu().numpy()[:, :SAO_SEC * SAO_SR]
        write_wav(path, a, SAO_SR)

    @torch.no_grad()
    def smoke(self):
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        self.load()
        w = self.spec.published_w
        ev = songdescriber.eval_row(0)
        keep = (0, 1, 50, 98, 99)
        cap = dict(cb={}, fwd={}, raw={}, sig=[])
        n = [0]

        def hook(mod, args, kwargs, out):
            if n[0] in keep:
                cap["fwd"][n[0]] = (args[0].clone(), args[1].clone(), kwargs["cross_attn_cond"].clone(),
                                    kwargs["global_embed"].clone(), kwargs["cfg_scale"], kwargs.get("scale_phi"),
                                    kwargs.get("negative_cross_attn_cond"), out.clone())
            n[0] += 1

        orig = self.dit._forward

        def rec(*a, **kw):
            o = orig(*a, **kw)
            if n[0] in keep:
                cap["raw"][n[0]] = o.clone()
            return o

        def cb(d):
            cap["sig"].append(d["sigma"].clone())
            if d["i"] in keep:
                cap["cb"][d["i"]] = (d["x"].clone(), d["denoised"].clone())

        h = self.dit.register_forward_hook(hook, with_kwargs=True)
        self.dit._forward = rec
        try:
            lat_pub = generate_diffusion_cond(
                self.model, steps=self.spec.steps, cfg_scale=w, sample_size=SAO_WIN, sigma_min=SAO_SMIN,
                sigma_max=SAO_SMAX, sampler_type=SAO_SAMPLER, device="cuda", seed=0, return_latents=True, callback=cb,
                conditioning=[{"prompt": ev["caption"], "seconds_start": 0, "seconds_total": SAO_SEC}])
        finally:
            h.remove()
            del self.dit._forward
        assert n[0] == self.spec.steps and len(cap["sig"]) == self.spec.steps, (n[0], len(cap["sig"]))
        assert torch.equal(torch.stack(cap["sig"]).float(), self.sig), "sampler sigmas differ from levels()"
        cond = self.cond([ev["caption"]])
        for k in keep:
            x, D = cap["cb"][k]
            fx, ft, fca, fge, fcfg, fphi, fneg, fout = cap["fwd"][k]
            c_skip, c_out, c_in, t = self._scalings(x, k)
            vc, vu = self._v_pair(x * c_in, t, cond)
            vg = vu + (vc - vu) * w
            Dc, Du = self.denoise_pair(x, k, cond)
            chk = dict(t=torch.equal(ft, t.to(ft.dtype)), x_in=torch.equal(fx, x * c_in),
                       cross_attn=torch.equal(fca, cond["cross_attn_cond"]), global_=torch.equal(fge, cond["global_cond"]),
                       halves=torch.equal(torch.cat([vc, vu]), cap["raw"][k]), guided_v=torch.equal(vg, fout),
                       denoised=torch.equal(vg * c_out + x * c_skip, D),
                       pair=torch.equal(Dc, vc * c_out + x * c_skip) and torch.equal(Du, vu * c_out + x * c_skip))
            assert all(chk.values()) and fcfg == w and not fphi and fneg is None, (k, chk, fcfg, fphi)
        lat_sch = self.run_sampler(ev["caption"], seed=0, sched=[w] * self.spec.steps)
        dl = float((lat_sch - lat_pub).abs().max())
        assert dl <= 1e-4 * float(lat_pub.abs().max()), f"scheduled path at constant {w}: max diff {dl:.2e}"
        row = 0
        audio = songdescriber.clip("calib", row)[None].cuda()
        torch.manual_seed(123)
        ref = self.model.pretransform.encode(audio)
        g = torch.Generator("cuda").manual_seed(123)          # the stream randn_like draws after manual_seed(123)
        ours = self.encode(audio, torch.randn(1, 64, SAO_WIN // 2048, generator=g, device="cuda"))
        dv = float((ref - ours).abs().max())
        assert dv <= 1e-4 * float(ref.abs().max()), f"VAE posterior sample vs pretransform.encode: {dv:.2e}"
        print(f"{self.spec.name} smoke: sampler sigmas = levels; at calls {keep} timestep, inputs, conditioning, both "
              f"halves, guided v and denoised identical; scheduled path at constant {w} max diff {dl:.1e} (bitwise "
              f"{torch.equal(lat_sch, lat_pub)}); VAE sample formula max diff {dv:.1e}")
