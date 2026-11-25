"""The paper's audio metrics, each model scored with its own evaluation suite (paper App. impl).

AudioLDM 2 on AudioCaps test (all 4,411 captions): audioldm_eval's passt_replace_panns branch (a2fde54, a month
before the AudioLDM 2 paper), plus independent metrics.
    fad                    VGGish frame embeddings (torchvggish, no PCA, no final ReLU) at 16 kHz; Frechet over frames
    fd_passt, <k-NN>_passt PaSST embeddings (768-d, 32 kHz, 10 s); precision / recall / density / coverage, k = 3
    is_passt               Inception Score of the PaSST logits (10 splits after a RandomState(2020) shuffle)
    kl, kl_sigmoid         per caption: KL(real || generated) of PaSST class probabilities (softmax; sigmoid form)
    clap                   per caption: AudioLDM 2's own CLAP (HTSAT-base, music_speech_audioset_epoch_15_esc_89.98,
                           the CLAP AudioLDM 2 is conditioned on), audio at 32 kHz, first 10 s
    clap_laion             per caption: laion/larger_clap_general, audio at 48 kHz, first 10 s
  The real set is the test clips themselves, each once (884): FAD, FD and the k-NN metrics compare every
  generated caption with that fixed set, and KL pairs a caption with its own clip. Every loader subtracts the
  clip mean (WaveDataset). The own CLAP gives different embeddings on different GPU types for identical audio
  (cosine down to 0.56), so the reference records its GPU and every arm must be scored on the same type.

Stable Audio Open on Song Describer (586 captions without singing): stable-audio-metrics (fd55536), the suite
of the Stable Audio Open paper, with its shipped references.
    fd_openl3              OpenL3 (music, mel256, 512-d per channel, stereo 1024-d, hop 0.5 s) at 44.1 kHz after a
                           -1 dB peak normalisation, against load/openl3_fd/song_describer-nosinging__...batch4.npz
    kl                     per caption: KL against load/passt_kld/song_describer-nosinging__collectmean (PaSST,
                           10 s windows with 5 s overlap, mean logits, softmax)
    clap                   per caption: LAION CLAP 630k-audioset-fusion-best, 48 kHz mono, -1 dB peak, int16 round trip
  The shipped OpenL3 reference is a mean and a covariance, passed as the space's fd_stats.

Beyond fd / prdc / fd_stats the spaces use three options: fixed_ref (the real features are one fixed set, not one
per evaluation row), rows (several features per evaluation row, each one's row position in fake[rows]) and stats
(a statistic of the arm's own features, here the Inception Score).

Repositories and checkpoints: VMG_AUDIO_REPOS holds torchvggish, audioldm_eval (a2fde54) and AudioLDM2 (for the
CLAP model configs that audioldm_eval's branch lacks); VMG_SAO_METRICS is a stable-audio-metrics checkout with
load/clap_score/630k-audioset-fusion-best.pt (see reproduce/envs/README.md).
"""
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import audiocaps, songdescriber

CLAP_REPO, CLAP_FILE = "lukewys/laion_clap", "music_speech_audioset_epoch_15_esc_89.98.pt"
CLAP_LAION = "laion/larger_clap_general"
BATCH = 16


def spaces(model, dataset):
    if model.spec.key == "sao":
        z = np.load(_sam() / "load" / "openl3_fd" / f"{SAO_REF}__channels2__44100__openl3music__openl3hopsize0.5__batch4.npz")
        return {"openl3": dict(fd="fd_openl3", prdc=False, rows="openl3_row",
                               fd_stats=(np.asarray(z["mu_ref"], np.float64), np.asarray(z["sigma_ref"], np.float64)))}
    return {"vggish": dict(fd="fad", prdc=False, rows="vggish_row", fixed_ref=True),
            "passt": dict(fd="fd_passt", prdc=True, fixed_ref=True),
            "passt_logits": dict(fd=None, prdc=False, fixed_ref=True, stats={"is_passt": inception_score})}


def reference(model, dataset, ids):
    return (_sao_reference if model.spec.key == "sao" else _a2_reference)(model, dataset, ids)


def score_arm(model, dataset, ids, src, ref):
    return (_sao_score if model.spec.key == "sao" else _a2_score)(model, dataset, ids, src, ref)


# ------------------------------------------------------------------------------------------------ shared pieces

def inception_score(logits, splits=10, seed=2020):
    """audioldm_eval's calculate_isc on PaSST logits (fewer splits only when there are fewer rows than splits)."""
    f = torch.as_tensor(np.asarray(logits)[np.random.RandomState(seed).permutation(len(logits))]).double()
    p, lp = f.softmax(1), f.log_softmax(1)
    N = len(f)
    splits = min(splits, N)
    sc = []
    for i in range(splits):
        pc, lc = p[i * N // splits:(i + 1) * N // splits], lp[i * N // splits:(i + 1) * N // splits]
        sc.append(float((pc * (lc - pc.mean(0, keepdim=True).log())).sum(1).mean().exp()))
    return float(np.mean(sc))


def kl_rows(gen_logits, real_logits):
    """Per row KL(real || generated), summed over classes: audioldm_eval's calculate_kl before its mean over rows,
    in the softmax form and the sigmoid form."""
    g, r = torch.as_tensor(np.asarray(gen_logits)).float(), torch.as_tensor(np.asarray(real_logits)).float()
    soft = F.kl_div((g.softmax(1) + 1e-6).log(), r.softmax(1), reduction="none").sum(1)
    sig = F.kl_div((g.sigmoid() + 1e-6).log(), r.sigmoid(), reduction="none").sum(1)
    return soft.double().numpy(), sig.double().numpy()


def cos_rows(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)


def read_wav(path):
    """(int16 [channels, n], sample rate) of a PCM16 wav."""
    with wave.open(str(path), "rb") as f:
        assert f.getsampwidth() == 2, f"{path}: not PCM16"
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).reshape(-1, f.getnchannels()).T
        return x.copy(), f.getframerate()


def _device():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def _device_file(model, dataset, ids):
    from .common import ref_path
    return Path(f"{ref_path(model, dataset, ids)}.device")


# ------------------------------------------------------------------------------------------------ AudioLDM 2

def _repos():
    d = os.environ.get("VMG_AUDIO_REPOS")
    assert d and Path(d, "audioldm_eval").is_dir(), "set VMG_AUDIO_REPOS (see reproduce/envs/README.md)"
    return Path(d)


class Detectors:
    """VGGish, PaSST and the two CLAPs, each fed the way audioldm_eval (PaSST branch) feeds it."""

    def __init__(self, device="cuda"):
        import torch.nn as nn
        from hear21passt.base import get_basic_model
        R = _repos()
        self.device = device
        vg = torch.hub.load(str(R / "torchvggish"), "vggish", source="local")
        vg.postprocess = False                                                # use_pca=False
        vg.embeddings = nn.Sequential(*list(vg.embeddings.children())[:-1])   # use_activation=False
        vg.device = device
        self.vggish = vg.to(device).eval()
        self.passt = get_basic_model(mode="all").to(device).eval()            # cat([logits 527, embedding 768])
        if "audioldm_eval" not in sys.modules:       # its package __init__ imports the whole evaluation helper
            import types
            pkg = types.ModuleType("audioldm_eval")
            pkg.__path__ = [str(R / "audioldm_eval" / "audioldm_eval")]
            sys.modules["audioldm_eval"] = pkg
        from audioldm_eval.clap_score.clap.open_clip import factory
        from audioldm_eval.clap_score.model import CLAPAudioEmbeddingClassifierFreev2
        # the checkpoint carries RoBERTa's position_ids buffer, which current transformers no longer persists
        lsd = factory.load_state_dict

        def load_state_dict(path, map_location="cpu", skip_params=True):
            import functools
            load = torch.load              # a 2023 checkpoint holding numpy scalars: torch >= 2.6's weights_only refuses it
            torch.load = functools.partial(load, weights_only=False)
            try:
                sd = lsd(path, map_location, skip_params)
            finally:
                torch.load = load
            sd.pop("text_branch.embeddings.position_ids", None)
            return sd

        factory.load_state_dict = load_state_dict
        # the branch ships the CLAP code without open_clip/model_configs; AudioLDM 2 carries the same configs
        factory.add_model_config(str(R / "AudioLDM2" / "audioldm2" / "clap" / "open_clip" / "model_configs"))
        from huggingface_hub import hf_hub_download
        self.clap = CLAPAudioEmbeddingClassifierFreev2(
            pretrained_path=hf_hub_download(CLAP_REPO, CLAP_FILE), sampling_rate=32000, embed_mode="audio",
            amodel="HTSAT-base", unconditional_prob=0.0).to(device).eval()      # its default blanks 10% at random
        from transformers import ClapModel, ClapProcessor
        self.clapi = ClapModel.from_pretrained(CLAP_LAION, dtype=torch.float32).to(device).eval()
        self.clapi_proc = ClapProcessor.from_pretrained(CLAP_LAION)

    @staticmethod
    def _load(x16, sr, to_sr):
        """WaveDataset.read_from_file: first channel, minus its mean, torchaudio resample, padded to >= 32000."""
        import torchaudio
        a = torch.from_numpy(np.asarray(x16, dtype=np.float32) / 32768.0)[None]
        a = a - a.mean()
        if sr != to_sr:
            a = torchaudio.functional.resample(a, orig_freq=sr, new_freq=to_sr)
        if a.shape[-1] < 32000:
            a = F.pad(a, (0, 32000 - a.shape[-1]))
        return a[0]

    @staticmethod
    def _fix(a, n):
        return a[:n] if a.shape[-1] >= n else F.pad(a, (0, n - a.shape[-1]))

    @torch.no_grad()
    def audio(self, clips, srs):
        """int16 clips (first channel), each at its rate -> vggish frames (list per clip), passt logits / emb, clap,
        clapi embeddings."""
        out = {k: [] for k in ("vggish", "passt_logits", "passt", "clap", "clapi")}
        for b0 in range(0, len(clips), BATCH):
            idx = range(b0, min(b0 + BATCH, len(clips)))
            for i in idx:
                a = self._load(clips[i], srs[i], 16000)
                out["vggish"].append(self.vggish.forward(a.numpy(), 16000).detach().float().cpu().numpy())
            a32 = [self._load(clips[i], srs[i], 32000) for i in idx]
            x = torch.stack([self._fix(a, 320000) for a in a32]).to(self.device)              # PaSST: 10 s
            p = self.passt(x).double().cpu().numpy()
            out["passt_logits"].append(p[:, :527])
            out["passt"].append(p[:, 527:])
            out["clap"].append(self.clap(x[:, None])[:, 0].double().cpu().numpy())
            a48 = [self._fix(self._load(clips[i], srs[i], 48000), 480000).numpy() for i in idx]
            inp = self.clapi_proc(audio=a48, sampling_rate=48000, return_tensors="pt").to(self.device)
            f = self.clapi.get_audio_features(**inp)
            out["clapi"].append(getattr(f, "pooler_output", f).double().cpu().numpy())
        return {k: (v if k == "vggish" else np.concatenate(v)) for k, v in out.items()}

    @torch.no_grad()
    def text(self, captions):
        a, b = [], []
        for b0 in range(0, len(captions), 64):
            cs = list(captions[b0:b0 + 64])
            self.clap.embed_mode = "text"
            a.append(self.clap(cs)[:, 0].double().cpu().numpy())
            self.clap.embed_mode = "audio"
            inp = self.clapi_proc(text=cs, padding=True, return_tensors="pt").to(self.device)
            f = self.clapi.get_text_features(**inp)
            b.append(getattr(f, "pooler_output", f).double().cpu().numpy())
        return np.concatenate(a), np.concatenate(b)


_DET = []


def _det():
    if not _DET:
        _DET.append(Detectors())
    return _DET[0]


def _a2_reference(model, dataset, ids):
    """The rows' own clips, each once, plus the captions' text embeddings and the real clips' CLAP scores."""
    clips = np.asarray(audiocaps.clips("eval", ids))
    uniq = np.unique(clips)
    d = _det()
    got = [audiocaps.wav("eval", int(c)) for c in uniq]
    f = d.audio([g[0] for g in got], [g[1] for g in got])
    print(f"  reference: {len(uniq)} clips of {len(ids)} captions on {_device()}", flush=True)
    t_clap, t_clapi = d.text(audiocaps.captions("eval", ids))
    pos = np.searchsorted(uniq, clips)
    feats = dict(vggish=np.concatenate(f["vggish"]), passt=f["passt"], passt_logits=f["passt_logits"],
                 clip_pos=pos, clap_text=t_clap, clapi_text=t_clapi)
    rows = dict(clap=cos_rows(f["clap"][pos], t_clap), clap_laion=cos_rows(f["clapi"][pos], t_clapi))
    _device_file(model, dataset, ids).write_text(_device())
    return feats, rows


def _a2_score(model, dataset, ids, src, ref):
    real, _ = ref
    dev = _device_file(model, dataset, ids)
    assert not dev.exists() or dev.read_text() == _device(), (
        f"the reference was extracted on {dev.read_text()}, this arm on {_device()}: AudioLDM 2's own CLAP differs "
        f"across GPU types, so score every arm on the reference's type")
    files = [src / f"{i:04d}.wav" for i in ids]
    missing = [p.name for p in files if not p.exists()]
    assert not missing, f"{len(missing)} of {len(files)} samples missing in {src}, e.g. {missing[:3]}"
    got = [read_wav(p) for p in files]
    f = _det().audio([g[0][0] for g in got], [g[1] for g in got])
    pos = np.asarray(real["clip_pos"])
    kl, kl_sig = kl_rows(f["passt_logits"], np.asarray(real["passt_logits"])[pos])
    feats = dict(vggish=np.concatenate(f["vggish"]), passt=f["passt"], passt_logits=f["passt_logits"],
                 vggish_row=np.concatenate([np.full(len(v), j) for j, v in enumerate(f["vggish"])]).astype(np.int64))
    rows = dict(kl=kl, kl_sigmoid=kl_sig, clap=cos_rows(f["clap"], real["clap_text"]),
                clap_laion=cos_rows(f["clapi"], real["clapi_text"]))
    print(f"  {src.name}: {len(ids)} clips on {_device()}", flush=True)
    return feats, rows


# ------------------------------------------------------------------------------------------------ Stable Audio Open

SAO_REF = "song_describer-nosinging"


def _sam():
    d = os.environ.get("VMG_SAO_METRICS")
    assert d and Path(d, "src", "openl3_fd.py").exists(), (
        "set VMG_SAO_METRICS to a stable-audio-metrics checkout (see reproduce/envs/README.md)")
    if d not in sys.path:
        sys.path.insert(0, d)
    return Path(d)


def _sao_reference(model, dataset, ids):
    """Nothing to extract: FD_OpenL3 is against the shipped statistics (spaces), KL against the shipped per-caption
    probabilities (score_arm)."""
    return {}, {}


def openl3_frames(paths, batch=4):
    """Per file [frames, 1024]: stable-audio-metrics' extract_embeddings (channels 2, 44.1 kHz, music, hop 0.5 s,
    batches of 4 files as its reference was made), with each file's embeddings kept apart."""
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")      # TensorFlow next to torch on one GPU
    import librosa
    import openl3
    import pyloudnorm as pyln
    import soxr
    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)
    out = []
    for b0 in range(0, len(paths), batch):
        L, R = [], []
        for p in paths[b0:b0 + batch]:
            audio, sr = librosa.load(str(p), sr=None, mono=False)
            audio = pyln.normalize.peak(audio.T, -1.0)
            audio = soxr.resample(audio, sr, 44100)
            if audio.ndim == 1:
                L.append(audio)
                R.append(audio)
            else:
                L.append(audio[:, 0])
                R.append(audio[:, 1])
        el, _ = openl3.get_audio_embedding(L, [44100] * len(L), model=model, verbose=False, hop_size=0.5, batch_size=batch)
        er, _ = openl3.get_audio_embedding(R, [44100] * len(R), model=model, verbose=False, hop_size=0.5, batch_size=batch)
        out += [np.concatenate([l, r], axis=1) for l, r in zip(el, er)]
        print(f"  openl3 {len(out)}/{len(paths)}", flush=True)
    return out


def _sao_score(model, dataset, ids, src, ref):
    """stable-audio-metrics' three metrics on an arm's 47 s wavs <row>.wav (StableAudioOpen.sample)."""
    import contextlib
    import pickle
    root = _sam()
    files = [src / f"{i:04d}.wav" for i in ids]
    missing = [p.name for p in files if not p.exists()]
    assert not missing, f"{len(missing)} of {len(files)} samples missing in {src}, e.g. {missing[:3]}"
    ev = [songdescriber.eval_row(int(i)) for i in ids]
    import librosa
    import laion_clap
    import pyloudnorm as pyln
    from clap_module.factory import load_state_dict
    from src.clap_score import float32_to_int16, int16_to_float32
    clap = laion_clap.CLAP_Module(enable_fusion=True, device="cuda")
    pkg = load_state_dict(str(root / "load" / "clap_score" / "630k-audioset-fusion-best.pt"))
    pkg.pop("text_branch.embeddings.position_ids", None)
    clap.model.load_state_dict(pkg)
    clap.eval()
    cl = []
    with torch.no_grad():
        temb = torch.cat([clap.get_text_embedding([e["caption"] for e in ev[b:b + 64]], use_tensor=True)
                          for b in range(0, len(ev), 64)])
        for j, p in enumerate(files):
            a, _ = librosa.load(str(p), sr=48000, mono=True)
            a = pyln.normalize.peak(a, -1.0).reshape(1, -1)
            a = torch.from_numpy(int16_to_float32(float32_to_int16(a))).float()
            ae = clap.get_audio_embedding_from_data(x=a, use_tensor=True)
            cl.append(float(F.cosine_similarity(ae, temb[j:j + 1], dim=1, eps=1e-8)[0]))
    del clap
    torch.cuda.empty_cache()
    from hear21passt.base import get_basic_model
    from src.passt_kld import return_probabilities
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        pm = get_basic_model(mode="logits").eval().cuda()
    refp = pickle.load(open(root / "load" / "passt_kld" / f"{SAO_REF}__collectmean__reference_probabilities.pkl", "rb"))
    kl = []
    for e, p in zip(ev, files):
        q = return_probabilities(pm, str(p), collect="mean")
        r = refp[e["caption_id"]]
        kl.append(float(F.kl_div((r + 1e-6).log(), q, reduction="sum", log_target=False)))
    del pm
    torch.cuda.empty_cache()
    fr = openl3_frames(files)
    feats = dict(openl3=np.concatenate(fr), openl3_row=np.concatenate([np.full(len(v), j) for j, v in enumerate(fr)]))
    return feats, dict(clap=np.asarray(cl), kl=np.asarray(kl))
