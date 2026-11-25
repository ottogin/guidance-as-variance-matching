"""Cosmos-Predict2.5-2B robot/action-cond on Bridge: the world model (paper tab:impl-models, tab:impl-data).

The checkpoint (nvidia/Cosmos-Predict2.5-2B, robot/action-cond, experiment `..._bridge_13frame_256x320`) maps one
256x320 frame plus 12 robot actions to the next 12 frames; an episode is rolled out autoregressively, each 13-frame
chunk conditioned on the last frame of the previous one, as cosmos_predict2/action_conditioned.py does.

Guidance, from the released code (repo a2c298b, action_conditioned_video2world_rectified_flow_model.py):

    velocity = cond_v + g * (cond_v - uncond_v)

the EXCESS form, so the paper's w = g + 1. Actions condition both branches and conditioning frames are not guided,
so CFG contrasts only the text: prompt "" (D1) against the default negative prompt (D0). The shipped eval config
(assets/action_conditioned/basic/inference_params.json) has g = 0, i.e. w = 1, the published weight; the CLI default
g = 7 is w = 8. Sampler: FlowUniPC, 35 steps, shift 5, x_t = (1 - s) x0 + s eps, v = eps - x0, D = x_t - s v.

The weight enters without editing the released path: `guidance` is passed as an object whose __mul__ supplies the
step's g (Guide) or records cond - uncond (Probe); a Guide holding the published constant runs the same arithmetic
as the float, which smoke() checks bitwise. The moments cover only the 3 generated latent frames of each chunk.

Environment (not pip-installable): VMG_COSMOS is a folder with
    cp25/    git clone https://github.com/nvidia-cosmos/cosmos-predict2.5 at a2c298b, then
             `uv sync --python 3.10 --extra cu128 --frozen --no-dev` (flash-attn is cp310-only); run vmg with cp25/.venv
    hf/      HF_HOME holding a token with the gated nvidia/Cosmos-Predict2.5-2B license accepted (the released
             checkpoint_db downloads the robot/action-cond EMA weights, tokenizer.pth and Cosmos-Reason1-7B into it)
    bin/     uv (checkpoint_db shells out to `uvx hf download`) and an ffmpeg binary (mediapy reads the videos)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .. import guidance as G
from ..data import bridge
from .base import Model, Spec, Split

NEG = ("The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
       "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
       "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
       "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
       "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
       "Overall, the video is of poor quality.")
# the shipped eval config (LFS asset), minus paths and guidance
PUB = dict(name="basic", input_json_sub_folder="annotation/test", resolution="256,320", camera_id=0, start=0, end=100,
           fps_downsample_ratio=1, gripper_scale=1.0, gripper_key="continuous_gripper_state", state_key="state",
           reverse=False, single_chunk=False, start_frame_idx=0, save_fps=20, num_latent_conditional_frames=1,
           action_scaler=20.0, use_quat=False, action_load_fn="cosmos_predict2.action_conditioned.load_default_action_fn",
           negative_prompt=NEG, seed=0, prompt=None)
CHUNK, STEPS, SHIFT = bridge.CHUNK, 35, 5.0
BRIDGE = {"bridge": Split(calib="Bridge validation, 1,024 chunks, teacher-forced", n_calib=bridge.N_CALIB,
                          eval="100 Bridge test episodes", n_eval=bridge.N_EVAL, reference="their recorded frames",
                          seed="released, per chunk")}


class Guide:
    """Cosmos `guidance` per sampler step: cond + g[k] (cond - uncond)."""

    def __init__(self, g_steps):
        self.g = [float(x) for x in g_steps]
        assert len(self.g) == STEPS, len(self.g)
        self.k = -1

    def __mul__(self, other):
        return self.g[self.k] * other

    __rmul__ = __mul__


class Probe:
    """Makes the velocity function return cond_v and keeps d = cond_v - uncond_v."""

    def __init__(self):
        self.k, self.d = -1, None

    def __mul__(self, other):
        self.d = other
        return other * 0.0

    __rmul__ = __mul__


def _install(model):
    """Wrap get_velocity_fn_from_batch so a Guide / Probe knows which step it is on."""
    orig = model.get_velocity_fn_from_batch

    def patched(data_batch, guidance=1.5, is_negative_prompt=False):
        fn = orig(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        if not isinstance(guidance, (Guide, Probe)):
            return fn
        guidance.k = -1

        def vf(noise, x, timestep):
            guidance.k += 1
            return fn(noise, x, timestep)
        return vf

    model.get_velocity_fn_from_batch = patched


class Cosmos(Model):
    spec = Spec(key="cosmos", name="Cosmos-Predict2.5 2B", modality="world", checkpoint="nvidia/Cosmos-Predict2.5-2B",
                reference="default negative prompt", sampler="released sampler (FlowUniPC, shift 5); autoregressive",
                steps=STEPS, output="13-frame chunks, 320x256", published_w=1.0,
                published_note="the shipped eval config's guidance 0; Cosmos writes D1 + g (D1 - D0), so w = g + 1 "
                               "(the CLI default 7 is w = 8)",
                datasets=BRIDGE, calib_levels=STEPS)
    batch = 1

    def __init__(self):
        self.cli = None
        self._ep = (None, None)

    @staticmethod
    def sample_file(outdir, r):
        return outdir / f"{r:03d}.npz"

    # --- the released code ------------------------------------------------------------------------------------
    def _root(self):
        r = Path(os.environ.get("VMG_COSMOS", ""))
        assert (r / "cp25" / "cosmos_predict2").is_dir(), "set VMG_COSMOS (see this module's docstring)"
        return r

    def _env(self):
        r = self._root()
        os.environ.update(HF_HOME=str(r / "hf"), TOKENIZERS_PARALLELISM="false", PATH=f"{r / 'bin'}:{os.environ['PATH']}",
                          UV_CACHE_DIR=str(r / "uvcache"), UV_PYTHON_INSTALL_DIR=str(r / "uvpython"),
                          UV_TOOL_DIR=str(r / "uvtools"), XDG_DATA_HOME=str(r / "xdg"), XDG_CACHE_HOME=str(r / "xdgcache"))
        if str(r / "cp25") not in sys.path:
            sys.path.insert(0, str(r / "cp25"))

    def load(self):
        """Video2WorldInference for the released checkpoint, built the way action_conditioned.py builds it."""
        if self.cli is not None:
            return
        self._env()
        cwd = os.getcwd()
        os.chdir(self._root() / "cp25")              # the experiment config is resolved relative to the repo
        try:
            from cosmos_oss.init import init_environment
            init_environment()
            from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
            from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey, ModelVariant
            ck = MODEL_CHECKPOINTS[ModelKey(variant=ModelVariant.ROBOT_ACTION_COND)]
            self.cli = Video2WorldInference(
                experiment_name=ck.experiment, ckpt_path=ck.s3.uri, s3_credential_path="", context_parallel_size=1,
                config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py")
        finally:
            os.chdir(cwd)
        torch.set_grad_enabled(False)
        _install(self.cli.model)
        m = self.cli.model
        sch = m.sample_scheduler
        sch.set_timesteps(STEPS, device=m.tensor_kwargs["device"], shift=SHIFT,
                          use_kerras_sigma=m.config.use_kerras_sigma_at_inference)
        self.ts, self.sig = sch.timesteps.clone(), sch.sigmas[:STEPS].clone().double().cpu().numpy()

    def args(self, guidance=0):
        self._env()
        from cosmos_predict2.action_conditioned_config import ActionConditionedInferenceArguments
        return ActionConditionedInferenceArguments.model_validate(
            dict(PUB, input_root=str(bridge.root()), save_root=str(Path(os.environ.get("TMPDIR", "/tmp")) / "cosmos"),
                 guidance=guidance))

    def episode(self, path):
        """(actions, first frame, ground-truth video uint8 [T, 256, 320, 3]) through the published loader."""
        if self._ep[0] == path:
            return self._ep[1]
        self._env()
        import mediapy
        from cosmos_predict2.action_conditioned import load_default_action_fn
        a = self.args()
        j = json.load(open(path))
        v = j["videos"][a.camera_id]
        d = load_default_action_fn()(j, os.path.join(str(a.input_root), v["video_path"] if isinstance(v, dict) else v), a)
        gt = d["video_array"]
        h, w = map(int, a.resolution.split(","))
        if gt.shape[1:3] != (h, w):
            gt = np.stack([mediapy.resize_image(f, (h, w)) for f in gt])
        self._ep = (path, (d["actions"], d["initial_frame"], gt))
        return self._ep[1]

    @staticmethod
    def _pad(a):
        return np.concatenate([a, np.zeros((CHUNK - a.shape[0],) + a.shape[1:], a.dtype)], 0) if a.shape[0] < CHUNK else a

    @staticmethod
    def _chunk_input(img):
        """[first frame, 12 zero frames] uint8 (1, 3, 13, H, W), as action_conditioned.py builds it."""
        import torchvision
        t = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
        v = torch.cat([t, torch.zeros_like(t).repeat(CHUNK, 1, 1, 1)], dim=0)
        return (v * 255.0).to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _u8(video):
        v = (video + 1) / 2
        return (torch.clamp(v[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()

    def rollout(self, actions, img, guidance, first_only=False):
        """The published autoregressive loop (seed = chunk start index). Returns the per-chunk uint8 videos."""
        a = self.args()
        chunks = []
        for i in range(a.start_frame_idx, len(actions), CHUNK):
            video = self.cli.generate_vid2world(
                prompt=a.prompt or "", input_path=self._chunk_input(img), action=torch.from_numpy(self._pad(actions[i:i + CHUNK])).float(),
                guidance=guidance, num_video_frames=CHUNK + 1, num_latent_conditional_frames=a.num_latent_conditional_frames,
                resolution=a.resolution, seed=i, negative_prompt=a.negative_prompt, num_steps=STEPS)
            if isinstance(guidance, Guide):
                assert guidance.k == STEPS - 1, guidance.k
            chunks.append(self._u8(video))
            img = chunks[-1][-1]
            if first_only:
                break
        return chunks

    def chunk_batch(self, gt13, ac):
        """The data batch generate_vid2world builds, with the ground-truth chunk as the video; and its clean latent."""
        a = self.args()
        vid = torch.from_numpy(gt13).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        db = self.cli._get_data_batch_input(video=vid, prompt=a.prompt or "", num_conditional_frames=1,
                                            negative_prompt=a.negative_prompt, use_neg_prompt=True,
                                            action=torch.from_numpy(self._pad(ac)).float())
        m = self.cli.model
        m._normalize_video_databatch_inplace(db)
        m._augment_image_dim_inplace(db)
        _, x0, _ = m.get_data_and_condition(db)
        return db, x0.float()

    # --- the Model interface ----------------------------------------------------------------------------------
    def levels(self):
        self.load()
        s = self.sig
        with np.errstate(divide="ignore"):
            return dict(sigma=np.where(s < 1, s / (1 - s), np.inf), t=self.ts.double().cpu().numpy())

    def rows(self, dataset="bridge", split="calib"):
        assert dataset == "bridge", dataset
        return bridge.rows(split)

    @torch.no_grad()
    def data(self, dataset, rows):
        """x0 [1, 16, 4, 32, 40] (the conditioning latent frame and 3 generated ones); cond = (the published velocity
        function with a Probe as its guidance, the Probe, x0)."""
        self.load()
        assert len(rows) == 1, "Cosmos calibrates one chunk per batch"
        ep, i = bridge.calib_chunks()[rows[0]]
        actions, _, gt = self.episode(ep)
        db, x0 = self.chunk_batch(gt[i:i + CHUNK + 1], actions[i:i + CHUNK])
        probe = Probe()
        return x0, (self.cli.model.get_velocity_fn_from_batch(db, probe, is_negative_prompt=True), probe, x0)

    def noise(self, x0, k, eps):
        s = float(self.sig[k])
        return (1 - s) * x0 + s * eps.to(x0.device)

    @torch.no_grad()
    def denoise_pair(self, x, k, cond):
        """(D1, D0) = x - s (cond_v, uncond_v). The velocity function also takes the noise, which it uses only for the
        conditioning frame (v = eps - x0 there), outside the moments; it is recovered from x and x0."""
        vf, probe, x0 = cond
        s = float(self.sig[k])
        probe.k = k - 1                              # the wrapper increments before the call
        vc = vf((x - (1 - s) * x0) / s, x, torch.stack([self.ts[k]]).unsqueeze(0))
        return x - s * vc, x - s * (vc - probe.d)

    def flat(self, x, cond):
        return x[:, :, 1:].reshape(x.shape[0], 1, -1), None          # the generated latent frames only

    @torch.no_grad()
    def smoke(self):
        """(1) the Probe's cond velocity is bitwise the published path at g = 0 and its uncond matches g = -1;
        (2) our rollout at a Guide holding g is bitwise the plain float g, for g = 0 (published) and 7 (CLI default)."""
        self.load()
        m = self.cli.model
        x0, (vf, probe, _) = self.data("bridge", [0])
        ep, i = bridge.calib_chunks()[0]
        actions, img, gt = self.episode(ep)
        db, _ = self.chunk_batch(gt[i:i + CHUNK + 1], actions[i:i + CHUNK])
        vf0 = m.get_velocity_fn_from_batch(db, g0 := Guide([0.0] * STEPS), is_negative_prompt=True)
        vfm = m.get_velocity_fn_from_batch(db, gm := Guide([-1.0] * STEPS), is_negative_prompt=True)
        worst = 0.0
        for k in (0, 17, 34):
            eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(k)).to(x0.device)
            x = self.noise(x0, k, eps)
            D1, D0 = self.denoise_pair(x, k, (vf, probe, x0))
            t, s = torch.stack([self.ts[k]]).unsqueeze(0), float(self.sig[k])
            g0.k = gm.k = k - 1
            e = (x - (1 - s) * x0) / s
            assert torch.equal(D1, x - s * vf0(e, x, t)), f"cond branch at level {k}"
            ref = x - s * vfm(e, x, t)
            worst = max(worst, float((D0 - ref)[:, :, 1:].norm() / ref[:, :, 1:].norm()))
        assert worst < 1e-3, worst                   # cond - (cond - uncond) vs the released uncond: rounding only
        actions, img, _ = self.episode(bridge.eval_episodes()[0])
        for g in (0, 7):
            a = self.rollout(actions, img, g, first_only=True)[0]
            b = self.rollout(actions, img, Guide([g] * STEPS), first_only=True)[0]
            assert np.array_equal(a, b), f"Guide({g}) differs from the plain guidance {g}"
        print(f"{self.spec.name} smoke: cond branch bitwise the released velocity at g=0, uncond within {worst:.1e} of "
              f"g=-1; a Guide at g=0 and g=7 reproduces the plain rollout bitwise")

    @torch.no_grad()
    def sample(self, dataset, rows, w, outdir):
        """<row>.npz per evaluation episode: pred uint8 [T, 256, 320, 3], frame j predicting ground-truth frame j
        (frame 0 is the given frame). w None is the published g = 0; a constant runs as the plain float g = w - 1."""
        self.load()
        for r in rows:
            path = self.sample_file(outdir, r)
            if path.exists():
                continue
            if w is None:
                g = 0
            elif np.ndim(w) == 0:
                g = float(w) - 1.0
            else:
                g = Guide(G.weights(w, self.levels()) - 1.0)
            actions, img, gt = self.episode(bridge.eval_episodes()[r])
            chunks = self.rollout(actions, img, g)
            pred = np.concatenate([chunks[0]] + [c[1:] for c in chunks[1:]], 0)[:len(gt)]
            tmp = outdir / f"{r:03d}.tmp.npz"
            np.savez_compressed(tmp, pred=pred, n_gt=len(gt), episode=os.path.basename(bridge.eval_episodes()[r]))
            os.replace(tmp, path)
