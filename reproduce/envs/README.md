# Environments

One Python environment per family of models. Each `*.lock.txt` is `python reproduce/envs/freeze.py` run inside the
cluster environment the paper's pipeline was checked in (2026-11-08), so `pip install -r <env>.lock.txt` rebuilds it on
another machine with the same CUDA (the torch wheels are `+cu124`, Cosmos's `+cu128`). Select one per job with
`reproduce/slurm/job.sbatch --env NAME`, which runs `$PY_NAME` from `reproduce/slurm/site.env`.

| env | lock | for | notes |
|---|---|---|---|
| core | `core.lock.txt` | SD 1.5, SD 3.5 M/L, FLUX.2 [klein], the toys, image metrics, figures | python 3.10 |
| video | `video.lock.txt` | Wan2.1, CogVideoX and their metrics | core + `av`, `imageio-ffmpeg` |
| ucf | `ucf.lock.txt` | SVD, Latte-XL/2 (UCF-101) and their metrics | core + `av`, `imageio`, `imageio-ffmpeg`; Latte also needs its repository (below) |
| cosmos | `cosmos.lock.txt` | Cosmos-Predict2.5 and its metrics | not pip-installable alone: see below |
| audio | `audio.lock.txt` | AudioLDM 2 and its metrics | a venv over core: see below |
| sao | `sao.lock.txt` | Stable Audio Open (sampling, moments) | its own torch / transformers 4.49 / numpy 1.26: see below |
| saomet | `saomet.lock.txt` | Stable Audio Open's metrics | stable-audio-metrics' own pins (torch 2.1, TF 2.13): see below |
| mdm | `mdm.lock.txt` | MDM, sampling and its evaluation | a venv over core (numpy 1.23.5); MDM's repository: see below |

## Latte-XL/2

The model and its DDPM are the Latte repository's code, loaded by path (never installed):

    git clone https://github.com/Vchitect/Latte $VMG_LATTE_REPO && git -C $VMG_LATTE_REPO checkout f8980c5

Its `models/latte.py` needs `timm` and `einops` (both in the ucf lock).

## Cosmos-Predict2.5

Set `VMG_COSMOS` to a folder holding:

    cp25/   git clone https://github.com/nvidia-cosmos/cosmos-predict2.5 cp25 && git -C cp25 checkout a2c298b
            cd cp25 && uv sync --python 3.10 --extra cu128 --frozen --no-dev     # flash-attn is cp310-only
    hf/     HF_HOME with a token whose account accepted the nvidia/Cosmos-Predict2.5-2B license (gated); the
            released checkpoint_db downloads the robot/action-cond EMA weights, tokenizer.pth and Cosmos-Reason1-7B
    bin/    uv (checkpoint_db shells out to `uvx hf download`) and an ffmpeg binary (mediapy reads the videos;
            imageio-ffmpeg's static binary linked as bin/ffmpeg works)

and run vmg with `cp25/.venv/bin/python` (`PY_cosmos`). The repository's own `uv.lock` is the authoritative lock;
`cosmos.lock.txt` records what that sync installed.

## Audio and motion

The two envs layered over core (`audio`, `mdm`) are `uv venv --python <core python> --system-site-packages`, so
their locks list the venv's own packages first; the rest come from core.

**audio** (`PY_audio`): core plus AudioLDM 2's data path and its evaluation suite.

    uv venv --python $CORE_PY --system-site-packages $E
    uv pip install --python $E/bin/python --no-deps torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python $E/bin/python -c <(printf 'numpy==2.2.6\ntorch==2.6.0\n') soundfile pyarrow \
        transformers==5.17.0 hear21passt librosa torchlibrosa ftfy regex h5py resampy braceexpand webdataset wget pandas

(the last four only because audioldm_eval's CLAP code imports its training data loader; uv also installs its own
torch 2.6.0 into the venv for torchvision, which the lock records).

`VMG_AUDIO_REPOS` holds three clones, loaded by path: `torchvggish` (github.com/harritaylor/torchvggish, f3f9096),
`audioldm_eval` (github.com/haoheliu/audioldm_eval, a2fde54 on its passt_replace_panns branch) and `AudioLDM2`
(github.com/haoheliu/AudioLDM2, b5786c5; only its CLAP model configs, which that branch lacks). Checkpoints come from
the hub on first use: lukewys/laion_clap `music_speech_audioset_epoch_15_esc_89.98.pt` (2.35 GB), laion/larger_clap_general
and roberta-base into `HF_HOME`; VGGish and PaSST into `TORCH_HOME`. Set both to writable folders (a read-only home
is not).

**sao** (`PY_sao`): stable-audio-tools 0.0.19 without its training dependencies.

    uv venv --python python3.10 $E
    uv pip install --python $E/bin/python torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python $E/bin/python --no-deps stable-audio-tools==0.0.19
    uv pip install --python $E/bin/python -c <(printf 'torch==2.6.0\ntorchvision==0.21.0\ntorchaudio==2.6.0\nnumpy==1.26.4\n') \
        k-diffusion==0.1.1 alias-free-torch==0.0.6 einops einops-exts PyWavelets safetensors transformers==4.49.0 \
        sentencepiece protobuf torchsde huggingface_hub packaging tqdm numpy==1.26.4 av soundfile setuptools==69.5.1

stabilityai/stable-audio-open-1.0 is gated: `HF_HOME` needs a token whose account accepted its license (or a local
copy under `$VMG_WEIGHTS`); its T5 conditioner loads t5-base from the hub.

**saomet** (`PY_saomet`): the stable-audio-metrics checkout's own `requirements.txt` (`VMG_SAO_METRICS`,
github.com/Stability-AI/stable-audio-metrics, fd55536), with every `==` pin overriding later transitive
constraints (kapre's PyPI metadata now contradicts the repository's TF 2.13.1 / librosa 0.10.1) and `setuptools<70`
(laion-clap and PaSST import pkg_resources):

    uv venv --python python3.10 $E
    sed -E 's#^-e git\+(https://github.com/kkoutini/passt_hear21)@([0-9a-f]+)\#egg=hear21passt#hear21passt @ git+\1@\2#' \
        $VMG_SAO_METRICS/requirements.txt | grep -v '^ipdb\|^ipython\|^jedi\|^matplotlib-inline\|^backcall' > req.txt
    grep -E '^[A-Za-z0-9_.-]+==' req.txt > override.txt
    uv pip install --python $E/bin/python -r req.txt "setuptools<70" --override override.txt

plus the CLAP checkpoint at `$VMG_SAO_METRICS/load/clap_score/630k-audioset-fusion-best.pt`
(huggingface.co/lukewys/laion_clap). Its reference statistics ship in the checkout's `load/`. OpenL3 (TensorFlow)
runs on the CPU here, about 2 h for 586 clips on 8 cores.

**mdm** (`PY_mdm`): a venv over core with numpy 1.23.5 (MDM's SMPL code imports chumpy), spaCy 3.7 and openai's
`clip` (`pip install --no-build-isolation chumpy==0.70` first: its setup.py imports pip); MDM itself is its repository at `VMG_MDM_ROOT` (github.com/GuyTevet/motion-diffusion-model, ef8edce) with
KIT-ML, GloVe, its T2M evaluators and the kit_trans_enc_512 checkpoint (vmg/data/kitml.py lists the steps).
`VMG_CLIP_CACHE` may point at a pre-fetched ViT-B-32.pt.

## Metric detectors

Downloaded on first use into `$VMG_CACHE` (default `$VMG_RUNS/_cache`) and checked by md5: StyleGAN-V's I3D and
UCF-101 C3D torchscripts (vmg/metrics/video.py), EDM2's Inception pickle and DINOv2 (vmg/metrics/image.py, which
also needs `VMG_EDM2`), and CLIP ViT-L/14 from the Hugging Face hub (or `$VMG_WEIGHTS/openai/clip-vit-large-patch14`).
`VMG_CLIP` names another CLIP repository for the image and video CLIP scores, and `TORCH_HUB` the folder of
torch.hub's DINOv2 code (default `$VMG_CACHE/torch_hub`).
