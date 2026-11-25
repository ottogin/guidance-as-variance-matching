# Data

Every dataset is read through a module in `vmg/data/`, which fixes the paper's row order, calibration and
evaluation splits, and preprocessing. Each lives in its own folder, `$VMG_<NAME>` or by default
`$VMG_DATA/<name>`. Most fetch themselves; a few are manual downloads.

| dataset | used by | module / env var | how to get it | size |
|---|---|---|---|---|
| COCO val2014 | SD 1.5, SD 3.5 M/L (calibration rows 0-5,704, evaluation 5,705-15,704) | `coco`, `VMG_COCO` | `python -m vmg.data.coco fetch` | 6.6 GB |
| MJHQ-30K | FLUX.2 [klein], SD 3.5 M and Sana (Fig. 2) | `mjhq`, `VMG_MJHQ` | `python -m vmg.data.mjhq fetch` (HF `playgroundai/MJHQ-30K`) | 30,000 images at 1024² |
| OpenVid-1M (parts 30-140, chosen clips only) | Wan2.1, CogVideoX | `openvid`, `VMG_OPENVID` | `python -m vmg.data.openvid fetch calib` and `fetch eval` (HTTP range reads out of the remote zips) | the 1,470 chosen clips (280 + 1,190) |
| UCF-101 | Latte-XL/2, SVD | `ucf101`, `VMG_UCF101` | `python -m vmg.data.ucf101 fetch` then `prep` (rebuilds `meta.json`, md5 a026600d) | 6.5 GB |
| Bridge (IRASim split) | Cosmos-Predict2.5 | `bridge`, `VMG_BRIDGE` | `python -m vmg.data.bridge fetch` (resumable; keeps annotations and the val/test videos) | 33 GB download |
| ImageNet ILSVRC2012 val | EDM2 autoguidance and CFG, DiT/SiT/MaskDiT/REPA/U-ViT | `imagenet`, `VMG_IMAGENET`, `VMG_IMAGENET_VAL` | **manual**: download `ILSVRC2012_img_val` from image-net.org and sort it into one folder per synset (its tar is flat; the devkit's ground truth gives each image's synset), then `python -m vmg.data.imagenet prep <64, 256 or 512> --src <val>` (needs `VMG_EDM2`, below; 256/512 encode with a VAE on a GPU). Keep the JPEGs in `$VMG_IMAGENET_VAL` (default `$VMG_IMAGENET/val`): the metrics' held-out reference images are read from them. A tree with only those files works if `$VMG_IMAGENET_VAL_LIST` names the 50,000 relative paths in listing order | 6.4 GB + arrays |
| ImageNet ILSVRC2012 train | the EDM2-XS trainings (fig:teaser (d), fig:cfg-regimes, fig:cross-edm2xs) and the ImageNet-64 images of fig:moments (b), fig:diversity-lx, fig:errfield-lx | `VMG_IMAGENET_TRAIN` | **manual**: download `ILSVRC2012_img_train` from image-net.org (one folder per synset, as its per-class tars unpack). `run_all.py` then runs `reproduce/experiments/spread_in64.py prep` (CPU: `$VMG_IMAGENET/img64_train_u8.npy`, the same crop at 64) and `reproduce/experiments/seedpair_in64.py prep` (the EDM2 training zip, checked against NVIDIA's `img64.pkl`) | 147 GB + a 16 GB array |
| AudioCaps | AudioLDM 2 (validation calibrates, test evaluates) | `audiocaps`, `VMG_AUDIOCAPS` | `python -m vmg.data.audiocaps fetch` (HF mirror `OpenSound/AudioCaps`) | 6 GB |
| Song Describer | Stable Audio Open | `songdescriber`, `VMG_SONGDESCRIBER` | `python -m vmg.data.songdescriber fetch` (Zenodo 10072001, md5-checked; read in place from the zip) | 3.3 GB |
| KIT-ML | MDM | `kitml`, `VMG_MDM_ROOT` | **manual**: clone MDM at ef8edce, run its `prepare/` scripts, and unpack KIT-ML and the `kit_trans_enc_512` checkpoint from their Google Drive links (steps in `vmg/data/kitml.py`) | small |
| DOCCI (test split; 2,000 frozen landscape photos) | Fig. 2's other text-to-image pairs (SD 1.x family, SD 3 / 3.5 Medium, Chroma, Ovis-Image, Z-Image, CogView4, Sana) and the cross-model pairs | `docci`, `VMG_DOCCI` | `python -m vmg.data.docci fetch` (Google Storage; CC BY 4.0) | 17 GB tarball |
| CIFAR-10 | EDM VP CFG (Fig. 2) | `cifar10`, `VMG_CIFAR10` | `python -m vmg.data.cifar10 fetch` (the original python batches) | 170 MB |

## Notes

- **Verify what lands on disk.** Near-full network filesystems can write files short without an error, and
  a file count or an archive CRC will not catch it. The fetchers check sizes where the source states them;
  for staged image sets, open every file once (`PIL.Image.open(p).load()`).
- COCO keeps only photos whose short side is at least 480 px, in one fixed order. The order is cached under
  `$VMG_RUNS/_data/` the first time, and `reproduce/checks/sd35m.py` checks it against the paper's (G1).
- The image metrics need the MJHQ and COCO **evaluation** images too (rows 5,705-15,704), not just the
  calibration ones.
- Model weights come from the Hugging Face Hub by repository id, or from a local mirror `$VMG_WEIGHTS/<org>/<repo>`.
  SD 3 Medium, SD 3.5 Medium and Large, Stable Audio Open and Cosmos-Predict2.5 are gated (accept their licences
  on the Hub, then set `HF_TOKEN`). NVIDIA's EDM2/EDM pickles, the ImageNet-256 checkpoints and NVIDIA's ImageNet-64 / -512
  training-set statistics (`edm2/dataset-refs/img{64,512}.pkl`, the FID and FD_DINOv2 reference of every
  ImageNet table) are fetched by `python -m vmg.models._edm fetch`. Two post-hoc EMAs that NVIDIA used but did not
  release (the ImageNet-64 CFG pair `edm2-64-cfg-s-xsu-fid`) are rebuilt from NVIDIA's raw snapshots by
  `python -m vmg.models._edm rebuild` (30-160 GB of snapshots per training run). CommonCanvas and SD 2.1-base are
  encoded and decoded with SD 1.5's VAE (its encoder is byte-identical to theirs), as the paper's passes were.
- **Code of other projects.** Every EDM2/EDM script and the image metrics need a git clone of NVlabs/edm2 at commit
  4bf8162 in `$VMG_EDM2`; `python -m vmg.models._edm clone` makes it, and the ImageNet-256 projects' clones under
  `$VMG_SRC` (default `$VMG_RUNS/_src`). It must be a git clone: `slurm/train_xs_2e27.sbatch` applies
  `patches/edm2_label_dropout.patch` to a `git archive` of it. MDM, Latte, Cosmos and stable-audio-metrics are cloned
  as their dataset or model rows say.
