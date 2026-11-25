# Supplementary material: Guidance as Variance Matching

| | |
|---|---|
| `appendix.pdf` | The appendix the paper cites as Appendix A–F: proofs and derivations (A), experimental setup (B), additional experiments (C), ablations (D), full results and the user study (E), more samples (F). |
| `user-study.html` | The blind pairwise user study (paper, user-study table; Appendix E), as one offline page. |
| `sample_examples/` | Published guidance against ours on every model of the paper: images, video, audio, motion. |
| `code/` | The method, and a script for every figure and table of the paper. |

Everything is self-contained: no page loads anything from the network.

## `user-study.html`

Open it in Chrome, Firefox or Safari. It is the page the raters used, closed and running offline. It takes a few
seconds to load because the images are inside the file.

- **Rate.** A caption and two images made from it with the same initial noise by two of the seven settings, in random
  order. Pick the better image or a tie. Your votes stay in your browser.
- **Leaderboard › Everyone.** The 1,458 votes from 41 people behind the paper's user-study table: Bradley–Terry
  ratings on the Elo scale, with 95% bootstrap intervals over raters. The authors' votes and those of AI raters are
  not included (Appendix E.3 reports one AI rater separately).
- **Leaderboard › Just me.** Ratings from your own votes only.

SD 3.5 Medium, 495 held-out COCO captions, seven settings: the FID-tuned constant (w = 1.6), the
FD<sub>DINOv2</sub>-tuned constant (2.4), our constant (4.14), our schedule w(σ), and three hand-designed schedules
at the released w = 4.5 (interval guidance, linear, TV-CFG). The settings' names appear only on the leaderboard.
To fit the upload, the images are 256 px WebP; raters saw 480 px.

## `sample_examples/`

Open `sample_examples/index.html`. Every model is sampled at its released guidance weight (Pub.) and at our
variance-matched weight (Ours), with the same seed and prompt for both:

- Images: SD 3.5 Medium (all seven study settings and the released w), SD 3.5 Large, SD 1.5, FLUX.2 [klein] 4B.
- Video: Wan2.1 1.3B, CogVideoX-2B, Stable Video Diffusion, Cosmos-Predict 2.5.
- Audio: AudioLDM 2, Stable Audio Open.
- Motion: MDM.

Rows 01–10 of every model are its first ten evaluation prompts, not selected. Image rows after those are the ones
hand-picked for the paper's figures; ★ in each `prompts.md` marks them. `sample_examples/README.md` lists every
model's weights. The files are web encodes (JPEG, H.264, MP3), not the full-resolution originals.

## `code/`

Start at `code/README.md`.

- `compute_w.py` chooses the guidance weight of a guided pair (CFG or autoguidance) from calibration data alone. It
  evaluates the two networks once per input and noise level and samples nothing. `sample.py` samples with that
  weight.
- `reproduce/paper.py` has one function per figure and table of the paper and its appendix. `python
  reproduce/run_all.py --list` prints each one, what it runs and its rough GPU-hours.
- `code/user_study/arena/` is the web app the study ran on. Its anonymised votes are in `code/reproduce/data/`.
- A first check on a CPU: `pip install -e . && python compute_w.py --model toy` fits a 2-D example with exact
  denoisers in seconds. `pytest tests` runs the CPU tests.

Every model and dataset is public; the code downloads the released weights.
