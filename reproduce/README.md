# Reproducing the paper

Every figure and table of the paper is one function in [`paper.py`](paper.py), named after its label
(`fig:wmse-published` → `fig_wmse_published`). A function returns the pipeline jobs the float needs (calibration
passes, sampling, metrics) and the commands that draw it. [`run_all.py`](run_all.py) runs them, sharing the jobs
that several floats need.

```bash
python reproduce/run_all.py --list                         # every float, its jobs, generator and GPU-hours
python reproduce/run_all.py --only tab:objective           # one float from scratch (labels or function names)
python reproduce/run_all.py --budget                       # rough device-hours, shared jobs counted once
python reproduce/run_all.py --launcher slurm --dry-run     # the chained sbatch commands (site.env, below)
python reproduce/run_all.py --smoke                        # every code path on a few rows
python reproduce/check_paper.py --out $VMG_RUNS/paper/full --tex <paper source>/main.tex   # the tables, number by number
```

`check_paper.py` reads the paper's LaTeX source, which this repository does not include. EDM2 checkpoints that NVIDIA used but did not release are rebuilt from its raw snapshots by
`fetch_weights.py`, which the floats that need them run first.

Nothing is read from stored results: a float is rebuilt from the released checkpoints and datasets
([`DATA.md`](DATA.md)). The only data shipped are the anonymised user-study votes and rows
([`data/`](data/)) and the small test fixtures. On a cluster, copy `slurm/site.example.env` to `slurm/site.env`.
`envs/*.lock.txt` are the exact package versions of the paper's runs. `tests/test_registry.py` checks that every
`fig:`/`tab:` label of the paper has a function.

## The paper's floats

| function | label(s) | generator | what it needs |
|---|---|---|---|
| `fig_teaser` | fig:teaser (wone, random_guidance, teaser-samples, teaser-nested) | `figures/fig_teaser.py` | every model's fit; SD 3.5 M renders; the SD 1.5 ↔ CommonCanvas sweep; two EDM2-XS runs (`experiments/nested_in64.py`, `slurm/train_xs_2e27.sbatch`) |
| `fig_concept_story` | fig:concept, fig:story | `figures/fig_concept_story.py` | `toys/story2d.py`: 1,024 MLP denoisers per N on the 2-D mixture, Adam and SGD |
| `fig_wmse_published` | fig:wmse-published | `figures/fig_wmse_published.py` | calibration of the 50 pairs of `rosters/wmse_pairs.py` |
| `fig_moments` | fig:moments | `figures/fig_moments_row.py` | `figures/fig_moments.py --compute` (2-D), `toys/moments_highd.py` (ImageNet-64) |
| `fig_cfg_regimes` | fig:cfg-regimes | `figures/fig_cfg_regimes.py` | `experiments/cfg_regimes_in64.py`: two label-dropout EDM2-XS runs (`slurm/train_xs_2e27.sbatch`), EDM2-XL and L, windowed-guidance FIDs |
| `fig_tuned_w_metric` | fig:tuned-w-metric | `figures/fig_tuned_w_metric.py` | constant-w sweeps of SD 3.5 M and SD 1.5 |
| `tab_tuned_w` | tab:tuned-w | `tables/tab_tuned_w.py` | the SD 3.5 M sweep, our constant and schedule |
| `tab_pub_all` | tab:pub-all | `tables/tab_pub_all.py` | Pub. vs ours on the eleven models |
| `tab_pub_sched` | tab:pub-sched | `tables/tab_pub_sched.py` | the published schedules at both weights |
| `tab_user_study` | tab:user-study, tab:user-study-h2h | `tables/tab_user_study.py` | `data/user_study_votes.json` |
| `fig_wmse_sigma` | fig:wmse-sigma | `figures/fig_wmse_sigma.py` | the same calibrations, with the extra pairs |
| `fig_shared_error` | fig:shared-error | `figures/fig_shared_error.py` | `experiments/errors_in64.py gram / analyze`: six EDM2 nets on 10,000 noised ImageNet-64 val images |
| `fig_mean_worse` | fig:mean-worse | `figures/fig_mean_worse.py` | the same analysis; EDM2-S ← XS at 14 constants (`experiments/ens_edm2.py sample / eval`) |
| `fig_cov_trajectory` | fig:cov-trajectory, fig:cov-trajectory-in64 | `figures/fig_cov_trajectory.py` | `experiments/seedcov.py` (1,000 seeds of SD 3.5 M, one prompt) and `experiments/seedcov_edm2.py` on the EDM2 CFG and AG pairs |
| `fig_covmatch` | fig:covmatch | `figures/fig_covmatch.py` | the SD 3.5 M calibration with the `chancov` sums |
| `fig_cross_edm2xs` | fig:cross-edm2xs | `figures/fig_cross_seedpair.py` | two EDM2-XS trainings (`experiments/seedpair_in64.py`, `slurm/*seedpair*`) |
| `fig_cross_pairs` | fig:cross-pairs, fig:cross-sd15-sd21, fig:cross-sd21-cc | `figures/fig_cross_pairs.py` | the SD 1.5 ↔ SD 2.1 and SD 2.1 ↔ CommonCanvas sweeps |
| `fig_diversity_lx` | fig:diversity-lx | `figures/fig_diversity_lx_in64.py` | `toys/corr_diversity_in64.py` |
| `fig_diversity_ladder` | fig:diversity-ladder | `figures/fig_ag_ladder.py` | `toys/ag_diversity.py` (32 trained 2-D pairs) and `toys/corr_flow_sim.js` (node) |
| `fig_errfield_lx` | fig:errfield-lx | `figures/fig_errfield_lsigma.py` | `experiments/errfield_edm2.py spread / spreadt` |
| `fig_frontier_sd35m` | fig:frontier-sd35m | `figures/fig_frontier_sd35m.py` | w = 1..8 and our schedule |
| `tab_edm2_fail` | tab:edm2-fail | `tables/tab_edm2_fail.py` | NVIDIA's four EDM2-S presets (CFG on ImageNet-512, autoguidance on ImageNet-64) at Pub., our constant and our schedule, 10,000 samples each |
| `fig_samples_pub` | fig:samples-pub | `figures/fig_samples_pub.py` | 24 rows of SD 1.5, SD 3.5 L, FLUX.2 at Pub. and ours |
| `fig_samples_sd35m` | fig:samples-sd35m | `figures/fig_samples_sd35m.py` | the user study's eight arms, 500 rows |
| `tab_full_{image,video,audio,motion}` | tab:full-* | `tables/tab_full_*.py` | Pub. vs ours, every metric |
| `tab_impl` | tab:impl-models, tab:impl-data | `tables/tab_impl.py` | the model Specs |
| `tab_ncalib` | tab:ncalib | `tables/tab_ncalib.py` | `experiments/ncalib.py blocks / gram / fit`: the SD 3.5 M and AudioLDM 2 calibration passes with 8-row block sums, 2,000 subsets per size |
| `tab_noise_control` | tab:noise-control | `tables/tab_noise_control.py` | `sample.py --isonoise / --shapenoise` |
| `tab_objective` | tab:objective | `tables/tab_objective.py` | `compute_w.py`'s three pools on the eleven models |
| `tab_user_study_fable` | tab:user-study-fable | `tables/tab_user_study.py --rater fable` | the votes |

## Extras

Built only on request (`run_all.py --extras`, or `--only <label>`): floats the paper dropped, and other
methods' baselines beside ours.

| function | what |
|---|---|
| `fig_moments_density`, `fig_scalecopy`, `fig_vgap`, `fig_ellipse` | earlier figures (`toys/bigboot16.py` is fig:scalecopy's 81,000 nets) |
| `fig_diversity_pair`, `fig_ergodic_gap` | 2-D toys of correlated errors (`toys/corr_flow_sim.js`, node) and of averaging over inputs versus retrainings (`toys/ergodic_cov.py`) |
| `tab_extra_latte`, `tab_pub_sched_all` | Latte's table; published schedules on nine models |
| `tab_baselines_sd35m` | SIMCA, Step AG and CFG-OEC beside ours (`baselines/`, `vmg/baselines.py`) |

Side studies with a script but no registry entry (each script's docstring has its commands):

| script | question |
|---|---|
| `experiments/ens_edm2.py`, `ens_sd35.py`, `ens_analyze.py`, `toys/ensemble_toy.py` | guidance with an ensemble of models |
| `toys/story2d_scales.py` | guidance scales of trained 2-D pairs, centered vs second-moment vs matched-Σ |
| `experiments/ddpm_posterior.py` (`vmg/ddpm.py`) | a DDPM step with x0 drawn around D1 with the covariance of D1 − D0 |
| `experiments/sizegap_in64.py` | what an EDM2-S minus EDM2-XS difference is made of |
| `experiments/spread_in64.py`, `figures/fig_spread_in64.py` | the spread of retrained denoisers against N on ImageNet-64 |
| `experiments/wvar_edm2.py`, `figures/fig_wvar_edm2.py` | the guided prediction's variance along EDM's trajectories |
| `figures/fig_ergodic_{cov,diff,mat}.py` | more views of `toys/ergodic_cov.py` |
| `figures/fig_errfield_{edm2,pairs,random,spread}.py` | more views of `experiments/errfield_edm2.py` |
| `toys/corr_diversity.py` | the 2-D correlated-noise toy behind fig:diversity-pair |
| `toys/ellipse_highd.py`, `figures/fig_ellipse_highd.py` | fig:ellipse at CIFAR-10 size |
| `figures/fig_firstmoment_sd35m.py`, `fig_steps_gain_ratio.py`, `fig_moments_grid.py` | first-moment matching; gain against steps; the 2×2 fig:moments |

## Checks

| check | what it shows |
|---|---|
| `checks/smoke.py --model M` | the adapter equals its released pipeline's network call; fits, the guidance hook and sampling run |
| `checks/sd35m.py` | SD 3.5 Medium end to end: data order, latents, network parity, the full fit |
| `checks/metrics_image.py`, `checks/legacy_*.py` | the image detectors against the stored reference features, and the figure scripts on the paper's stored results: maintainer checks that need the pre-release result files, which are not in this repository |
