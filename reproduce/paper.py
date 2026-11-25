"""Every figure and table of the paper, one function each: the pipeline jobs it needs and the script that draws it.

    python reproduce/run_all.py --list                  # this registry
    python reproduce/run_all.py --only tab:objective    # one float, from scratch

A function returns a Recipe: `jobs` (reproduce/plan.Job: GPU passes, sampling, metrics; shared jobs run once) and
`make`, the CPU commands that write the float from those runs. In commands, {out} is run_all.py's output folder,
{runs} the runs folder ($VMG_RUNS) and {task} a job array's index. A float whose generator is not in this
repository raises NotImplementedError("pending -- ..."), and run_all.py lists it as pending.

Sections: "main" and "appendix" are the paper's floats; "extra" are floats outside the paper and side
studies, built only with run_all.py --extras (reproduce/README.md lists the side studies without a recipe).
"""
from dataclasses import dataclass, field

from reproduce import plan
from reproduce.plan import (AV_MODELS, BB16_REDUCE, BB16_TRAIN, LATTE_SWEEP, MOMENTS_CACHE, Job, _arm, _wmse_roster,
                            av_pub_vs_ours, baseline_jobs, calibrate, cross_jobs, edm2_fail_jobs, frontier_jobs, image_pub_vs_ours,
                            mdm_mm_jobs, pub_arm, pub_sched_all_jobs, sched_jobs, sweep, tuned_w_jobs, vm_arm)


@dataclass
class Recipe:
    jobs: list = field(default_factory=list)
    make: list = field(default_factory=list)      # CPU commands (python scripts and their arguments)
    note: str = ""


@dataclass(frozen=True)
class Float:
    labels: tuple            # the paper's \label{}s this function builds (several when one generator writes them)
    files: str               # what it writes, as the paper includes it
    section: str             # main | appendix | extra
    fn: object

    @property
    def label(self):
        return self.labels[0]

    @property
    def name(self):
        return self.fn.__name__

    @property
    def doc(self):
        return (self.fn.__doc__ or "").strip().split("\n")[0]

    def recipe(self):
        r = self.fn()
        if isinstance(r.make, str):
            r.make = [r.make]
        return r

    def pending(self):
        """None if the float can be built here, else why not (the stub's message)."""
        try:
            self.recipe()
        except NotImplementedError as e:
            return str(e)
        return None


FLOATS = []


def paper_float(*labels, files, section):
    def deco(fn):
        FLOATS.append(Float(tuple(labels), files, section, fn))
        return fn
    return deco


def pending(what):
    raise NotImplementedError(f"pending -- {what}")


def by_label():
    """label -> Float, for every label (a Float with several labels appears under each)."""
    return {lab: f for f in FLOATS for lab in f.labels}


# ================================================================== main text

@paper_float("fig:teaser", "fig:wone", "fig:random_guidance", "fig:teaser-samples", "fig:teaser-nested",
             files="Figures/teaser-B.tex, teaser-B.pdf, samples/*.jpg, teaser/p2-*.jpg", section="main")
def fig_teaser():
    """Fig. 1: published vs our weights (a), samples (b), guidance in both directions (c), checkpoints (d)."""
    from reproduce.figures import fig_teaser as FT
    from reproduce.rosters.sweeps import arm
    jobs = [j for m, d in OBJECTIVE_MODELS for j in calibrate(m, d, 8, 3.0)]                                  # (a)
    jobs += (userstudy_jobs() + sweep("sd35m", "coco", [1.6, 2.4], 16, 2.3)
             + vm_arm("sd35m", "coco", "global", 16, 2.3))                                                    # (b)
    jobs += [j for key, v in FT.cross_arms()
             for j in _arm(key, "coco", f"--w {v:g}", arm(v), 2, 0.2, extra=f"--n {FT.N_C} --tag {FT.TAG_C}")]  # (c)
    return Recipe(jobs + nested_jobs(), ["reproduce/figures/fig_samples_sd35m.py --out {out} --teaser",
                                         "reproduce/figures/fig_teaser.py --out {out} --photos"],
                  note="(d) trains two EDM2-XS models (2^27 images each, 8 GPUs); prep needs ImageNet train ($VMG_IMAGENET_TRAIN, DATA.md)")


@paper_float("fig:concept", "fig:story", files="concept_story.pdf", section="main")
def fig_concept_story():
    """Fig. 2: the guidance mechanism, and trained 2-D denoisers on independent training sets."""
    d = "{runs}/_toys/story2d"
    sd = "reproduce/toys/story2d.py"
    train = [Job(f"{sd} train --optim {o} --task {{task}} --out {d}", "gpu", 7, 1.1, smoke=f"{sd} test")
             for o in ("adam", "sgd")]
    return Recipe(train, f"reproduce/figures/fig_concept_story.py --src {d} --out {{out}}/concept_story",
                  note="one array task per N = 64 .. 4,096 and optimizer: 1,024 MLPs side by side, ~10 min on one GPU")


@paper_float("fig:wmse-published", files="wmse-published.pdf", section="main")
def fig_wmse_published():
    """Fig. 3: MSE-optimal weights against published ones, 50 pairs."""
    return Recipe(_wmse_roster(), "reproduce/figures/fig_wmse_published.py --out {out}/wmse-published",
                  note="the 50 pairs of reproduce/rosters/wmse_pairs.py, with NVIDIA's published weights for the six "
                       "ImageNet-512 autoguidance pairs")


# ImageNet-64 train as one uint8 array ($VMG_IMAGENET/img64_train_u8.npy) from the official train JPEGs
# ($VMG_IMAGENET_TRAIN, DATA.md): fig:moments (b) reads it, and seedpair_in64.py prep makes the EDM2-XS training zip
IN64_TRAIN = Job("reproduce/experiments/spread_in64.py prep", "cpu", 1, 3.0,
                 smoke="reproduce/experiments/spread_in64.py prep --per-class 10")     # 10,000 images
MH = "reproduce/toys/moments_highd.py"


def moments_highd_jobs():
    """fig:moments panel (b): ImageNet-64 point masses, M(1) / M(2) / M(inf) on 5 chunks of 2e5 samples each."""
    prep = Job(f"{MH} prep --dss in64", "gpu", 1, 0.3, after=(IN64_TRAIN.cmd,), smoke=f"{MH} test")
    cells = [Job(f"{MH} cell --ds in64 --k {k} --chunk {{task}}", "gpu", 5, 2.0, after=(prep.cmd,),
                 smoke=f"{MH} test") for k in ("1", "2", "inf")]
    red = Job(f"{MH} figure --dss in64", "cpu", 1, 0.1, after=tuple(c.cmd for c in cells), smoke=f"{MH} test")
    return [IN64_TRAIN, prep] + cells + [red]


@paper_float("fig:moments", files="moment-samplers-row.pdf", section="main")
def fig_moments():
    """Fig. 5: samplers with the posterior's mean, mean + covariance, or the full posterior, against steps."""
    return Recipe([MOMENTS_CACHE] + moments_highd_jobs(),
                  "reproduce/figures/fig_moments_row.py --out {out}/moment-samplers-row",
                  note="(a) the 2-D mixture (fig_moments.py's cache); (b) 10,000 ImageNet-64 training images: prep "
                       "needs --in64 (or $VMG_IMAGENET), a folder with img64_train_u8.npy and img64_train_labels.npy")


@paper_float("fig:cfg-regimes", files="cfg-regimes-img64.pdf", section="main")
def fig_cfg_regimes():
    """CFG on ImageNet-64: label information versus model error, and windowed guidance (App. B.3)."""
    CR = "reproduce/experiments/cfg_regimes_in64.py"
    train = [xs_2e27_train(*r) for r in (("DA", 0, 0.1), ("DB", 1, 0.1), ("A", 0), ("B", 1))]
    gram = Job(f"{CR} gram --shard {{task}} --shards 4", "gpu", 4, 1.0, after=tuple(t.cmd for t in train),
               smoke=f"{CR} test")
    sw = Job(f"{CR} sweep --point {{task}}", "gpu", 12, 2.5, after=(train[0].cmd, train[1].cmd), smoke=f"{CR} test")
    analyze = Job(f"{CR} analyze", "cpu", 1, 0.0, after=(gram.cmd, sw.cmd), smoke=f"{CR} test")
    return Recipe([IN64_TRAIN, XS_PREP, *train, gram, sw, analyze],
                  "reproduce/figures/fig_cfg_regimes.py --out {out}/cfg-regimes-img64",
                  note="four EDM2-XS trainings of 2^27 images (8 GPUs each); A and B, shared with fig:teaser, give "
                       "App. B.3's seed-to-seed distance of the conditional-only runs")


@paper_float("fig:tuned-w-metric", files="tuned-w-metric-models.pdf", section="main")
def fig_tuned_w_metric():
    """Fig. 6: each metric against a constant w, with the published and our weights marked."""
    jobs = tuned_w_jobs(("sd35m", "sd15")) + [j for m, h in (("sd35m", 2.3), ("sd15", 1.0))
                                              for j in pub_arm(m, "coco", 16, h) + vm_arm(m, "coco", "global", 16, h)]
    return Recipe(jobs, "reproduce/figures/fig_tuned_w_metric.py --out {out}/tuned-w-metric-models")


@paper_float("tab:tuned-w", files="tuned-w.tex", section="main")
def tab_tuned_w():
    """Table 1: constants tuned per metric, the published one and ours, on every metric (SD 3.5 Medium)."""
    jobs = tuned_w_jobs() + vm_arm("sd35m", "coco", "global", 16, 2.3) + vm_arm("sd35m", "coco", "schedule", 16, 2.3)
    return Recipe(jobs, "reproduce/tables/tab_tuned_w.py --out {out}")


@paper_float("tab:pub-all", files="pub-all.tex", section="main")
def tab_pub_all():
    """Table 2: published weight against our schedule on all eleven models."""
    return Recipe(image_pub_vs_ours() + av_pub_vs_ours([m for m in AV_MODELS if m != "latte"]),     # Latte: extra
                  "reproduce/tables/tab_pub_all.py --out {out}")


@paper_float("tab:pub-sched", files="pub-sched.tex", section="main")
def tab_pub_sched():
    """Table 3: published schedules at the released weight and at ours (SD 3.5 Medium)."""
    return Recipe(sched_jobs("sd35m", "coco", 16, 2.3), "reproduce/tables/tab_pub_sched.py --out {out}")


@paper_float("tab:user-study", "tab:user-study-h2h", files="Figures/user-study-elo.tex, user-study-h2h.tex",
             section="main")
def tab_user_study():
    """Table 4 and App. E: Elo ratings and head-to-head results of the user study."""
    return Recipe([], "reproduce/tables/tab_user_study.py --out {out}",
                  note="from reproduce/data/user_study_votes.json (anonymised votes of people)")


# ================================================================== appendix

@paper_float("fig:wmse-sigma", files="wmse-sigma.pdf", section="appendix")
def fig_wmse_sigma():
    """w_MSE(sigma) per pair, every model of Fig. 3 and the extra pairs."""
    return Recipe(_wmse_roster(extra=True), "reproduce/figures/fig_wmse_sigma.py --out {out}/wmse-sigma")


EI, ENS, EID = "reproduce/experiments/errors_in64.py", "reproduce/experiments/ens_edm2.py", "{runs}/errors-in64"
EI_FETCH = Job(f"{EI} fetch", "cpu", 1, 0.1, smoke=f"{EI} fetch")     # downloads only


def errors_in64_jobs():
    """fig:shared-error and fig:mean-worse (a): per-image error Grams of six EDM2 networks on ImageNet-64 val."""
    gram = Job(f"{EI} gram --shard {{task}} --shards 8", "gpu", 8, 2.0, after=(EI_FETCH.cmd,),
               smoke=f"{EI} gram --n 16 --shard 0 --shards 32 --out {EID}/smoke")
    ana = Job(f"{EI} analyze", "cpu", 1, 0.1, after=(gram.cmd,), smoke=f"{EI} analyze --dir {EID}/smoke")
    return [EI_FETCH, gram, ana]


def errors_in64_sweep():
    """fig:mean-worse (b, c): EDM2-S guided by XS at the constants of errors_in64.SWEEP_W, FID and FD_DINOv2."""
    from reproduce.experiments.errors_in64 import SWEEP_W
    from reproduce.rosters.sweeps import arm
    jobs = [EI_FETCH]
    for w in SWEEP_W:
        f = f"--arm {arm(w)} --d1 S45 --d0 XS --w {w:g}"
        s = Job(f"{ENS} sample --out {EID}/sweep {f}", "gpu", 1, 0.15, after=(EI_FETCH.cmd,),
                smoke=f"{ENS} sample --out {EID}/smoke/sweep {f} --neval 8")
        jobs += [s, Job(f"{ENS} eval --out {EID}/sweep --arm {arm(w)}", "gpu", 1, 0.1, after=(s.cmd,),
                        smoke=f"{ENS} eval --out {EID}/smoke/sweep --arm {arm(w)} --n 8")]
    return jobs


@paper_float("fig:shared-error", files="shared-error.pdf", section="appendix")
def fig_shared_error():
    """The shared-error test with EDM2 S / XS / XL on ImageNet-64."""
    return Recipe(errors_in64_jobs(), "reproduce/figures/fig_shared_error.py --out {out}/shared-error")


@paper_float("fig:mean-worse", files="mean-worse.pdf", section="appendix")
def fig_mean_worse():
    """Guidance moves the prediction away from the posterior mean (EDM2 errors and an S <- XS sweep)."""
    return Recipe(errors_in64_jobs() + errors_in64_sweep(),
                  "reproduce/figures/fig_mean_worse.py --out {out}/mean-worse",
                  note="sweep: ens_edm2.py; FDs vs NVIDIA img64.pkl, k-NN reference from val JPEGs")


SEEDCOV = "reproduce/experiments/seedcov.py"


@paper_float("fig:cov-trajectory", "fig:cov-trajectory-in64", files="cov-trajectory.pdf", section="appendix")
def fig_cov_trajectory():
    """The two predictions of a guided pair along the unguided sampler: CFG on SD 3.5 Medium (one prompt, 1,000
    seeds), and EDM2-S on ImageNet-64 (school bus) under CFG and under autoguidance with EDM2-XS."""
    d = "{runs}/seedcov/sd35m"
    jobs = [Job(f"{SEEDCOV} work --out {d}", "gpu", 4, 1.0, smoke=f"{SEEDCOV} work --n 16 --out {d}")]
    for st in ("analyze", "full", "x0", "align"):
        jobs.append(Job(f"{SEEDCOV} {st} --out {d}", "cpu", 1, 0.3, after=(jobs[-1].cmd,),
                        smoke=f"{SEEDCOV} {st} --n 16 --out {d}"))
    for pair, model in (("cfg", "edm2-64-cfg-s-xsu-fid"), ("ag", "edm2-64-ag-s-xs-fid")):
        e = f"{{runs}}/seedcov/in64-{pair}"
        get = Job(f"reproduce/fetch_weights.py --model {model}", "cpu", 1, 1.5,     # CFG: rebuilds both EMAs
                  smoke=f"reproduce/fetch_weights.py --model {model} --check")
        sm = "edm2-64-xs-cfg" if pair == "cfg" else model    # smoke: a released CFG pair, no 70 GB rebuild
        w = Job(f"reproduce/experiments/seedcov_edm2.py work --model {model} --out {e}", "gpu", 1, 0.5,
                after=(get.cmd,), smoke=f"reproduce/experiments/seedcov_edm2.py work --model {sm} --n 8 --out {e}")
        al = f"reproduce/experiments/seedcov_edm2.py align --out {e}"
        jobs += [get, w, Job(al, "cpu", 1, 0.2, after=(w.cmd,), smoke=al)]
    return Recipe(jobs, [f"reproduce/figures/fig_cov_trajectory.py --sd35m {d} --cfg {{runs}}/seedcov/in64-cfg "
                         f"--ag {{runs}}/seedcov/in64-ag --out {{out}}/cov-trajectory"],
                  note="the CFG pair's two EMAs are rebuilt from NVIDIA's raw snapshots "
                       "(reproduce/fetch_weights.py: ~70 GB of snapshots, then reconstruct_phema.py)")


@paper_float("fig:covmatch", files="covmatch-pair-sd35m.pdf", section="appendix")
def fig_covmatch():
    """Matching the whole channel covariance gives nearly the schedule that matches its trace (App. D.1)."""
    return Recipe(calibrate("sd35m", "coco", 8, 3.0),
                  "reproduce/figures/fig_covmatch.py --out {out}/covmatch-pair-sd35m",
                  note="the SD 3.5 Medium calibration pass with the chancov extra (plan.EXTRAS)")


SP = "reproduce/experiments/seedpair_in64.py"
XS_PREP = Job(f"{SP} prep", "cpu", 1, 2.0, after=(IN64_TRAIN.cmd,),     # the EDM2-XS training zip
              smoke=f"{SP} prep --per-class 10")


def xs_2e27_train(run, seed, label_dropout=None):
    """One EDM2-XS ImageNet-64 training of 2^27 images (reproduce/slurm/train_xs_2e27.sbatch), shared by the floats
    that use it: A and B (fig:teaser-nested, fig:cfg-regimes), DA and DB with label dropout (fig:cfg-regimes)."""
    args = f"{run} {seed}" + (f" {label_dropout:g}" if label_dropout else "")
    return Job(f"reproduce/slurm/train_xs_2e27.sbatch {args}", "gpu", 1, 60.0, after=(XS_PREP.cmd,), runner="sbatch",
               res="--gres=gpu:8 --time=12:00:00", smoke=f"reproduce/slurm/train_xs_2e27.sbatch --smoke {args}")


NI = "reproduce/experiments/nested_in64.py"


def nested_jobs():
    """fig:teaser-nested: two EDM2-XS runs of 2^27 images, the Gram of their snapshots, the guidance sweeps."""
    train = [xs_2e27_train("A", 0), xs_2e27_train("B", 1)]
    tr = tuple(t.cmd for t in train)
    gram = Job(f"{NI} gram --shard {{task}} --shards 4", "gpu", 4, 2.0, after=tr, smoke=f"{NI} gram --smoke")
    sw = Job(f"{NI} sweep --task {{task}} --tasks 8", "gpu", 8, 35.0, after=tr, smoke=f"{NI} sweep --smoke")
    an = Job(f"{NI} analyze", after=(sw.cmd,), smoke=f"{NI} test")
    return [IN64_TRAIN, XS_PREP, *train, gram, sw, an, Job(f"{NI} fit", after=(an.cmd, gram.cmd), smoke=f"{NI} test")]


SPD = "{runs}/edm2-64-seedpair"


def seedpair_jobs():
    """Two EDM2-XS runs on ImageNet-64 that differ only in their seed, and the guided arms between them."""
    train = [Job(f"reproduce/slurm/train_seedpair.sbatch {r}", "gpu", 1, 40.0, after=(XS_PREP.cmd,), runner="sbatch",
                 res="--gres=gpu:4 --time=12:00:00", smoke=f"reproduce/slurm/train_seedpair.sbatch --smoke {r}")
             for r in ("A", "B")]     # ~10 h on 4 H200s
    phema = Job("reproduce/slurm/phema_seedpair.sbatch 0.035", "cpu", 1, 1.0, after=tuple(t.cmd for t in train),
                runner="sbatch", smoke="reproduce/slurm/phema_seedpair.sbatch --smoke 0.035")
    lst = f"{SPD}/arms-fig.txt"
    arms = Job(f"{SP} arms --list {lst}", "cpu", 1, 0.0, smoke=f"{SP} arms --list {lst} --smoke")
    work = Job(f"reproduce/slurm/seedpair_worker.sbatch {lst}", "gpu", 8, 15.0, after=(phema.cmd, arms.cmd),
               runner="sbatch", smoke=f"reproduce/slurm/seedpair_worker.sbatch --smoke {lst}")
    table = Job(f"{SP} table", "cpu", 1, 0.0, after=(work.cmd,), smoke=f"{SP} table")
    return [IN64_TRAIN, XS_PREP, *train, phema, arms, work, table]


@paper_float("fig:cross-edm2xs", files="cross-edm2xs-seeds.pdf", section="appendix")
def fig_cross_edm2xs():
    """Guidance in both directions between two EDM2-XS runs that differ only in their training seed."""
    return Recipe(seedpair_jobs(), "reproduce/figures/fig_cross_seedpair.py --out {out}",
                  note="trains two EDM2-XS models (32 Mi images each, 4 GPUs); prep needs the ImageNet train set "
                       "($VMG_IMAGENET_TRAIN, DATA.md); the worker and training jobs are Slurm scripts")


@paper_float("fig:cross-pairs", "fig:cross-sd15-sd21", "fig:cross-sd21-cc", files="cross-sd15-sd21.pdf, cross-sd21-cc.pdf",
             section="appendix")
def fig_cross_pairs():
    """SD 1.5 and SD 2.1 (a), SD 2.1 and CommonCanvas (b) guiding each other, in both directions."""
    return Recipe(cross_jobs(), "reproduce/figures/fig_cross_pairs.py --out {out}")


CD = "reproduce/toys/corr_diversity_in64.py"
CD_SETUP = Job(f"{CD} setup", "cpu", 1, 0.2, after=(XS_PREP.cmd,), smoke=f"{CD} setup")      # in64-10k.npz: fig:diversity-lx, fig:errfield-lx


@paper_float("fig:diversity-lx", files="diversity-lx-in64.pdf", section="appendix")
def fig_diversity_lx():
    """Fidelity and coverage of M(2) with a correlated error field on 1,000 ImageNet-64 Gaussians, against l_x."""
    work = Job(f"{CD} work --model gauss --sweep big", "gpu", 8, 8.0, after=(CD_SETUP.cmd,), smoke=f"{CD} test")
    units = "{runs}/_toys/corr_diversity_in64/gauss/big"
    return Recipe([IN64_TRAIN, XS_PREP, CD_SETUP, work],
                  f"reproduce/figures/fig_diversity_lx_in64.py --collect --units {units}/units --data {units}/points.json "
                  f"--out {{out}}/diversity-lx-in64.pdf",
                  note="the 10,000 images come from the seed-pair ImageNet-64 zip (seedpair_in64.py prep)")


@paper_float("fig:diversity-pair", files="diversity-field-pair.pdf", section="extra")
def fig_diversity_pair():
    """Final samples of M(2) under two draws of a correlated error field (2-D mixture)."""
    return Recipe([], "reproduce/figures/fig_diversity_pair.py --simulate --data {runs}/_toys/diversity_pair/samples.npz "
                      "--out {out}/diversity-field-pair.pdf", note="needs node (reproduce/toys/corr_flow_sim.js)")


AG = "reproduce/toys/ag_diversity.py"


@paper_float("fig:diversity-ladder", files="diversity-ladder.pdf", section="main")
def fig_diversity_ladder():
    """M(2) with a fixed error field of growing (l_x, l_t), above guidance of a trained 2-D pair with growing w."""
    d = "{runs}/_toys/ag_diversity/fresh"
    weak = "--weak 64,4,40000 --weak-n 256"
    train = Job(f"{AG} train --out {d} {weak}", "gpu", 1, 0.3,
                smoke=f"{AG} train --out {d}-smoke --pairs 2 --smoke --weak 64,4,100 --weak-n 64")
    samp = Job(f"{AG} sample --out {d} {weak} --interval 1", "gpu", 1, 0.1,
               after=(train.cmd,), smoke=f"{AG} sample --out {d}-smoke --smoke --weak 64,4,100 --weak-n 64 --interval 1")
    return Recipe([train, samp], f"reproduce/figures/fig_ag_ladder.py --simulate --m2dir {{runs}}/_toys/ag_diversity/m2ladder "
                                 f"--guided {d}/samples-w64d4s40k-n256-i1.npz --out {{out}}/diversity-ladder.pdf",
                  note="needs node (reproduce/toys/corr_flow_sim.js); 32 pairs side by side, ~15 min on one GPU")


EC = "reproduce/toys/ergodic_cov.py"


@paper_float("fig:ergodic-gap", files="ergodic-gap.pdf", section="extra")
def fig_ergodic_gap():
    """Averaging over inputs versus over retrainings (eq. 16, first line), 2-D mixture with trained MLPs."""
    d = "{runs}/_toys/ergodic"
    train = [Job(f"{EC} train --out {d} --n {n}", "gpu", 1, 0.2, smoke=f"{EC} train --out {d} --n {n} --G 4 --steps 100")
             for n in (128, 2048)]
    mat = Job(f"{EC} matrices --out {d}", "gpu", 1, 0.1, after=tuple(t.cmd for t in train), smoke=f"{EC} matrices --out {d}")
    return Recipe(train + [mat], f"reproduce/figures/fig_ergodic_gap.py --npz {d}/ergodic-mat.npz --ns 128,2048 "
                                 f"--paper --out {{out}}/ergodic-gap.pdf")


EF = "reproduce/experiments/errfield_edm2.py"


@paper_float("fig:errfield-lx", files="errfield-lx-in64.pdf", section="appendix")
def fig_errfield_lx():
    """Correlation lengths l_x, l_t of the CFG error difference along EDM2's sampler (EDM2-S, ImageNet-64)."""
    d = "{runs}/_toys/errfield_edm2/lsigma"
    halves = (",".join(map(str, range(16))), ",".join(map(str, range(16, 32))))   # EDM2's 32 levels, two jobs each
    jobs = [IN64_TRAIN, XS_PREP, CD_SETUP]               # units(): the 10,000 images of corr_diversity_in64 setup
    for p, lv in enumerate(halves):
        jobs += [Job(f"{EF} spread --d0 XSu --levels {lv} --out {d} --tag spread-cfg-c850-p{p}", "gpu", 1, 0.15,
                     after=(CD_SETUP.cmd,),
                     smoke=f"{EF} spread --d0 XSu --out {d} --n-base 8 --per-base 3 --levels 0,8 --tag smoke-spread"),
                 Job(f"{EF} spreadt --d0 XSu --levels {lv} --out {d} --tag spreadt-cfg-c850-p{p}", "gpu", 1, 0.2,
                     after=(CD_SETUP.cmd,), smoke=f"{EF} spreadt --d0 XSu --out {d} --n-base 8 --levels 8 --tag smoke-spreadt")]
    return Recipe(jobs, f"reproduce/figures/fig_errfield_lsigma.py {d} --tag spread-cfg-c850 --ttag spreadt-cfg-c850 "
                        f"--paper --out {{out}}/errfield-lx-in64.pdf")


@paper_float("fig:frontier-sd35m", files="frontier-sd35m.pdf", section="appendix")
def fig_frontier_sd35m():
    """Quality trade-offs of SD 3.5 Medium: constants w = 1..8 against our schedule."""
    return Recipe(frontier_jobs(), "reproduce/figures/fig_frontier_sd35m.py --layout shade --out {out}/frontier-sd35m")


@paper_float("tab:edm2-fail", files="tab-edm2-fail.tex", section="appendix")
def tab_edm2_fail():
    """Where the rule fails (App. C.7): EDM2-XXL at NVIDIA's FID-tuned CFG and autoguidance presets, Pub. against
    ours."""
    return Recipe(edm2_fail_jobs(), "reproduce/tables/tab_edm2_fail.py --out {out}",
                  note="two NVIDIA pairs on ImageNet-512 (CFG by the unconditional XS, autoguidance by M): the fits "
                       "read 1,024 ImageNet val images and the k-NN metrics 10,000 others ($VMG_IMAGENET, prep 512); "
                       "python -m vmg.models._edm fetch gets the pickles and NVIDIA's dataset-refs")


@paper_float("fig:samples-edm2-cfg", "fig:samples-edm2-ag",
             files="samples-edm2-{cfg,ag}-{rows.pdf,1.tex,rest.tex}", section="appendix")
def fig_samples_edm2():
    """App. F: the first 100 rows of tab:edm2-fail's arms, Pub. against our schedule, for both EDM2-XXL pairs."""
    return Recipe(edm2_fail_jobs(), "reproduce/figures/fig_samples_edm2.py --out {out}",
                  note="draws each seed's class on the GPU (or --classes JSON)")


SHOWCASE = (("sd15", "coco", ""), ("sd35l", "coco", " --fp32-state"), ("flux2-klein", "mjhq", ""))


@paper_float("fig:samples-pub", files="samples-pub.tex, samples-pub/*.jpg", section="appendix")
def fig_samples_pub():
    """Pub. against Ours on the first 24 evaluation captions of SD 1.5, SD 3.5 Large and FLUX.2 [klein]."""
    jobs = []
    for m, d, f32 in SHOWCASE:
        cal = calibrate(m, d, 8, 3.0)
        md = f"sample.py --model {m} --dataset {d} --n 24 --tag showcase{f32}"
        sm = f"sample.py --model {m} --dataset {d} --n 8 --tag smoke{f32}"
        jobs += cal + [Job(f"{md} --published --arm pub", "gpu", 1, 0.1, smoke=f"{sm} --published --arm pub"),
                       Job(f"{md} --vm schedule --arm vm-schedule", "gpu", 1, 0.1, after=(cal[1].cmd,),
                           smoke=f"{sm} --vm schedule --arm vm-schedule")]
    return Recipe(jobs, "reproduce/figures/fig_samples_pub.py --out {out}",
                  note="the paper's images were rendered with the pre-release pipeline's schedules (the same rule); a "
                       "rebuild uses this repository's fits")


# the user study's eight settings (tab:user-study), SD 3.5 Medium on COCO rows 5,705-6,204, float32 latent state
USERSTUDY_ARMS = ("--vm global", "--vm schedule", "--w 1.6", "--w 2.4", "--published", "--sched interval --base pub",
                  "--sched linear --base pub", "--sched tvcfg --base pub")


def userstudy_jobs():
    cal = calibrate("sd35m", "coco", 8, 3.0)
    md = "sample.py --model sd35m --dataset coco --n 500 --tag userstudy --fp32-state"
    sm = "sample.py --model sd35m --dataset coco --n 8 --tag smoke --fp32-state"
    return cal + [Job(f"{md} {flags}", "gpu", 1, 0.15, after=(cal[1].cmd,), smoke=f"{sm} {flags}")
                  for flags in USERSTUDY_ARMS]


@paper_float("fig:samples-sd35m", files="samples-sd35m-{1,rest,2..13}.tex, samples-app/*.jpg", section="appendix")
def fig_samples_sd35m():
    """100 unselected user-study rows of SD 3.5 Medium, eight settings each."""
    return Recipe(userstudy_jobs(), "reproduce/figures/fig_samples_sd35m.py --out {out}",
                  note="the study's rows, hidden rows and prompts: reproduce/data/user_study_rows.json; the paper's "
                       "images are the study's own renders (--renders), 2-8/255 from a rebuild")


@paper_float("tab:full-audio", files="full-audio.tex", section="appendix")
def tab_full_audio():
    """Every metric of AudioLDM 2 and Stable Audio Open, Pub. against Ours."""
    return Recipe(av_pub_vs_ours(["audioldm2", "sao"]), "reproduce/tables/tab_full_audio.py --out {out}")


@paper_float("tab:full-image", files="full-image.tex", section="appendix")
def tab_full_image():
    """Every metric of the four image models, Pub. against Ours."""
    return Recipe(image_pub_vs_ours(), "reproduce/tables/tab_full_image.py --out {out}")


@paper_float("tab:full-motion", files="full-motion.tex", section="appendix")
def tab_full_motion():
    """Every metric of MDM (KIT-ML), Pub. against Ours, with MultiModality."""
    return Recipe(av_pub_vs_ours(["mdm"]) + mdm_mm_jobs(), "reproduce/tables/tab_full_motion.py --out {out}")


@paper_float("tab:full-video", files="full-video.tex", section="appendix")
def tab_full_video():
    """Every metric of SVD, Wan 2.1, CogVideoX and Cosmos, Pub. against Ours."""
    return Recipe(av_pub_vs_ours(["svd", "wan", "cogvideox", "cosmos"]), "reproduce/tables/tab_full_video.py --out {out}")


@paper_float("tab:impl-models", "tab:impl-data", files="impl.tex", section="appendix")
def tab_impl():
    """Models, samplers and data (App. B), from the model Specs."""
    return Recipe([], "reproduce/tables/tab_impl.py --out {out}")


NC = "reproduce/experiments/ncalib.py"
# tab:ncalib: (model, dataset, shards, rough GPU-hours of the pass over every calibration row)
NCALIB = (("sd35m", "coco", 8, 3.0), ("audioldm2", "audiocaps", 4, 1.3))


def ncalib_jobs():
    """Per model: the calibration pass with 8-row block sums (GPU), their Grams, the subset refits (CPU)."""
    jobs = []
    for m, d, shards, h in NCALIB:
        md = f"--model {m} --dataset {d}"
        b = Job(f"{NC} blocks {md} --shard {{task}} --shards {shards}", "gpu", shards, h,
                smoke=f"{NC} blocks {md} --n 32 --tag smoke")
        g = Job(f"{NC} gram {md} --shards {shards}", "cpu", 1, 0.5, after=(b.cmd,), smoke=f"{NC} gram {md} --tag smoke")
        f = Job(f"{NC} fit {md}", "cpu", 1, 0.2, after=(g.cmd,), smoke=f"{NC} fit {md} --sizes 16 --subsets 5 --tag smoke")
        jobs += [b, g, f]
    return jobs


@paper_float("tab:ncalib", files="ncalib.tex", section="appendix")
def tab_ncalib():
    """The constant scale fitted from n calibration examples (SD 3.5 Medium, AudioLDM 2)."""
    return Recipe(ncalib_jobs(), "reproduce/tables/tab_ncalib.py --out {out}",
                  note="2,000 random subsets per size n = 16..2,048, each fitted from its own examples (means "
                       "included): 5,705 COCO photos x 40 levels and 2,223 AudioCaps captions x 200 levels")


@paper_float("tab:noise-control", files="noise-control.tex", section="appendix")
def tab_noise_control():
    """Matching the variance without the guidance direction: noise that ignores x_t (SD 3.5 Medium)."""
    cal = calibrate("sd35m", "coco", 8, 3.0)
    jobs = cal + sweep("sd35m", "coco", [1.0], 16, 2.3) + vm_arm("sd35m", "coco", "schedule", 16, 2.3)
    for flag, arm in (("--isonoise", "isonoise"), ("--shapenoise", "shapenoise")):
        jobs += _arm("sd35m", "coco", flag, arm, 16, 2.3, after=(cal[1].cmd,))
    return Recipe(jobs, "reproduce/tables/tab_noise_control.py --out {out}")


OBJECTIVE_MODELS = [("sd15", "coco"), ("sd35m", "coco"), ("sd35l", "coco"), ("flux2-klein", "mjhq"),
                    ("svd", "ucf101"), ("wan", "openvid"), ("cogvideox", "openvid"), ("cosmos", "bridge"),
                    ("audioldm2", "audiocaps"), ("sao", "songdescriber"), ("mdm", "kitml")]


@paper_float("tab:objective", files="objective.tex", section="appendix")
def tab_objective():
    """The constant scale under three objectives (ours, L1, L2) on eleven models (App. D.2)."""
    jobs = [j for m, d in OBJECTIVE_MODELS for j in calibrate(m, d, 8, 3.0)]
    return Recipe(jobs, "reproduce/tables/tab_objective.py --out {out}",
                  note="the three pools are compute_w.py --mode global --pool mean | l1 | l2 of one fit")


@paper_float("tab:user-study-fable", files="user-study-elo-fable.tex", section="appendix")
def tab_user_study_fable():
    """The same study rated by Claude Fable 5.1, kept apart from the people's votes."""
    return Recipe([], "reproduce/tables/tab_user_study.py --rater fable --out {out}")


# ================================================================== extras: outside the paper, side studies

@paper_float("fig:moments-density", files="moment-samplers-m12-density.pdf", section="extra")
def fig_moments_density():
    """Densities of the M(1) and M(2) samples on the 2-D mixture."""
    return Recipe([MOMENTS_CACHE], "reproduce/figures/fig_moments.py --show 1,2 --density 1,2 --out {out}/moment-samplers-m12")


@paper_float("fig:scalecopy", files="scalecopy.pdf", section="extra")
def fig_scalecopy():
    """81,000 trained nets per family: the strong-weak difference is a scaled copy of the posterior spread."""
    return Recipe([BB16_TRAIN, BB16_REDUCE], "reproduce/figures/fig_scalecopy.py --out {out}/scalecopy")


@paper_float("fig:vgap", files="vgap-interval.pdf", section="extra")
def fig_vgap():
    """The velocity gap of SD 3.5 Medium and the guidance interval."""
    return Recipe(calibrate("sd35m", "coco", 8, 3.0), "reproduce/figures/fig_vgap.py --out {out}/vgap-interval")


@paper_float("fig:ellipse", files="ellipse-share-merged.pdf", section="extra")
def fig_ellipse():
    """The two-dimensional example with exact denoisers."""
    return Recipe([], "reproduce/figures/fig_ellipse.py --out {out}/ellipse-share")


@paper_float("tab:extra-latte", files="extra-latte.tex", section="extra")
def tab_extra_latte():
    """Latte on UCF-101: Pub., the best constant and ours."""
    return Recipe(av_pub_vs_ours(["latte"]) + sweep("latte", "ucf101", LATTE_SWEEP, 16, AV_MODELS["latte"][1]),
                  "reproduce/tables/tab_extra_latte.py --out {out}")


@paper_float("tab:pub-sched-all", files="pub-sched-all.tex", section="extra")
def tab_pub_sched_all():
    """Published schedules on nine models."""
    return Recipe(pub_sched_all_jobs(), "reproduce/tables/tab_pub_sched_all.py --out {out}")


@paper_float("tab:baselines-sd35m", files="baselines-sd35m.tex", section="extra")
def tab_baselines_sd35m():
    """SIMCA, Step AG and CFG-OEC beside ours on SD 3.5 Medium."""
    return Recipe(baseline_jobs(), "reproduce/tables/tab_baselines_sd35m.py --out {out}")


assert len({lab for f in FLOATS for lab in f.labels}) == sum(len(f.labels) for f in FLOATS), "a label is registered twice"
_ = plan          # re-exported for run_all.py
