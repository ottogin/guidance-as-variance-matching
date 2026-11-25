"""The pipeline jobs behind the paper's floats, and the helpers that build them (reproduce/paper.py uses these).

A Job is one pipeline command. {task} is the array index, {runs} the runs folder ($VMG_RUNS) and {out} the output
folder of run_all.py. Jobs that several floats request are run once.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    cmd: str                 # a python script under the repository root and its arguments (runner "python"), or a
                             # bash script that is also a complete sbatch script (runner "sbatch")
    kind: str = "cpu"        # gpu | cpu
    array: int = 1           # number of array tasks ({task} = 0 .. array - 1)
    hours: float = 0.0       # rough device-hours of the full (non-smoke) job, all tasks together
    smoke: str = ""          # the command in --smoke mode (default: the same command)
    after: tuple = ()        # commands (cmd strings) that must finish first
    runner: str = "python"   # python | sbatch (run with bash here; submitted as itself on Slurm)
    res: str = ""            # extra sbatch options for this job (e.g. --gres=gpu:4), after the site's own


# the python each model's jobs run in (reproduce/envs/README.md): job.sbatch --env NAME runs $PY_NAME from site.env,
# and the local launcher runs $PY_NAME when it is set. Models not listed run in the core env ($PY).
ENVS = {"wan": "video", "cogvideox": "video", "svd": "ucf", "svd-frame1": "ucf", "latte": "ucf", "cosmos": "cosmos",
        "audioldm2": "audio", "sao": "sao", "mdm": "mdm", "chroma": "video"}     # Chroma: sentencepiece
METRIC_ENVS = {"sao": "saomet"}           # evaluate.py / compare.py, where a model's metrics need another env
# evaluate.py jobs that must all run on one GPU type: AudioLDM 2's own CLAP embeds identical audio differently on
# different GPUs (cosine down to 0.56), and its reference records the type. The slurm launcher sends these jobs
# with $VMG_SBATCH_GPU_<TYPE> (e.g. VMG_SBATCH_GPU_L40S) instead of $VMG_SBATCH_GPU.
EVAL_GPU = {"audioldm2": "L40S"}


def model_of(cmd):
    import re
    m = re.search(r"--model (\S+)", cmd)
    return m.group(1) if m else None


def env_of(cmd):
    """The env a job command runs in (None: the core env)."""
    m = model_of(cmd)
    if cmd.startswith("reproduce/experiments/") and m in METRIC_ENVS:
        return METRIC_ENVS[m]
    return ENVS.get(m)


def gpu_type_of(cmd):
    """The GPU type a job is pinned to (None: any)."""
    return EVAL_GPU.get(model_of(cmd)) if cmd.startswith("reproduce/experiments/evaluate.py") else None


# extra sums some calibration passes collect for other floats (vmg/extras.py; no extra network evaluations)
EXTRAS = {("sd35m", "coco"): "chancov"}           # fig:covmatch


def calibrate(model, dataset, shards, hours, n="all"):
    """compute_w + compute_wmse on one model, on every calibration row (--n all) or the first n: the GPU moments
    pass, then the CPU fits."""
    ex = f" --extras {EXTRAS[model, dataset]}" if (model, dataset) in EXTRAS else ""
    md = f"--model {model} --dataset {dataset} --n {n}"
    mom = f"compute_w.py {md} --stage moments --shard {{task}} --shards {shards}{ex}"
    fit = f"compute_w.py {md} --stage fit"
    dm = f"compute_w.py {md} --stage fit --datamean"
    wm = f"compute_wmse.py {md} --stage fit"
    sm = f"compute_w.py --model {model} --dataset {dataset} --smoke --stage moments{ex}"
    return [Job(mom, "gpu", shards, hours, smoke=sm),
            Job(fit, after=(mom,), smoke=fit + " --tag smoke"),
            Job(dm, after=(mom,), smoke=dm + " --tag smoke"),
            Job(wm, after=(mom,), smoke=wm + " --tag smoke")]


# evaluation rows of a --smoke arm (default 8); an MDM row is a whole replication (~7 min, MultiModality ~40)
SMOKE_N = {("mdm", "kitml"): 2, ("mdm", "kitml-mm"): 1}


def _arm(model, dataset, flag, arm, shards, hours, after=(), extra=""):
    """sample.py with `flag` into `arm`, then evaluate.py on it; `extra` goes to both (e.g. "--n 1024 --tag sub")."""
    md = f"--model {model} --dataset {dataset}"
    ex = f" {extra}" if extra else ""
    n = SMOKE_N.get((model, dataset), 8)
    s = f"sample.py {md} {flag}{ex} --shard {{task}} --shards {shards}"
    e = f"reproduce/experiments/evaluate.py {md} --arm {arm}{ex}"
    return [Job(s, "gpu", shards, hours, after=after, smoke=f"sample.py {md} {flag} --n {n} --tag smoke"),
            Job(e, "gpu", 1, 0.3, after=(s,), smoke=f"reproduce/experiments/evaluate.py {md} --arm {arm} --n {n} "
                                                   f"--tag smoke")]


def pub_arm(model, dataset, shards, hours, extra=""):
    return _arm(model, dataset, "--published", "pub", shards, hours, extra=extra)


def vm_arm(model, dataset, kind, shards, hours, calib_shards=8, calib_hours=3.0, extra="", calib_n="all"):
    """calibrate (on calib_n calibration rows), then sample and evaluate with our weights (kind global | schedule).
    sample.py --vm reads the n<calib_n> fit when there is no fit on every row."""
    cal = calibrate(model, dataset, calib_shards, calib_hours, calib_n)
    return cal + _arm(model, dataset, f"--vm {kind}", f"vm-{kind}", shards, hours, after=(cal[1].cmd,), extra=extra)


def sweep(model, dataset, ws, shards, hours):
    """sample and evaluate one arm per constant weight (reproduce/rosters/sweeps.py)."""
    from reproduce.rosters.sweeps import arm
    return [j for w in ws for j in _arm(model, dataset, f"--w {w:g}", arm(w), shards, hours)]


def pub_vs_ours(model, dataset, shards, hours, kind="schedule", fd=False):
    """the published arm and ours, both evaluated, and their paired comparison (reproduce/experiments); fd also
    intervals the Frechet distances (the paper stars FVD and FAD, not the image FIDs)."""
    jobs = pub_arm(model, dataset, shards, hours) + vm_arm(model, dataset, kind, shards, hours)
    c = f"reproduce/experiments/compare.py --model {model} --dataset {dataset} --a pub --b vm-{kind}" + \
        (" --fd" if fd else "")
    after = tuple(j.cmd for j in jobs if j.cmd.startswith("reproduce/experiments/evaluate.py"))
    return jobs + [Job(c, after=after, smoke=f"{c} --reps 2 --tag smoke")]


# (model, dataset, rough device-hours to sample 10,000 evaluation rows once, all shards together)
IMAGE_MODELS = [("sd15", "coco", 1.0), ("sd35m", "coco", 2.3), ("sd35l", "coco", 6.0), ("flux2-klein", "mjhq", 10.0)]


def image_pub_vs_ours():
    return [j for m, d, h in IMAGE_MODELS for j in pub_vs_ours(m, d, 16, h)]


# video, world, audio and motion models: (dataset, rough H200-hours to sample its evaluation rows once), from the
# timings of the paper's runs (Latte 7.9 s per clip, SVD 18.9 s, Wan2.1 93 s, AudioLDM 2 1.9 s, MDM 20 replications)
AV_MODELS = {"latte": ("ucf101", 22.0), "svd": ("ucf101", 11.0), "wan": ("openvid", 27.0),
             "cogvideox": ("openvid", 20.0), "cosmos": ("bridge", 8.0), "audioldm2": ("audiocaps", 2.3),
             "sao": ("songdescriber", 1.7), "mdm": ("kitml", 2.2)}
LATTE_SWEEP = (1, 3, 5, 9)               # tab:extra-latte's constants besides Pub. (7.0)


def av_pub_vs_ours(models):
    return [j for m in models for j in pub_vs_ours(m, AV_MODELS[m][0], 16, AV_MODELS[m][1], fd=True)]


def tuned_w_jobs(models=("sd35m",)):
    from reproduce.rosters import sweeps as SW
    grid = {"sd35m": ("coco", SW.SD35M, 2.3), "sd15": ("coco", SW.SD15, 1.0), "edm2-512": ("imagenet512", SW.EDM2S, 0.5)}
    jobs = []
    for m in models:
        d, ws, h = grid[m]
        jobs += sweep(m, d, ws, 16, h)
    return jobs


# tab:edm2-fail: NVIDIA's FID-tuned EDM2-XXL guidance presets on ImageNet-512, (model, dataset, rough H200-hours to
# sample 10,000 images once). Our weights are fitted on the first EDM2_FAIL_N calibration rows (the paper's table).
EDM2_FAIL = [("edm2-512-cfg-xxl-xsu-fid", "imagenet512", 2.5), ("edm2-512-ag-xxl-m-fid", "imagenet512", 3.5)]
EDM2_FAIL_N = 1024


def edm2_fail_jobs():
    """tab:edm2-fail (and App. F's EDM2 samples, its first 100 rows): every preset at its published weight, our
    constant and our schedule, fitted on that pair."""
    return [j for m, d, h in EDM2_FAIL
            for j in pub_arm(m, d, 20, h) + vm_arm(m, d, "global", 20, h, calib_n=EDM2_FAIL_N)
            + vm_arm(m, d, "schedule", 20, h, calib_n=EDM2_FAIL_N)]


def sched_jobs(model, dataset, shards, hours, extra=""):
    """tab:pub-sched: the released constant, ours, and the three published schedules at both base weights."""
    from vmg.schedules import KINDS
    cal = calibrate(model, dataset, 8, 3.0)
    jobs = pub_arm(model, dataset, shards, hours, extra) + vm_arm(model, dataset, "global", shards, hours, extra=extra) \
        + vm_arm(model, dataset, "schedule", shards, hours, extra=extra)
    for k in KINDS:
        jobs += _arm(model, dataset, f"--sched {k} --base pub", f"pub-{k}", shards, hours, extra=extra)
        jobs += _arm(model, dataset, f"--sched {k} --base vm", f"gw-{k}", shards, hours, after=(cal[1].cmd,),
                     extra=extra)
    return jobs


def mdm_mm_jobs():
    """MDM's MultiModality (dataset kitml-mm: 5 mm_short replications) at Pub. and at our schedule fitted on kitml."""
    cal = calibrate("mdm", "kitml", 8, 3.0)
    jobs = _arm("mdm", "kitml-mm", "--published", "pub", 5, 3.5) + \
        _arm("mdm", "kitml-mm", "--vm schedule --weights-from kitml", "vm-schedule", 5, 3.5, after=(cal[1].cmd,))
    c = "reproduce/experiments/compare.py --model mdm --dataset kitml-mm --a pub --b vm-schedule"
    after = tuple(j.cmd for j in jobs if j.cmd.startswith("reproduce/experiments/evaluate.py"))
    return cal + jobs + [Job(c, after=after, smoke=f"{c} --reps 2 --tag smoke")]


def cross_jobs():
    """fig:cross-pairs: every point of the paper's two panels, 1,000 COCO evaluation rows each (tag 1k)."""
    from reproduce.figures.fig_cross_pairs import N, PAIRS, TAG, arms
    from reproduce.rosters.sweeps import arm
    return [j for _, m0, m1 in PAIRS for key, v, _ in arms(m0, m1)
            for j in _arm(key, "coco", f"--w {v:g}", arm(v), 2, 0.2, extra=f"--n {N} --tag {TAG}")]


def pub_sched_all_jobs():
    """tab:pub-sched-all: sched_jobs on its nine models (AudioLDM 2 on its first 1,024 test captions)."""
    hours = {m: h for m, _, h in IMAGE_MODELS} | {m: h for m, (_, h) in AV_MODELS.items()}
    hours["audioldm2"] *= 1024 / 4411                            # AudioLDM 2 on its first 1,024 captions
    data = {m: d for m, d, _ in IMAGE_MODELS} | {m: d for m, (d, _) in AV_MODELS.items()}
    extra = {"audioldm2": "--n 1024 --tag sub1024"}
    return [j for m in ("sd15", "sd35m", "sd35l", "flux2-klein", "wan", "cogvideox", "audioldm2", "sao", "mdm")
            for j in sched_jobs(m, data[m], 16, hours[m], extra.get(m, ""))]


def frontier_jobs():
    from reproduce.rosters.sweeps import SD35M
    return sweep("sd35m", "coco", [w for w in SD35M if w == int(w) and w <= 8], 16, 2.3) + \
        vm_arm("sd35m", "coco", "schedule", 16, 2.3)


BASELINE_TAG = "baselines"
# tab:baselines-sd35m: (sample.py flags without --fp32-state, arm, reference arm of its paired comparison, needs)
# needs: "fit" our fit, "sf" / "sb1" / "sb2" a SIMCA fit (reproduce/baselines/fit_simca.py)
BASELINE_ARMS = [
    ("--published", "pub-f32", None, ()),
    ("--vm global", "vm-global-f32", "pub-f32", ("fit",)),
    ("--vm schedule", "vm-schedule-f32", "pub-f32", ("fit",)),
    ("--mix simca-sf", "simca-sf-f32", "pub-f32", ("sf",)),
    ("--mix simca-sb1", "simca-sb1-f32", "pub-f32", ("sb1",)),
    ("--mix simca-sb2", "simca-sb2-f32", "pub-f32", ("sb2",)),
    ("--stepag 0.5 --base pub", "stepag50-pub-f32", "pub-f32", ()),
    ("--stepag 0.3 --base pub", "stepag30-pub-f32", "pub-f32", ()),
    ("--stepag 0.5 --base vm-schedule", "stepag50-vms-f32", "vm-schedule-f32", ("fit",)),
    ("--stepag 0.5 --base vm-refit", "stepag50-vmr-f32", "vm-global-f32", ("fit",)),
    ("--oec 0.9 --base pub", "oec90-pub-f32", "pub-f32", ()),
    ("--oec 0.9 --base vm-schedule", "oec90-vms-f32", "vm-schedule-f32", ("fit",)),
]


def baseline_jobs(model="sd35m", dataset="coco", shards=16, hours=2.3):
    """tab:baselines-sd35m (reproduce/baselines/): other methods' baselines beside ours, every arm with the paper's
    float32 latent state and our fitted weights, sampled under tag "baselines"; the fits read the untagged
    calibration. Every arm waits for the GPU gates of reproduce/baselines/smoke.py."""
    md = f"--model {model} --dataset {dataset}"
    cal = calibrate(model, dataset, 8, 3.0)
    gate = Job("reproduce/baselines/smoke.py", "gpu", 1, 0.2)
    fs = "reproduce/baselines/fit_simca.py"
    fits = {"sf": Job(f"{fs} {md} --form sf", after=(cal[3].cmd,), smoke=f"{fs} {md} --form sf --tag smoke"),
            "sb1": Job(f"{fs} {md} --form 1 --n 256", "gpu", 1, 0.3, smoke=f"{fs} {md} --form 1 --tag smoke"),
            "sb2": Job(f"{fs} {md} --form 2 --n 256", "gpu", 1, 0.3, smoke=f"{fs} {md} --form 2 --tag smoke")}
    pt = "reproduce/baselines/posterior_trace.py"
    trace = [Job(f"{pt} {md} --check-h", "gpu", 1, 0.2), Job(f"{pt} {md} --n 128", "gpu", 1, 1.2,
                                                             smoke=f"{pt} {md} --tag smoke")]
    need = {"fit": cal[1].cmd, **{k: j.cmd for k, j in fits.items()}}
    jobs, evals = cal + [gate] + list(fits.values()) + trace, {}
    for flag, arm, ref, needs in BASELINE_ARMS:
        aj = _arm(model, dataset, f"{flag} --fp32-state", arm, shards, hours,
                  after=(gate.cmd,) + tuple(need[n] for n in needs), extra=f"--tag {BASELINE_TAG}")
        evals[arm] = aj[1].cmd
        jobs += aj
    for _, arm, ref, _ in BASELINE_ARMS:
        if ref:
            c = f"reproduce/experiments/compare.py {md} --a {ref} --b {arm}"
            jobs.append(Job(f"{c} --tag {BASELINE_TAG}", after=(evals[ref], evals[arm]), smoke=f"{c} --reps 2 --tag smoke"))
    return jobs


def calibrate_pairs(pairs, shards=8):
    """calibrate() on every distinct (model, dataset) run behind a roster of pairs."""
    from reproduce.rosters import wmse_pairs as R
    hours = {R.AUTOG: 1.0, R.CFG: 1.0, R.T2I: 2.0, R.AVM: 4.0, R.CROSS: 2.0}     # rough GPU-hours per moments pass
    group = {(p.model, p.dataset): p.group for p in pairs}
    return [j for run in R.runs_of(pairs) for j in calibrate(*run, shards, hours[group[run]])]


def _wmse_roster(extra=False):
    from reproduce.rosters import wmse_pairs as R
    return calibrate_pairs(R.PAIRS + (R.EXTRA if extra else []))


BB16_TRAIN = Job("reproduce/toys/bigboot16.py --stage train --shard {task}", "gpu", 170, 55.0,
                 smoke="reproduce/toys/bigboot16.py --stage train --shard 0")
BB16_REDUCE = Job("reproduce/toys/bigboot16.py --stage reduce", "cpu", 1, 0.1,
                  smoke="reproduce/toys/bigboot16.py --stage reduce --partial", after=(BB16_TRAIN.cmd,))

MOMENTS_CACHE = Job("reproduce/figures/fig_moments.py --compute --procs 8", "cpu", 1, 0.7,
                    smoke="reproduce/figures/fig_moments.py --test")     # the samplers' moment checks only
