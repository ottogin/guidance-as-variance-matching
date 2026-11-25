#!/usr/bin/env python
"""Build the paper's figures and tables from scratch: the pipeline jobs they need, then the floats themselves.

    python reproduce/run_all.py --list                          # every float: its function, generator, what it needs
    python reproduce/run_all.py                                 # every float of the paper, running jobs here
    python reproduce/run_all.py --only fig:covmatch,tab_impl    # just these (labels or function names)
    python reproduce/run_all.py --extras --only fig:vgap        # floats outside the paper (section "extra")
    python reproduce/run_all.py --budget                        # rough device-hours per float
    python reproduce/run_all.py --smoke                         # one job per code path at smoke size (plumbing)
    python reproduce/run_all.py --launcher slurm --dry-run      # print the chained sbatch commands
    python reproduce/run_all.py --launcher slurm                # submit them (reproduce/slurm/site.env)

The registry is reproduce/paper.py: one function per float, returning its jobs and the commands that draw it.
Floats are written to --out (default $VMG_RUNS/paper/full); nothing writes into paper/. Floats whose generator is
not in this repository (paper.pending) are listed as pending at the end instead of failing. For --launcher slurm,
site.env sets VMG_SBATCH_GPU and VMG_SBATCH_CPU (partition, account, QoS, resources); jobs that are Slurm scripts
themselves (Job.runner "sbatch") are submitted as they are, with the site's options and their own (Job.res).
--smoke runs each code path once on a few rows (a sweep's constants collapse to one arm) and skips the figure and
table scripts, which read full runs. The local launcher runs a Slurm-script job with bash, where
reproduce/slurm/site.env is optional; for --launcher slurm, site.env also sets VMG_SBATCH_GPU_<TYPE> for the jobs
pinned to one GPU type (plan.EVAL_GPU).
"""
import argparse
import dataclasses
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce import paper, plan  # noqa: E402
from vmg import runs  # noqa: E402


def select(a):
    """The Floats to build: --only (labels or function names), else every paper float (and extras with --extras)."""
    fl = paper.FLOATS
    if a.only:
        want = a.only.split(",")
        by = {**paper.by_label(), **{f.name: f for f in fl}}
        missing = [w for w in want if w not in by]
        assert not missing, f"unknown floats {missing}; see --list"
        return list(dict.fromkeys(by[w] for w in want))
    return [f for f in fl if f.section != "extra" or a.extras]


def ordered_jobs(recipes, smoke):
    """Unique jobs of these recipes, dependencies first."""
    jobs = {}
    for r in recipes:
        for j in r.jobs:
            jobs.setdefault(j.cmd, j)
    order, seen = [], set()

    def visit(j):
        if j.cmd in seen:
            return
        seen.add(j.cmd)
        for c in j.after:
            if c in jobs:
                visit(jobs[c])
        order.append(j)
    for j in list(jobs.values()):
        visit(j)
    if smoke:                                    # one job per code path: a sweep's arms differ only in the weight
        keep, alias = {}, {}
        for j in order:
            alias[j.cmd] = keep.setdefault(shape(j), j).cmd
        order = [dataclasses.replace(j, after=tuple(dict.fromkeys(alias.get(c, c) for c in j.after)))
                 for j in keep.values()]
    return order


def shape(j):
    """A job's code path: its command with the constant weight and the published schedule's kind made generic, so
    that the arms of a sweep (and their evaluations) count once in --smoke. Distinct arms (pub, vm-*) stay. It reads
    the smoke command, so two jobs that differ only in their row count (e.g. --n 2000 and --n 1024) count once."""
    c = j.smoke or j.cmd
    for pat, rep in ((r"--w \S+", "--w W"), (r"--arm w[-0-9.]+\b", "--arm wW"), (r"--sched \S+", "--sched K"),
                     (r"--arm (pub|gw)-[a-z]+", r"--arm \1-K"), (r"--b w[-0-9.]+\b", "--b wW")):
        c = re.sub(pat, rep, c)
    return c


def runs_root(a):
    """Where the jobs write: $VMG_RUNS, or its _smoke subfolder for a local --smoke (on Slurm the jobs take VMG_RUNS
    from site.env, so a smoke run there should point site.env at a scratch folder)."""
    local_smoke = a.smoke and getattr(a, "launcher", "local") == "local"
    return runs.root() / SMOKE_ROOT if local_smoke else runs.root()


def fill(cmd, a, task=None):
    """A command with {runs}, {out} and (for an array task) {task} filled in."""
    cmd = cmd.replace("{runs}", str(runs_root(a))).replace("{out}", a.out)
    return cmd if task is None else cmd.replace("{task}", str(task))


def command(j, smoke):
    return j.smoke if smoke and j.smoke else j.cmd


def python_for(cmd):
    """$PY_<env> for a command that needs its own env (plan.ENVS), else this interpreter."""
    env = plan.env_of(cmd)
    py = os.environ.get(f"PY_{env}") if env else None
    if env and not py:
        print(f"warning: PY_{env} is not set; running {cmd.split()[0]} with {sys.executable}", flush=True)
    return py or sys.executable


SMOKE_ROOT = "_smoke"


def env(smoke=False):
    """The jobs' environment. --smoke runs every job under $VMG_RUNS/_smoke, so its small stand-ins (a 10-per-class
    ImageNet zip, 128 Ki-image trainings) never overwrite full runs; weights, caches and clones stay shared."""
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join([str(REPO)] + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else []))
    root = runs.root()
    if smoke:
        for var, sub in (("VMG_WEIGHTS", "_weights"), ("VMG_CACHE", "_cache"), ("VMG_SRC", "_src")):
            e.setdefault(var, str(root / sub))
        e["VMG_RUNS"] = str(root / SMOKE_ROOT)
    e.setdefault("VMG_RUNS", str(root))          # the Slurm scripts run here with bash read these under set -u;
    e.setdefault("PY", sys.executable)           # reproduce/slurm/site.env, when it exists, overrides both
    return e


def run_local(recipes, jobs, a):
    for j in jobs:
        cmd = command(j, a.smoke)
        tasks = [0] if a.smoke and "{task}" not in (j.smoke or "") else range(j.array)
        for t in tasks:
            line = fill(cmd, a, t)
            exe = ["bash"] if j.runner == "sbatch" else [python_for(line)]
            pe = plan.env_of(line)
            print(f"$ {'bash' if j.runner == 'sbatch' else '$PY_' + pe if pe else 'python'} {line}", flush=True)
            if not a.dry_run:
                subprocess.run([*exe, *shlex.split(line)], cwd=REPO, check=True, env=env(a.smoke))
    for r in recipes if not a.smoke else []:     # the generators read full runs (see --smoke)
        for m in r.make:
            line = fill(m, a)
            print(f"$ python {line}", flush=True)
            if not a.dry_run:
                subprocess.run([sys.executable, *shlex.split(line)], cwd=REPO, check=True, env=env())


def run_slurm(floats, recipes, jobs, a):
    gpu, cpu = os.environ.get("VMG_SBATCH_GPU"), os.environ.get("VMG_SBATCH_CPU")
    if not (gpu and cpu) and not a.dry_run:
        sys.exit("set VMG_SBATCH_GPU and VMG_SBATCH_CPU (e.g. in reproduce/slurm/site.env, then source it)")
    gpu, cpu = gpu or "<VMG_SBATCH_GPU>", cpu or "<VMG_SBATCH_CPU>"
    pinned = sorted({t for j in jobs if (t := plan.gpu_type_of(fill(command(j, a.smoke), a)))})
    unset = [f"VMG_SBATCH_GPU_{t}" for t in pinned if not os.environ.get(f"VMG_SBATCH_GPU_{t}")]
    if unset and not a.dry_run:
        sys.exit(f"set {', '.join(unset)}: the sbatch options of the jobs pinned to that GPU type (plan.EVAL_GPU; "
                 f"reproduce/slurm/site.example.env)")
    ids = {}
    tmpl = f"--chdir={REPO} reproduce/slurm/job.sbatch"

    def submit(opts, line, deps, script=False):
        dep = f" --dependency=afterok:{':'.join(deps)}" if deps else ""
        s = f"sbatch --parsable{dep} {opts} " + (f"--chdir={REPO} {line}" if script else f"{tmpl} {line}")
        print(s, flush=True)
        if a.dry_run:
            return f"<{len(ids)}>"
        return subprocess.check_output(shlex.split(s), cwd=REPO, text=True).strip().split(";")[0]
    for j in jobs:
        line = fill(command(j, a.smoke), a)
        arr = "" if a.smoke or j.array == 1 else f" --array=0-{j.array - 1}"
        deps = [ids[c] for c in j.after if c in ids]
        opts = gpu if j.kind == "gpu" else cpu
        typ = plan.gpu_type_of(line)
        if typ:                                      # jobs pinned to one GPU type (plan.EVAL_GPU)
            opts = os.environ.get(f"VMG_SBATCH_GPU_{typ}") or f"<VMG_SBATCH_GPU_{typ}>"
        opts += arr + (f" {j.res}" if j.res and not a.smoke else "")     # smoke: one GPU, the site's --time
        if j.runner == "sbatch":
            ids[j.cmd] = submit(opts, line, deps, script=True)
        else:
            pe = plan.env_of(line)
            ids[j.cmd] = submit(opts, (f"--env {pe} " if pe else "") + line, deps)
    for r in recipes if not a.smoke else []:
        deps = [ids[j.cmd] for j in r.jobs if j.cmd in ids]
        for m in r.make:                             # a float's commands run in order, after its jobs
            deps = [submit(cpu, fill(m, a), deps)]


def budget(floats):
    """Device-hours (Job.hours, rough) per float: its own total, and what it adds after the floats above it."""
    seen, tot = {}, {"gpu": 0.0, "cpu": 0.0}
    rec = {f.name: f.recipe() for f in floats if not f.pending()}
    print(f"{'float':26s} {'GPU-h own':>10s} {'GPU-h new':>10s}  jobs")
    for f in sorted((f for f in floats if f.name in rec), key=lambda f: -sum(j.hours for j in rec[f.name].jobs
                                                                              if j.kind == "gpu")):
        jobs = rec[f.name].jobs
        own = sum(j.hours for j in {j.cmd: j for j in jobs}.values() if j.kind == "gpu")
        new = sum(j.hours for j in {j.cmd: j for j in jobs}.values() if j.kind == "gpu" and j.cmd not in seen)
        for j in jobs:
            if j.cmd not in seen:
                seen[j.cmd] = j
                tot[j.kind] += j.hours
        print(f"{f.label:26s} {own:10.1f} {new:10.1f}  {len(jobs)}")
    print(f"total: {tot['gpu']:.0f} GPU-hours (rough, H200-class), {len(seen)} distinct jobs")


def listing(floats):
    for f in floats:
        p = f.pending()
        labels = ", ".join(f.labels)
        if p:
            print(f"{f.section:8s} {f.name:24s} PENDING  {labels}\n{'':34s}{p}")
            continue
        r = f.recipe()
        gens = "; ".join(m.split(" ")[0] for m in r.make)
        gpu = sum(j.hours for j in {j.cmd: j for j in r.jobs}.values() if j.kind == "gpu")
        print(f"{f.section:8s} {f.name:24s} {len(r.jobs):3d} jobs {gpu:6.1f} GPU-h  {labels}  <- {gens}")
        if r.note:
            print(f"{'':34s}{r.note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--budget", action="store_true", help="rough device-hours per float, shared jobs counted once")
    ap.add_argument("--only", help="comma-separated labels or function names, e.g. fig:covmatch,tab_objective")
    ap.add_argument("--extras", action="store_true", help="also the floats the paper dropped (section 'extra')")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--launcher", choices=("local", "slurm"), default="local")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None, help="where floats are written (default $VMG_RUNS/paper/full)")
    ap.add_argument("--skip", help="a regex: jobs whose command matches count as already done (not run; jobs that "
                                   "depend on them start without them), e.g. 'compute_w|compute_wmse' when the "
                                   "calibration's outputs already exist")
    a = ap.parse_args()
    a.out = a.out or str(runs.root() / "paper" / ("smoke" if a.smoke else "full"))
    floats = select(a)
    if a.list:
        listing(floats)
        return
    if a.budget:
        budget(floats)
        return
    todo = [(f, f.pending()) for f in floats if f.pending()]
    ready = [f for f in floats if not f.pending()]
    recipes = [f.recipe() for f in ready]
    os.makedirs(a.out, exist_ok=True)
    jobs = ordered_jobs(recipes, a.smoke)
    if a.skip:
        skipped = [j for j in jobs if re.search(a.skip, command(j, a.smoke))]
        jobs = [j for j in jobs if j not in skipped]
        print(f"skipping {len(skipped)} jobs matching {a.skip!r} (taken as done)")
    print(f"{len(ready)} floats, {len(jobs)} pipeline jobs, output {a.out}")
    if a.launcher == "local":
        run_local(recipes, jobs, a)
    else:
        run_slurm(ready, recipes, jobs, a)
    if todo:
        print(f"\npending ({len(todo)}): " + "; ".join(f"{f.label} ({p.split(' -- ')[0]})" for f, p in todo))


if __name__ == "__main__":
    main()
