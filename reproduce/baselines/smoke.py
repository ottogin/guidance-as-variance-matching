#!/usr/bin/env python
"""GPU gates for the baseline rules inside the SD 3.5 Medium sampler (one GPU, a few minutes).

    python reproduce/baselines/smoke.py

  B1  a constant 4.5 through guide_hook (Mix) is bitwise the plain weight path, with and without --fp32-state
  B2  --fp32-state: every scheduler step saw a float32 state (asserted inside generate), and it changes the result
  B3  Free2 with alpha = 4.5, beta = -3.5 (the same rule in float32) is Mix(4.5) to bf16 rounding at every call
  B4  Step AG reaches every level, and its pruned levels are the conditional prediction
  B5  CFG-OEC fires on some (row, level) pairs and changes the result
Two evaluation rows, 40 steps.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
failed = []


def gate(name, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)
    if not ok:
        failed.append(name)


def main():
    import torch
    from vmg import baselines as B
    from vmg.data import coco
    from vmg.models import get
    m = get("sd35m")
    m.load()
    rows = coco.rows("eval")[:2]
    caps, z = coco.captions(rows), m.initial_noise(rows)
    L = m.spec.steps
    rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm())     # noqa: E731
    for f32 in (False, True):
        plain = m.generate(caps, z, np.full(L, 4.5), fp32_state=f32)
        mixed = m.generate(caps, z, None, guide=B.Mix(np.full(L, 4.5)), fp32_state=f32)
        gate(f"B1 Mix(4.5) = plain{' (fp32 state)' if f32 else ''}", torch.equal(plain, mixed),
             f"relative L2 {rel(mixed, plain):.1e}, dtype {mixed.dtype}")
        if f32:
            gate("B2 fp32 state", mixed.dtype == torch.float32 and not torch.equal(plain16, plain),
                 f"state {plain.dtype}; bf16-state vs fp32-state latents differ by {rel(plain16, plain):.1e}")
            ref = plain
        else:
            plain16 = plain
    # Free2 forms the same rule in float32, Mix in bf16 as diffusers does. Call by call they must agree to bf16
    # rounding; over 40 steps that rounding grows like any other (bf16 vs fp32 state above moves the result by
    # rel(plain16, ref)), so the whole trajectory is held to that scale, not to rounding.
    mix, free2, per_call = B.Mix(np.full(L, 4.5)), B.Free2(np.full(L, 4.5), np.full(L, -3.5)), []

    class Both:
        def reset(self):
            pass

        def __call__(self, k, vu, vc):
            a = mix(k, vu, vc)
            per_call.append(rel(free2(k, vu, vc), a))
            return a
    m.generate(caps, z, None, guide=Both(), fp32_state=True)
    free = m.generate(caps, z, None, guide=free2, fp32_state=True)
    scale = rel(plain16, ref)
    gate("B3 Free2(4.5, -3.5) = Mix(4.5)", max(per_call) < 1e-2 and rel(free, ref) < 2 * scale,
         f"per call within {max(per_call):.1e} (bf16 rounding); trajectories {rel(free, ref):.1e} apart, against "
         f"{scale:.1e} for bf16 vs fp32 state")
    log = []
    ks = B.k_stop(0.5, L)

    class Spy(B.StepAG):
        def __call__(self, k, vu, vc):
            out = super().__call__(k, vu, vc)
            log.append((k, bool(torch.equal(out, vc))))
            return out
    m.generate(caps, z, None, guide=Spy(np.full(L, 4.5), ks), fp32_state=True)
    lv = [k for k, _ in log]
    gate("B4 Step AG", lv == list(range(L)) and all(c == (k >= ks) for k, c in log),
         f"{len(log)} calls at levels {lv[0]}..{lv[-1]}; conditional-only from level {ks}")
    oec = B.OEC(np.full(L, 4.5), 0.9)
    out = m.generate(caps, z, None, guide=oec, fp32_state=True)
    st = oec.stats()
    gate("B5 CFG-OEC fires", st["fired"] > 0 and not torch.equal(out, ref),
         f"fired on {st['fired']} of {st['calls']} (row, level) pairs; cos(A, B) quantiles {st['cos_quantiles']}; "
         f"result moves {rel(out, ref):.1e} from Pub.")
    print("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
