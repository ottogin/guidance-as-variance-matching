#!/usr/bin/env python
"""Guidance with an ensemble of SD 3.5 Medium and Large (a side study).

Both models share the SD 3 VAE latent space, so their clean predictions combine. Four members per call:
Mc, Mu (Medium conditional / unconditional) and Lc, Lu (Large). The guided prediction is

    Dtilde = D0 + w (D1 - D0) + alpha sum_{k in eg} z_k (D^k - mean_eg D),    z ~ N(0, I) fresh per call,

with D1, D0 the plain means of their member lists. Sampler: Large's released Euler grid (28 steps, shift 3),
float32 latent state, each member on its own pipeline's text encoding (empty prompt for the unconditional ones).

    python reproduce/experiments/ens_sd35.py moments --n 512 --shard i --shards K   # Grams of the 4 members
    python reproduce/experiments/ens_sd35.py sample --arm A --d1 Lc --d0 Lu --w 3.5 [--eg ...] --n 5000 --shard i --shards K
then reproduce/experiments/evaluate.py --model sd35m --dataset coco --arm A --tag ens --n 5000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402
from vmg.models import get  # noqa: E402
from vmg.models.sd import captions  # noqa: E402

MEMBERS = ("Mc", "Mu", "Lc", "Lu")


class Ens:
    def __init__(self, need):
        self.need = need
        self.M = get("sd35m")
        self.L = get("sd35l")
        self.M.load()
        if any(m.startswith("L") for m in need):
            self.L.load()
            vm, vl = self.M.pipe.vae, self.L.pipe.vae
            same = all(torch.equal(a, b) for a, b in zip(vm.state_dict().values(), vl.state_dict().values()))
            assert same, "SD 3.5 M and L VAEs differ: predictions are not in one latent space"
        sch = (self.L if any(m.startswith("L") for m in need) else self.M).pipe.scheduler
        sch.set_timesteps(28, device="cuda")
        self.sig, self.ts = sch.sigmas.float(), sch.timesteps

    @torch.no_grad()
    def enc(self, model, prompts):
        e, ne, p, npl = model.pipe.encode_prompt(prompt=list(prompts), prompt_2=None, prompt_3=None, device="cuda",
                                                 do_classifier_free_guidance=True, max_sequence_length=model.max_tokens)
        return dict(c=(e, p), u=(ne, npl))

    def text(self, prompts):
        out = {}
        if any(m.startswith("M") for m in self.need):
            out["M"] = self.enc(self.M, prompts)
        if any(m.startswith("L") for m in self.need):
            out["L"] = self.enc(self.L, prompts)
        return out

    @torch.no_grad()
    def preds(self, x, k, txt):
        """{member: D (float32)} at state x (float32), level k, for the members in self.need."""
        s = self.sig[k]
        out = {}
        for fam, model in (("M", self.M), ("L", self.L)):
            br = [m[1] for m in self.need if m[0] == fam]
            if not br:
                continue
            emb = torch.cat([txt[fam][b][0] for b in br])
            pool = torch.cat([txt[fam][b][1] for b in br])
            xin = torch.cat([x] * len(br)).to(model.dtype)
            v = model.pipe.transformer(hidden_states=xin, timestep=self.ts[k].expand(len(xin)),
                                       encoder_hidden_states=emb, pooled_projections=pool, return_dict=False)[0]
            for b, vb in zip(br, v.chunk(len(br))):
                out[fam + b] = x - s * vb.float()
        return out


@torch.no_grad()
def moments(a):
    E = Ens(list(MEMBERS))
    m = E.M
    rows = m.rows("coco", "calib")[:a.n]
    rows = [r for i, r in enumerate(rows) if i % a.shards == a.shard]
    S, K = len(E.ts), len(MEMBERS)
    gram = np.zeros((S, K + 1, K + 1))
    mean = None
    cnt = 0
    t0 = time.time()
    for b0 in range(0, len(rows), a.batch):
        rb = rows[b0:b0 + a.batch]
        x0 = m.latents(m.images("coco", rb)).float()
        txt = E.text(captions("coco", rb))
        for k in range(S):
            g = torch.Generator().manual_seed(7919 * k + 1)
            eps = torch.stack([torch.randn(x0.shape[1:], generator=torch.Generator().manual_seed(100000 * k + r))
                               for r in rb]).cuda()
            s = float(E.sig[k])
            x = (1 - s) * x0 + s * eps
            P = E.preds(x, k, txt)
            Y = torch.stack([P[n].reshape(len(rb), -1).double() for n in MEMBERS] + [x0.reshape(len(rb), -1).double()])
            gram[k] += torch.einsum("ibd,jbd->ij", Y, Y).cpu().numpy()
            if mean is None:
                mean = np.zeros((S, K + 1, Y.shape[-1]))
            mean[k] += Y.sum(1).cpu().numpy()
        cnt += len(rb)
        print(f"{cnt}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    d = runs.run_dir("sd35m", "coco", "ens") / "moments"
    d.mkdir(exist_ok=True)
    np.savez(d / f"ens-{a.shard:03d}-of-{a.shards:03d}.npz", names=np.array(MEMBERS), sigma=E.sig[:-1].cpu().numpy(),
             n=cnt, gram=gram, mean=mean)


def parse_w(s, S):
    if s.startswith("interval:"):
        return s
    v = np.array([float(x) for x in s.split(",")])
    return np.full(S, v[0]) if len(v) == 1 else v


@torch.no_grad()
def sample(a):
    d1 = a.d1.split(",")
    d0 = a.d0.split(",") if a.d0 else []
    eg = a.eg.split(",") if a.eg else []
    need = sorted(set(d1) | set(d0) | set(eg))
    E = Ens(need)
    S = len(E.ts)
    ws, als = parse_w(a.w, S), parse_w(a.alpha, S)
    if isinstance(ws, str):           # interval:w:tlo:thi on the flow time t = sigma
        _, w, lo, hi = ws.split(":")
        sg = E.sig[:-1].cpu().numpy()
        ws = np.where((sg > float(lo)) & (sg <= float(hi)), float(w), 1.0)
    m = E.M
    ids = m.rows("coco", "eval")[:a.n]
    rows = [r for i, r in enumerate(ids) if i % a.shards == a.shard]
    out = runs.samples_dir("sd35m", "coco", a.arm, "ens")
    json.dump(dict(vars(a), w_levels=list(map(float, ws))), open(out / "args.json", "w"))
    t0 = time.time()
    for b0 in range(0, len(rows), a.batch):
        rb = [r for r in rows[b0:b0 + a.batch] if not (out / f"{r:05d}.png").exists()]
        if not rb:
            continue
        txt = E.text(captions("coco", rb))
        z = m.initial_noise(rb)
        x = z.to("cuda", torch.bfloat16).float()
        gz = torch.Generator(device="cuda").manual_seed(int(a.zseed) * 1000003 + rb[0])
        with torch.no_grad():
            for k in range(S):
                P = E.preds(x, k, txt)
                D1 = torch.stack([P[n] for n in d1]).mean(0)
                if d0:
                    D0 = torch.stack([P[n] for n in d0]).mean(0)
                    D = D0 + float(ws[k]) * (D1 - D0)
                else:
                    D = D1
                if eg and float(als[k]) != 0:
                    Em = torch.stack([P[n] for n in eg])
                    zz = torch.randn(len(eg), len(rb), 1, 1, 1, generator=gz, device="cuda")
                    D = D + float(als[k]) * (zz * (Em - Em.mean(0))).sum(0)
                s, sn = E.sig[k], E.sig[k + 1]
                v = (x - D) / s
                x = x + (sn - s) * v
        for r, im in zip(rb, m.decode(x)):
            runs.save_image(im, out / f"{r:05d}.png")
        print(f"{a.arm}: {b0 + len(rb)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["moments", "sample"])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arm")
    ap.add_argument("--d1", default="Lc")
    ap.add_argument("--d0", default="")
    ap.add_argument("--w", default="1")
    ap.add_argument("--eg", default="")
    ap.add_argument("--alpha", default="0")
    ap.add_argument("--zseed", default="0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    {"moments": moments, "sample": sample}[a.stage](a)


if __name__ == "__main__":
    main()
