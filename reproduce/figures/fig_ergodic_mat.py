#!/usr/bin/env python
"""Eq. 16's first line as 2x2 matrices, one network at a time (no guidance), from ergodic-mat.npz.

    python reproduce/figures/fig_ergodic_mat.py --npz DIR/ergodic-mat.npz --ns 128,2048 --out fig.png

Per network size N (one row each):
(a) eigenvalues of the difference  Delta = mean_g Cov_x(D^g) - Cov_{p,x}(D) = -Cov_g(mean_x D^g)  (both <= 0; shown
    as |.|), next to the eigenvalues of the retraining spread Vp = E_x Cov_p(D|x_t) for scale.
(b) the error part per direction: generalized eigenvalues of one net's error covariance over x_t, Cov_x(e^g), against
    Vp (eigenvalues of Vp^-1/2 Cov_x(e^g) Vp^-1/2). 1 = one net over inputs reproduces retraining in that direction.
    Mean net (lines) and the 25-75 % band of single nets.
(c) ellipses (1 sd) at three noise levels: Vp (black), Cov_x(e^g) of 8 single nets (thin), their mean (thick).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#0b0b0b", "#8a8984"


def inv_sqrt(M):
    w, V = np.linalg.eigh(M)
    return V @ np.diag(1 / np.sqrt(np.clip(w, 1e-30, None))) @ V.T


def ellipse(M, k=1.0):
    w, V = np.linalg.eigh(M)
    t = np.linspace(0, 2 * np.pi, 120)
    return (V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ np.stack([np.cos(t), np.sin(t)])) * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ns", default="128,2048")
    ap.add_argument("--show", default="0.2,1.1,6")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.npz)
    ns = [int(v) for v in a.ns.split(",")]
    sd = float(np.sqrt(np.trace(z[f"{ns[0]}/Cx0"].mean(0)) / 2))
    s = z["sigmas"] / sd
    show = [int(np.argmin(np.abs(s - float(v)))) for v in a.show.split(",")]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "lines.linewidth": 1.5})
    fig, ax = plt.subplots(len(ns), 2 + len(show), figsize=(2.3 * (2 + len(show)), 2.3 * len(ns)),
                           constrained_layout=True, squeeze=False)
    for r, n in enumerate(ns):
        Cg, Cpool, E, Vp = z[f"{n}/C"], z[f"{n}/Cpool"], z[f"{n}/E"], z[f"{n}/Vp"]
        diff = Cg.mean(1) - Cpool                                                           # [S,2,2]
        ld = np.linalg.eigvalsh(diff)                                                       # ascending
        lv = np.linalg.eigvalsh(Vp)
        A = ax[r, 0]
        A.plot(s, -ld[:, 0], color=C[0], marker="o", ms=3, label=r"$|\Delta|$, larger eig.")
        A.plot(s, -ld[:, 1], color=C[0], marker="o", ms=3, ls="--", mfc="white", label=r"$|\Delta|$, smaller")
        A.plot(s, lv[:, 1], color=INK, lw=1.0, label=r"$V_p$, larger eig.")
        A.plot(s, lv[:, 0], color=INK, lw=1.0, ls="--", label=r"$V_p$, smaller")
        A.set(xscale="log", yscale="log", xlabel=r"$\sigma/\sigma_{\rm data}$", ylabel="eigenvalue",
              title=f"(a) N = {n}: " + r"$\mathrm{Cov}_{x_t}(D^g)-\mathrm{Cov}_{p,x_t}(D)$")
        if r == 0:
            A.legend(frameon=False, fontsize=6.5, loc="lower right")
        # generalized eigenvalues, per net
        W = inv_sqrt_batch(Vp)                                                              # [S,2,2]
        ge = np.linalg.eigvalsh(np.einsum("sij,sgjk,skl->sgil", W, E, W))                   # [S,G,2]
        gm = np.linalg.eigvalsh(np.einsum("sij,sjk,skl->sil", W, E.mean(1), W))              # [S,2]
        A = ax[r, 1]
        for j, (ls, lab) in enumerate(((":", "smaller"), ("-", "larger"))):
            q25, q75 = np.percentile(ge[:, :, j], [25, 75], axis=1)
            A.fill_between(s, q25, q75, color=C[1 + j], alpha=0.2, lw=0)
            A.plot(s, gm[:, j], color=C[1 + j], marker="o", ms=3, label=f"mean net, {lab}")
        A.axhline(1, color=MUTED, lw=0.8, ls=":")
        A.set(xscale="log", ylim=(0, 1.6), xlabel=r"$\sigma/\sigma_{\rm data}$",
              ylabel=r"eig. of $V_p^{-1/2}\,\mathrm{Cov}_{x_t}(e^g)\,V_p^{-1/2}$",
              title="(b) error field: one net / retraining")
        if r == 0:
            A.legend(frameon=False, fontsize=6.5, loc="lower left")
        for c, i in enumerate(show):
            A = ax[r, 2 + c]
            k = np.sqrt(np.trace(Vp[i]))
            for g in range(8):
                e = ellipse(E[i, g]) / k
                A.plot(e[0], e[1], color=C[0], lw=0.6, alpha=0.6)
            e = ellipse(E[i].mean(0)) / k
            A.plot(e[0], e[1], color=C[0], lw=2.0)
            e = ellipse(Vp[i]) / k
            A.plot(e[0], e[1], color=INK, lw=1.4, ls="--")
            A.set_aspect("equal")
            A.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xticks=[], yticks=[],
                  title=fr"(c) $\sigma/\sigma_{{\rm data}}$ = {s[i]:.2g}")
            A.text(0.02, 0.02, "black dashed: $V_p$\nblue: 8 single nets, thick = mean" if c == 0 and r == 0 else "",
                   transform=A.transAxes, fontsize=6, color=MUTED)
    fig.savefig(a.out, dpi=200)
    for n in ns:
        Cg, Cpool, E, Vp = z[f"{n}/C"], z[f"{n}/Cpool"], z[f"{n}/E"], z[f"{n}/Vp"]
        W = inv_sqrt_batch(Vp)
        gm = np.linalg.eigvalsh(np.einsum("sij,sjk,skl->sil", W, E.mean(1), W))
        ld = np.linalg.eigvalsh(Cg.mean(1) - Cpool)
        print(f"N={n}")
        for i in range(len(s)):
            print(f"  sig/sd {s[i]:7.3f}  Delta eig {ld[i, 0]:+.3e} {ld[i, 1]:+.3e}  Vp eig {np.linalg.eigvalsh(Vp[i])[1]:.3e} "
                  f"{np.linalg.eigvalsh(Vp[i])[0]:.3e}  gen-eig(mean E / Vp) {gm[i, 1]:.3f} {gm[i, 0]:.3f}")


def inv_sqrt_batch(M):
    return np.stack([inv_sqrt(m) for m in M])


if __name__ == "__main__":
    main()
