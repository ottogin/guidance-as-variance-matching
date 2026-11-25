#!/usr/bin/env python
"""tab:user-study{,-h2h,-fable} from the anonymised votes (reproduce/data/user_study_votes.json).

    python reproduce/tables/tab_user_study.py --out paper-tables/ [--rater people|fable] [--all] [--boot 1000]

Writes user-study-elo.tex (two side-by-side tabulars: constants | schedules, setting, w, Elo +- 1.96 bootstrap
sd) and user-study-h2h.tex (how often the row setting beat the column setting, ties half, games in brackets).
Rating: Bradley-Terry, a tie = half a win to each side, one prior tie per pair; Elo scale 1000 + 400 log10(p)
with mean 1000. Intervals: bootstrap over raters, then votes within each rater. By default only the votes behind
the paper (the file's snapshot: 1,458 votes from 41 people, no authors, no AI raters); --all uses every vote.
--rater fable writes the same two tables, suffixed -fable, from Claude Fable 5.1's 2,500 votes: one rater, so the
bootstrap resamples its votes.
"""
import argparse
import json
import math
import random
from pathlib import Path

VOTES = Path(__file__).resolve().parents[1] / "data" / "user_study_votes.json"
CODES = list("fdosilv")
LABEL = {"f": ("FID", "1.6"), "d": (r"FD$_\mathrm{DINOv2}$", "2.4"), "o": ("Ours", "4.14"),
         "s": ("Ours", r"$w(\sigma)$"), "i": ("Interval", "4.5"), "l": ("Linear", "4.5"), "v": ("TV-CFG", "4.5")}
SHORT = {"f": "1.6", "d": "2.4", "o": "Ours 4.14", "s": r"Ours $w(\sigma)$", "i": "Int.", "l": "Lin.", "v": "TV"}
OURS = {"o", "s"}
BLOCKS = [("Constant weight", "o", "df"), ("Schedule", "s", "vil")]     # (title, ours, the others)


def wins(votes):
    K = len(CODES)
    W = [[0.0] * K for _ in range(K)]
    for L, R, r in votes:
        if r == "l":
            W[L][R] += 1
        elif r == "r":
            W[R][L] += 1
        else:
            W[L][R] += .5
            W[R][L] += .5
    return W


def bt(W):
    K = len(W)
    p = [1.0] * K
    for _ in range(1000):
        delta = 0.0
        for i in range(K):
            num = sum(W[i][j] + .5 for j in range(K) if j != i)
            den = sum((W[i][j] + W[j][i] + 1) / (p[i] + p[j]) for j in range(K) if j != i)
            new = num / den
            delta = max(delta, abs(math.log(new / p[i])))
            p[i] = new
        g = math.exp(sum(map(math.log, p)) / K)
        p = [x / g for x in p]
        if delta < 1e-10:
            break
    return [1000 + 400 * math.log10(x) for x in p]


def row(c, elo, pm):
    lab, w = LABEL[c]
    k = CODES.index(c)
    if c in OURS:
        return rf"\textbf{{{lab}}} & {{\boldmath{w}}} & \textbf{{{elo[k]:.0f}}} & {{\boldmath$\pm$}} & \textbf{{{pm[k]:.0f}}} \\"
    return rf"{lab} & {w} & {elo[k]:.0f} & $\pm$ & {pm[k]:.0f} \\"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--votes", default=str(VOTES))
    ap.add_argument("--rater", choices=["people", "fable"], default="people")
    ap.add_argument("--all", action="store_true", help="every vote, not only the paper's snapshot")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    d = json.load(open(a.votes))
    if a.rater == "fable":
        d = d["fable"]
    sfx = "-fable" if a.rater == "fable" else ""
    votes = d["votes"] if a.all else [v for v in d["votes"] if v[8] <= d["snapshot_ms"]]
    people = {}
    for rater, _, _, _, L, R, r, _, _ in votes:            # already in (rater, browser, seq) order
        people.setdefault(rater, []).append((CODES.index(L), CODES.index(R), r))
    groups = list(people.values())
    allv = [v for g in groups for v in g]
    W = wins(allv)
    elo = bt(W)
    rng = random.Random(0)
    boots = [[] for _ in CODES]
    for _ in range(a.boot):
        sample = []
        for _ in groups:
            g = groups[rng.randrange(len(groups))]
            sample += [g[rng.randrange(len(g))] for _ in g]
        for k, e in enumerate(bt(wins(sample))):
            boots[k].append(e)
    pm = [1.96 * math.sqrt(sum((x - sum(b) / len(b)) ** 2 for x in b) / (len(b) - 1)) for b in boots]
    who = "Claude Fable 5.1" if a.rater == "fable" else f"{len(groups)} raters"
    head = f"% written by reproduce/tables/tab_user_study.py --rater {a.rater}: {len(allv)} votes, {who}\n"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tex = []
    for bi, (title, ours, others) in enumerate(BLOCKS):
        rest = sorted(others, key=lambda c: -elo[CODES.index(c)])
        tex += [r"\begin{tabular}[t]{@{}llr@{\,}c@{\,}l@{}}", r"\toprule",
                rf"\multicolumn{{5}}{{@{{}}c@{{}}}}{{\emph{{{title}}}}} \\", r"\cmidrule{1-5}",
                r"Setting & $w$ & \multicolumn{3}{c@{}}{Elo} \\", r"\midrule", row(ours, elo, pm)]
        if bi == 0:
            tex.append(r"\multicolumn{5}{@{}l}{\emph{Grid search over:}} \\")
            tex += [r"\quad " + row(c, elo, pm) for c in rest]
        else:
            tex += [row(c, elo, pm) for c in rest]
        tex += [r"\bottomrule", r"\end{tabular}" + (r"\hspace{14pt}%" if bi == 0 else "")]
    (out / f"user-study-elo{sfx}.tex").write_text(head + "\n".join(tex) + "\n")

    order = [CODES.index(c) for _, ours, others in BLOCKS
             for c in [ours] + sorted(others, key=lambda c: -elo[CODES.index(c)])]
    n_const = 1 + len(BLOCKS[0][2])
    G = [[0] * len(CODES) for _ in CODES]
    for L, R, _ in allv:
        G[L][R] += 1
        G[R][L] += 1
    h = [r"\begin{tabular}{l" + "c" * n_const + "@{\\hspace{8pt}}" + "c" * (len(CODES) - n_const) + "}", r"\toprule",
         r"Row vs.\ column & " + " & ".join(SHORT[CODES[k]] for k in order) + r" \\", r"\midrule"]
    for i in order:
        cells = []
        for j in order:
            if i == j:
                cells.append("--")
            elif not G[i][j]:
                cells.append(r"\color{black!50}n/a")
            else:
                s = W[i][j] / G[i][j]
                a_ = round(min(1, abs(s - .5) / .3) * 75)              # full shade at 80/20
                shade = f"\\cellcolor{{{'vmbetter' if s > .5 else 'vmworse'}!{a_}}}" if a_ else ""
                cells.append(f"{shade}{100 * s:.0f}\\,{{\\scriptsize({G[i][j]})}}")
        if i == order[n_const]:
            h.append(r"\midrule")
        h.append(f"{SHORT[CODES[i]]} & " + " & ".join(cells) + r" \\")
    h += [r"\bottomrule", r"\end{tabular}"]
    (out / f"user-study-h2h{sfx}.tex").write_text(head + "\n".join(h) + "\n")
    print(f"{a.rater}: {len(allv)} votes, {who}")
    for k in order:
        print(f"  {CODES[k]} {LABEL[CODES[k]][0]:22s} {elo[k]:6.0f} +- {pm[k]:3.0f}")


if __name__ == "__main__":
    main()
