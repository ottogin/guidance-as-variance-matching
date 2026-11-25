#!/usr/bin/env python
"""Export the user-study votes from the arena database once, anonymised, into user_study_votes.json.

    python reproduce/data/export_votes.py                       # reads the live D1 database (npx wrangler, logged in)
    python reproduce/data/export_votes.py --raw votes-raw.json  # or a saved copy of the same query

The arena (user_study/arena, Cloudflare Pages + D1) keyed votes by browser and grouped browsers into people by the
name they typed; AI raters (the agent API, or imported) group by model name under
`agent:`. Only people go into `votes`, and the authors' votes are left out. Of the AI raters only Claude Fable 5.1 is
kept, in `fable` (it registered as `fabel5.1`, 30 API keys); every other AI rater is dropped.

Names never leave this script: people become rater 1..N in the order of their sorted person keys, and each
person's browsers (each Fable API key) become browser 1..M in the order of their ids, so the votes keep the order
the table script reads them in (person, browser, seq), which its bootstrap depends on. Each vote keeps its caption
position, the two arms, the result, the response time (none for Fable) and the time it was cast. `snapshot_ms` is
the cutoff of the votes behind the paper's tables: 1,458 votes from 41 people (the last cast 2026-11-11 15:48:47
UTC), and Fable's 2,500.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("user_study_votes.json")
QUERY = ("SELECT p.person AS person, v.rater AS browser, v.seq AS seq, v.pos AS pos, v.left_arm AS L, "
         "v.right_arm AS R, v.result AS r, v.rt_ds AS rt_ds, v.created_at AS t "
         "FROM votes v JOIN rater_person p ON p.id = v.rater ORDER BY p.person, v.rater, v.seq")
SNAPSHOT_MS = 1794422700000          # 2026-11-11 18:45:00 UTC: the people's votes behind the paper (last 15:48:47)
FABLE_SNAPSHOT_MS = 1794423600000    # 2026-11-11 19:00:00 UTC: all 2,500 of Fable's votes (last 18:54:53)
# the authors' person keys ("name:<typed name>"), given at export time so that no name is in the repository
AUTHORS = set(filter(None, os.environ.get("VMG_AUTHOR_KEYS", "").split(",")))
FABLE = "agent:fabel5.1"
ARMS = {"f": "FID-tuned constant 1.6", "d": "FD_DINOv2-tuned constant 2.4", "o": "ours, constant 4.14",
        "s": "ours, schedule w(sigma)", "i": "interval guidance at 4.5", "l": "linear schedule at 4.5",
        "v": "TV-CFG at 4.5"}
FIELDS = ["rater", "browser", "seq", "caption_pos", "left", "right", "result", "rt_deciseconds", "unix_ms"]


def fetch():
    for _ in range(4):               # the D1 API sometimes answers 7403 once, then works
        out = subprocess.run(["npx", "wrangler", "d1", "execute", "arena", "--remote", "--json", "--command", QUERY],
                             cwd=REPO / "user_study" / "arena", capture_output=True, text=True).stdout
        try:
            return json.loads(out)[0]["results"]
        except (ValueError, KeyError, IndexError):
            continue
    sys.exit("could not read the arena database: is `npx wrangler login` done?")


def anonymise(rows):
    rows = sorted(rows, key=lambda x: (x["person"], x["browser"], x["seq"]))
    person = {p: i + 1 for i, p in enumerate(sorted({x["person"] for x in rows}))}
    browser = {}
    for x in rows:
        b = browser.setdefault(x["person"], {})
        b.setdefault(x["browser"], len(b) + 1)
    return [[person[x["person"]], browser[x["person"]][x["browser"]], x["seq"], x["pos"], x["L"], x["R"], x["r"],
             x["rt_ds"], x["t"]] for x in rows]


def block(votes, snapshot_ms):
    snap = [v for v in votes if v[8] <= snapshot_ms]
    return dict(snapshot_ms=snapshot_ms, snapshot_votes=len(snap), snapshot_raters=len({v[0] for v in snap}),
                votes=votes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", help="a saved result of QUERY (wrangler --json output) instead of the live database")
    a = ap.parse_args()
    rows = json.load(open(a.raw))[0]["results"] if a.raw else fetch()
    people = anonymise([x for x in rows if not x["person"].startswith("agent:") and x["person"] not in AUTHORS])
    fable = anonymise([x for x in rows if x["person"] == FABLE])
    out = dict(study="SD 3.5 Medium, seven settings, 500 held-out COCO captions (5,705-6,204), pairwise",
               arms=ARMS, results={"l": "left better", "r": "right better", "t": "tie"}, fields=FIELDS,
               raters="people only; the authors' votes and every AI rater's are left out",
               **block(people, SNAPSHOT_MS),
               fable=dict(model="Claude Fable 5.1 through the agent API (registered as 'fabel5.1'); one rater, "
                                "browser = API key", **block(fable, FABLE_SNAPSHOT_MS)))
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {OUT}: {len(people)} votes from {len({v[0] for v in people})} people, snapshot "
          f"{out['snapshot_votes']} votes from {out['snapshot_raters']}; Fable {len(fable)} votes, snapshot "
          f"{out['fable']['snapshot_votes']}")


if __name__ == "__main__":
    main()
