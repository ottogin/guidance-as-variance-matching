"""Build the static arena site from the study's page build.

  python3 user_study/arena/build.py                         # -> arena/public/index.html + arena/public/s/NNN.webp
  python3 user_study/arena/build.py --import-db DIR         # also write arena/import.sql from a JSON export of an earlier copy's votes/
  python3 user_study/arena/build.py --offline DIR           # an offline copy: DIR/index.html + DIR/s/, no server
  python3 user_study/arena/build.py --single FILE.html      # the same copy as one file, sprites inlined (supplementary/)

The offline copy needs no network: "Everyone" is the anonymised votes behind the paper (reproduce/data/
user_study_votes.json, people only), and a visitor's own votes stay in their browser ("Just me"). Sprites are
re-encoded smaller (--px, --quality: 384 px at q80 is about 70 MB, 320 px about 52 MB; the site's 480 px q90 is 172 MB).
--single inlines them as base64 data URIs (a third larger on disk, about the same once zipped).

Takes PROMPTS / KEY / ARMS / HIDE from results/elo7/page/index.html (the study's page, built
with seed 20261104), so sprite slots match the key and the votes already cast.
Hidden captions get no sprite. The SQL import keeps each imported rater's vote order as seq.
"""
import argparse
import hashlib
import hmac
import secrets
import datetime
import glob
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))        # the repository
VOTE_RE = re.compile(r"^(\d{1,3})\.([a-z])([a-z])([lrt])\.(\d{1,5})$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{ROOT}/results/elo7/page")
    ap.add_argument("--import-db", help="directory holding votes/<uid>.json exported from an earlier copy of the page")
    ap.add_argument("--renders", default=f"{ROOT}/results/elo7/renders", help="original JPEGs, for the agent API tiles")
    ap.add_argument("--no-tiles", action="store_true", help="skip the per-image tiles (they need Pillow)")
    ap.add_argument("--offline", metavar="DIR", help="write a self-contained offline copy to DIR instead of public/")
    ap.add_argument("--single", metavar="FILE", help="write the offline copy as one self-contained HTML file")
    ap.add_argument("--votes", default=f"{ROOT}/reproduce/data/user_study_votes.json", help="--offline: votes behind Everyone")
    ap.add_argument("--px", type=int, default=384, help="--offline: image side in pixels (the site's sprites are 480)")
    ap.add_argument("--quality", type=int, default=80, help="--offline: WebP quality of the re-encoded sprites")
    a = ap.parse_args()

    built = open(f"{a.src}/index.html").read()
    grab = lambda pat: re.search(pat, built, re.S).group(1)
    vals = {"/*PROMPTS*/[]": grab(r"const PROMPTS = (\[.*?\]);"), '/*KEY*/""': grab(r'const KEY = ("[^"]*");'),
            "/*ARMS*/{}": grab(r"const ARMS = (\{.*?\});\s"), "/*HIDE*/[]": grab(r"const HIDE = (\[[^\]]*\]);")}
    hide = set(json.loads(vals["/*HIDE*/[]"]))
    n = len(json.loads(vals["/*PROMPTS*/[]"]))
    shared = open(f"{HERE}/functions/_shared.js").read()
    assert hide == set(json.loads(re.search(r"new Set\((\[[^\]]*\])\)", shared).group(1))), "HIDE differs from functions/_shared.js"

    html = open(f"{HERE}/page.template.html").read()
    vals["/*OFFLINE*/null"] = json.dumps(offline_board(a.votes)) if a.offline or a.single else "null"
    for ph, v in vals.items():
        assert html.count(ph) == 1, ph
        html = html.replace(ph, v)
    if a.offline or a.single:
        return offline(a, html, n, hide)
    os.makedirs(f"{HERE}/public/s", exist_ok=True)
    open(f"{HERE}/public/index.html", "w").write(html)
    for f in glob.glob(f"{HERE}/public/s/*.webp"):
        os.remove(f)
    total = 0
    for p in range(n):
        if p in hide:
            continue
        src = f"{a.src}/s/{p:03d}.webp"
        shutil.copyfile(src, f"{HERE}/public/s/{p:03d}.webp")
        total += os.path.getsize(src)
    print(f"index.html + {n - len(hide)} sprites ({total / 1e6:.0f} MB), {len(hide)} hidden")
    if not a.no_tiles:
        tiles(a, json.loads(vals["/*PROMPTS*/[]"]), hide)

    if a.import_db:
        lines = ["-- votes imported from an earlier copy of the page's db (votes/<uid>)"]
        for f in sorted(glob.glob(f"{a.import_db}/votes/*.json")):
            d = json.load(open(f))
            rid = re.sub(r"[^A-Za-z0-9_-]", "", d["uid"])[:64]
            ts = int(datetime.datetime.fromisoformat(d["updated"].replace("Z", "+00:00")).timestamp() * 1000)  # last save; per-vote times were not kept
            lines.append(f"INSERT OR IGNORE INTO raters (id, created_at, last_seen, skips, source) "
                         f"VALUES ('{rid}', {ts}, {ts}, {int(d.get('skips', 0))}, 'claude-db');")
            k = 0
            for v in d.get("v", []):
                m = VOTE_RE.match(v) if isinstance(v, str) else None
                if not m or m[2] == m[3]:
                    continue
                lines.append(f"INSERT OR IGNORE INTO votes VALUES ('{rid}', {k}, {int(m[1])}, '{m[2]}', '{m[3]}', '{m[4]}', {int(m[5])}, {ts});")
                k += 1
            print(f"import: rater {rid[:8]}… {k} votes")
        open(f"{HERE}/import.sql", "w").write("\n".join(lines) + "\n")


def offline_board(path):
    """The anonymised export as the board the page reads: one entry per person, [arm, opp, wins, ties, losses] per
    pair with arm < opp, wins counted for arm (as functions/api/board.js returns them)."""
    d = json.load(open(path))
    col = {k: i for i, k in enumerate(d["fields"])}
    per = {}
    for v in d["votes"]:
        L, R, r = v[col["left"]], v[col["right"]], v[col["result"]]
        a, b = sorted((L, R))
        k = 1 if r == "t" else 0 if (r == "l") == (L == a) else 2
        per.setdefault(v[col["rater"]], {}).setdefault(a + b, [0, 0, 0])[k] += 1
    board = [{"me": False, "c": [[ab[0], ab[1], *w] for ab, w in sorted(c.items())]} for _, c in sorted(per.items())]
    assert sum(sum(x[2:]) for e in board for x in e["c"]) == d["snapshot_votes"] and len(board) == d["snapshot_raters"]
    return {"board": board, "votes": d["snapshot_votes"], "raters": d["snapshot_raters"]}


def offline(a, html, n, hide):
    """DIR/index.html (no web fonts, no requests) + DIR/s/NNN.webp re-encoded at --px / --quality + DIR/README.txt,
    or with --single one HTML file holding the sprites as data URIs."""
    import base64
    import io
    from PIL import Image
    html = re.sub(r'<link rel="(preconnect|stylesheet)" href="https://fonts\.googleapis\.com[^"]*">\n', "", html)
    assert "https://" not in html and "fetch(" in html, "offline page still links out"
    sprites = {}
    for p in range(n):
        if p in hide:
            continue
        im = Image.open(f"{a.src}/s/{p:03d}.webp").convert("RGB")
        if a.px != im.height:
            im = im.resize((im.width * a.px // im.height, a.px), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=a.quality, method=6)
        sprites[p] = buf.getvalue()
    total = sum(map(len, sprites.values()))
    if a.single:
        line = 'const sprite = p => `s/${String(p).padStart(3, "0")}.webp`;'
        assert html.count(line) == 1, "page.template.html no longer builds sprite URLs the way --single expects"
        inline = json.dumps({p: base64.b64encode(b).decode() for p, b in sprites.items()}, separators=(",", ":"))
        html = html.replace(line, f"const SPRITES = {inline};\n"
                                  'const sprite = p => "data:image/webp;base64," + SPRITES[p];')
        os.makedirs(os.path.dirname(os.path.abspath(a.single)), exist_ok=True)
        open(a.single, "w").write(html)
        print(f"single: {a.single}, {n - len(hide)} sprites at {a.px} px q{a.quality} ({total / 1e6:.0f} MB of WebP, "
              f"{os.path.getsize(a.single) / 1e6:.0f} MB file)")
        return
    os.makedirs(f"{a.offline}/s", exist_ok=True)
    open(f"{a.offline}/index.html", "w").write(html)
    for p, b in sprites.items():
        open(f"{a.offline}/s/{p:03d}.webp", "wb").write(b)
    open(f"{a.offline}/README.txt", "w").write(
        "SD3.5 Medium user study, offline copy\n\n"
        "Open index.html in a web browser (Chrome, Firefox or Safari). It needs no server and no network.\n\n"
        "Leaderboard > Everyone: the votes behind the paper's user-study table, from the people who took part online\n"
        "(the authors' and AI raters' votes are left out). Rate: you can judge pairs yourself, blind; your votes stay in\n"
        "this browser only and make up Leaderboard > Just me.\n")
    print(f"offline: {a.offline}/index.html + {n - len(hide)} sprites at {a.px} px q{a.quality} ({total / 1e6:.0f} MB)")


ARMDIR = dict(f="c1.6", d="c2.4", o="c4.14", s="trace", i="pub-interval", l="pub-linear", v="pub-tvcfg")


def tiles(a, prompts, hide):
    """One WebP per (caption, setting) for the agent API, at public/t/<opaque id>.webp. The id is an HMAC of
    "pos:arm" under a local salt (arena/.tile-salt, gitignored), so file names reveal nothing and stay stable across
    rebuilds. functions/_tiles.js (gitignored, never served) maps "pos:arm" -> id and holds the captions."""
    from PIL import Image
    salt_f = f"{HERE}/.tile-salt"
    if not os.path.exists(salt_f):
        open(salt_f, "w").write(secrets.token_hex(32))
    salt = bytes.fromhex(open(salt_f).read().strip())
    key = json.load(open(f"{a.src}/key.json"))["rows"]
    os.makedirs(f"{HERE}/public/t", exist_ok=True)
    m, made = {}, 0
    for row in key:
        p = row["pos"]
        if p in hide:
            continue
        for c in ARMDIR:
            tid = hmac.new(salt, f"{p}:{c}".encode(), hashlib.sha256).hexdigest()[:24]
            out = f"{HERE}/public/t/{tid}.webp"
            if not os.path.exists(out):
                Image.open(f"{a.renders}/{ARMDIR[c]}/{row['coco']:06d}.jpg").convert("RGB").save(out, "WEBP", quality=90, method=6)
                made += 1
            m[f"{p}:{c}"] = tid
    caps = {p: prompts[p]["t"] for p in range(len(prompts)) if p not in hide}
    open(f"{HERE}/functions/_tiles.js", "w").write(
        "// generated by arena/build.py: (caption position:arm) -> opaque tile id, and captions. Never served.\n"
        f"export const TILES = {json.dumps(m)};\nexport const CAPTIONS = {json.dumps(caps, ensure_ascii=False)};\n")
    print(f"tiles: {len(m)} ({made} new)")


if __name__ == "__main__":
    main()
