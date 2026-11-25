"""Build the EDM2 ImageNet-512 arena (static page + the data its Pages Functions read) from sampled renders.

  python3 user_study/edm2-arena/build.py      # -> public/index.html, public/s/NNN.webp, public/t/<id>.webp,
                                              #    functions/_arms.js, functions/_tiles.js (gitignored)

Rows are the seeds of sample.py's evaluation rows (seed r fixes both the noise and the class, as EDM2's
generate_images.py draws them), so every arm of a row shows the same class from the same initial noise. Each arm's
renders are RENDERS/<code>/<seed:05d>.png and the class of each seed is RENDERS/classes.json (a list of ints, seed
order), all fetched from the cluster:

    R=$VMG_RUNS/edm2-512/imagenet512        # on the cluster
    rsync -a cluster:$R/samples/pub/     results/edm2-arena/renders/n/
    rsync -a cluster:$R/samples/w5.6658/ results/edm2-arena/renders/o/
    rsync -a cluster:$R/classes-seeds0-99.json results/edm2-arena/renders/classes.json

Per row, one horizontal sprite s/NNN.webp holds every arm's image in a shuffled slot order (seeded, so rebuilds are
identical); KEY is those slot orders XOR'd with the page's SALT and base64-encoded, which the page decodes.

Adding arms (e.g. EDM2-XXL <- XS-uncond at NVIDIA's w = 1.20 and at our constant): sample them for the same seeds,
add them to ARMS and GROUPS below, and widen the arm CHECK of `votes` in migrations/ (a new migration rebuilding the
table if the database already exists). build.py refuses to build while the newest migration's CHECK differs from ARMS.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import random
import re
import secrets

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))        # the repository

# code -> arm. `name` is the leaderboard's full label, `short` its chart label; neither is shown before voting.
ARMS = {
    "n": dict(name="NVIDIA FID-tuned, w = 1.40", short="FID-tuned w=1.40", w=1.40, ours=False),
    "o": dict(name="Ours, variance-matched constant, w = 5.67", short="Ours w=5.67", w=5.67, ours=True),
    # EDM2-XXL <- XS-uncond CFG (generate_images.py edm2-img512-xxl-guid-fid), once sampled for the same seeds:
    # "x": dict(name="EDM2-XXL, NVIDIA FID-tuned, w = 1.20", short="XXL FID-tuned w=1.20", w=1.20, ours=False),
    # "y": dict(name="EDM2-XXL, ours, w = ...", short="XXL ours w=...", w=None, ours=True),
}
# leaderboard sections, every arm in exactly one; e.g. add dict(title="EDM2-XXL ...", codes="xy")
GROUPS = [dict(title="EDM2-S, classifier-free guidance", codes="no")]
CROSS_GROUP = True        # False: raters and agents only see pairs of arms from the same group
SEEDS = range(100)
SALT = "og-edm2"          # must equal SALT in page.template.html
ORDER_SEED = 20261123     # slot order of the arms in each row's sprite
PX, QUALITY = 480, 90     # sprite / tile side and WebP quality (the SD3.5 arena's settings)


def class_names():
    """ImageNet-1k class index -> a short readable name (torchvision's category list, the first WordNet synonym)."""
    from torchvision.models import ResNet50_Weights
    return ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def arm_pairs():
    codes = list(ARMS)
    group = {c: k for k, g in enumerate(GROUPS) for c in g["codes"]}
    return [a + b for i, a in enumerate(codes) for b in codes[i + 1:] if CROSS_GROUP or group[a] == group[b]]


def check_config():
    in_groups = "".join(g["codes"] for g in GROUPS)
    assert sorted(in_groups) == sorted(ARMS), f"GROUPS {in_groups!r} must hold every arm of {list(ARMS)} exactly once"
    assert all(re.fullmatch(r"[a-z]", c) for c in ARMS), "arm codes are single lower-case letters (VOTE_RE)"
    assert len(ARMS) >= 2 and arm_pairs(), "need at least one pair of arms"
    migs = sorted(f for f in os.listdir(f"{HERE}/migrations") if f.endswith(".sql"))
    last = None
    for f in migs:
        m = re.search(r"left_arm\s+IN \(([^)]*)\)", open(f"{HERE}/migrations/{f}").read())
        if m:
            last = (f, set(re.findall(r"'([a-z])'", m.group(1))))
    assert last and last[1] == set(ARMS), f"migrations/{last and last[0]} allows arms {last and sorted(last[1])}, ARMS has {sorted(ARMS)}"
    shared = open(f"{HERE}/functions/_shared.js").read()
    assert re.search(r"export const HIDE = new Set\(\[\]\);", shared), "functions/_shared.js HIDE must be empty (as the page's)"
    page = open(f"{HERE}/page.template.html").read()
    assert f'const SALT = "{SALT}";' in page, "SALT differs from page.template.html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default=f"{ROOT}/results/edm2-arena/renders")
    ap.add_argument("--no-tiles", action="store_true", help="skip the per-image tiles of the agent API")
    a = ap.parse_args()
    check_config()
    from PIL import Image

    cls = json.load(open(f"{a.renders}/classes.json"))
    names = class_names()
    seeds = list(SEEDS)
    assert len(cls) >= len(seeds), f"classes.json has {len(cls)} seeds, need {len(seeds)}"
    for c in ARMS:
        miss = [s for s in seeds if not os.path.exists(f"{a.renders}/{c}/{s:05d}.png")]
        assert not miss, f"arm {c}: {len(miss)} renders missing, e.g. {a.renders}/{c}/{miss[0]:05d}.png"

    prompts = [dict(i=s, c=int(cls[s]), t=names[int(cls[s])]) for s in seeds]
    rng = random.Random(ORDER_SEED)
    slots = []
    for _ in seeds:
        order = list(ARMS)
        rng.shuffle(order)
        slots.append("".join(order))
    plain = "".join(slots)
    key = base64.b64encode(bytes(ord(ch) ^ ord(SALT[i % len(SALT)]) for i, ch in enumerate(plain))).decode()

    arms_js = {c: dict(name=v["name"], short=v["short"], w=v["w"], ours=v["ours"]) for c, v in ARMS.items()}
    vals = {"/*PROMPTS*/[]": json.dumps(prompts, ensure_ascii=False), '/*KEY*/""': json.dumps(key),
            "/*ARMS*/{}": json.dumps(arms_js, ensure_ascii=False), "/*GROUPS*/[]": json.dumps(GROUPS, ensure_ascii=False),
            "/*HIDE*/[]": "[]", "/*PAIRS*/[]": json.dumps(arm_pairs())}
    html = open(f"{HERE}/page.template.html").read()
    for ph, v in vals.items():
        assert html.count(ph) == 1, ph
        html = html.replace(ph, v)
    os.makedirs(f"{HERE}/public/s", exist_ok=True)
    open(f"{HERE}/public/index.html", "w").write(html)

    for f in os.listdir(f"{HERE}/public/s"):
        os.remove(f"{HERE}/public/s/{f}")
    total = 0
    for p, (s, order) in enumerate(zip(seeds, slots)):
        sheet = Image.new("RGB", (PX * len(order), PX))
        for k, c in enumerate(order):
            sheet.paste(Image.open(f"{a.renders}/{c}/{s:05d}.png").convert("RGB").resize((PX, PX), Image.LANCZOS), (k * PX, 0))
        out = f"{HERE}/public/s/{p:03d}.webp"
        sheet.save(out, "WEBP", quality=QUALITY, method=6)
        total += os.path.getsize(out)
    print(f"index.html + {len(seeds)} sprites of {len(ARMS)} arms ({total / 1e6:.1f} MB)")

    open(f"{HERE}/functions/_arms.js", "w").write(
        "// generated by edm2-arena/build.py from its ARMS: arm codes, display names, allowed pairs, number of rows.\n"
        f"export const CODES = {json.dumps(''.join(ARMS))};\n"
        f"export const NAMES = {json.dumps({c: v['name'] for c, v in ARMS.items()}, ensure_ascii=False)};\n"
        f"export const PAIRS = {json.dumps(arm_pairs())};\n"
        f"export const N_ROWS = {len(seeds)};\n")
    if not a.no_tiles:
        tiles(a, seeds, prompts)


def tiles(a, seeds, prompts):
    """One WebP per (row, arm) for the agent API, at public/t/<opaque id>.webp. The id is an HMAC of "pos:arm" under a
    local salt (.tile-salt, gitignored), so file names reveal nothing and stay stable across rebuilds.
    functions/_tiles.js (gitignored, never served) maps "pos:arm" -> id and holds the class captions."""
    from PIL import Image
    salt_f = f"{HERE}/.tile-salt"
    if not os.path.exists(salt_f):
        open(salt_f, "w").write(secrets.token_hex(32))
    salt = bytes.fromhex(open(salt_f).read().strip())
    os.makedirs(f"{HERE}/public/t", exist_ok=True)
    m, made = {}, 0
    for p, s in enumerate(seeds):
        for c in ARMS:
            tid = hmac.new(salt, f"{p}:{c}".encode(), hashlib.sha256).hexdigest()[:24]
            out = f"{HERE}/public/t/{tid}.webp"
            if not os.path.exists(out):
                Image.open(f"{a.renders}/{c}/{s:05d}.png").convert("RGB").resize((PX, PX), Image.LANCZOS).save(
                    out, "WEBP", quality=QUALITY, method=6)
                made += 1
            m[f"{p}:{c}"] = tid
    keep = {f"{t}.webp" for t in m.values()}
    for f in os.listdir(f"{HERE}/public/t"):       # tiles of arms or rows no longer in the study
        if f not in keep:
            os.remove(f"{HERE}/public/t/{f}")
    caps = {p: f"ImageNet class: {prompts[p]['t']}" for p in range(len(seeds))}
    open(f"{HERE}/functions/_tiles.js", "w").write(
        "// generated by edm2-arena/build.py: (row position:arm) -> opaque tile id, and class captions. Never served.\n"
        f"export const TILES = {json.dumps(m)};\nexport const CAPTIONS = {json.dumps(caps, ensure_ascii=False)};\n")
    print(f"tiles: {len(m)} ({made} new)")


if __name__ == "__main__":
    main()
