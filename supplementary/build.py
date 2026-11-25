#!/usr/bin/env python3
"""Assemble the CVPR supplementary zip from a clone of this repository and the paper's built appendix.

    python3 supplementary/build.py --appendix ../paper/supp.pdf            # -> supplementary.zip
    python3 supplementary/build.py --appendix supp.pdf --forbid 'Name1|Name2|Institute'   # also fail on these
    python3 supplementary/build.py --appendix supp.pdf --stage DIR      # the same files, unzipped, to read first

The zip holds
    README.md          supplementary/README.md: what each part is, for reviewers
    appendix.pdf       --appendix, the paper repository's supp.pdf (build it first; it is not kept here)
    user-study.html    supplementary/user-study.html: the user study's offline copy, one file
    sample_examples/   samples/ (Pub. vs Ours for every model, index.html views them)
    code/              every other tracked file of this repository at HEAD (git archive), without samples/ and
                       supplementary/

Only committed files go in (HEAD, not the working tree), so commit first. Checks before writing: no page outside code/ loads
anything over the network (CVPR: "All content must be self-contained"), the PDF names no author in its metadata,
nothing matches --forbid, and the zip fits --limit-mb (default 100; the guidelines state no limit, so check the
upload form). Standard library only.

user-study.html is built once from the study's local page (not in this repository):
    python3 user_study/arena/build.py --single supplementary/user-study.html --px 256 --quality 60
"""
import argparse
import io
import os
import re
import subprocess
import sys
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LINK = re.compile(rb"""(?:src|href)\s*=\s*["']?(?:https?:)?//|url\(\s*["']?(?:https?:)?//|@import\s+["']?(?:https?:)?//|https?://fonts\.""", re.I)

BASE64 = re.compile(rb"[A-Za-z0-9+/=]{200,}")


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True, capture_output=True).stdout


def tracked(rev):
    """{path: bytes} of every regular file at rev."""
    out = {}
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", rev)), mode="r:") as t:
        for m in t:
            if m.isfile():
                out[m.name] = t.extractfile(m).read()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--appendix", required=True, help="the built appendix, the paper repository's supp.pdf")
    ap.add_argument("--out", default=os.path.join(REPO, "supplementary.zip"))
    ap.add_argument("--rev", default="HEAD", help="the commit to take the repository's files from")
    ap.add_argument("--limit-mb", type=float, default=100.0, help="fail if the zip is larger (MB = 10^6 bytes)")
    ap.add_argument("--stage", metavar="DIR", help="write the files to DIR (which must not exist) instead of a zip")
    ap.add_argument("--forbid", action="append", default=[],
                    help="regex (case-insensitive) that must not occur in any text file of the zip, e.g. author names")
    a = ap.parse_args()

    files = tracked(a.rev)
    dirty = [l[3:] for l in git("status", "--porcelain", "--untracked-files=no").decode().splitlines() if l.strip()]
    if dirty:
        print(f"note: {len(dirty)} modified tracked files are left out (the zip takes {a.rev}): {', '.join(dirty[:5])}")
    pdf = open(a.appendix, "rb").read()
    assert pdf.startswith(b"%PDF"), f"{a.appendix} is not a PDF"

    entries = {"README.md": files["supplementary/README.md"], "appendix.pdf": pdf,
               "user-study.html": files["supplementary/user-study.html"]}
    for p, b in sorted(files.items()):
        if p.startswith("samples/"):
            entries["sample_examples/" + p[len("samples/"):]] = b
        elif not p.startswith("supplementary/"):
            entries["code/" + p] = b

    bad = []
    for p, b in entries.items():            # the pages a reviewer opens; code/ is source (the arena app links fonts)
        if p.endswith((".html", ".htm")) and not p.startswith("code/") and LINK.search(b):
            bad.append(f"{p}: loads something over the network ({LINK.search(b).group(0)[:40]!r})")
    m = re.search(rb"/Author\s*\(([^)]*)\)", pdf)
    if m and m.group(1).strip():
        bad.append(f"appendix.pdf: metadata names an author ({m.group(1)[:40]!r})")
    if a.forbid:
        pat = re.compile("|".join(f"(?:{f})" for f in a.forbid).encode(), re.I)
        for p, b in entries.items():
            if p.endswith((".pdf", ".jpg", ".png", ".webp", ".mp3", ".mp4", ".wav", ".gif", ".npz", ".npy", ".pt")):
                continue
            text = BASE64.sub(b" ", b)          # inlined images would match short patterns by chance
            if pat.search(text):
                bad.append(f"{p}: matches --forbid ({pat.search(text).group(0)[:40]!r})")
    if bad:
        sys.exit("not written:\n  " + "\n  ".join(bad))

    if a.stage:
        os.makedirs(a.stage)
        for p, b in entries.items():
            os.makedirs(os.path.join(a.stage, os.path.dirname(p)), exist_ok=True)
            open(os.path.join(a.stage, p), "wb").write(b)
        print(f"staged {len(entries)} files in {a.stage} ({sum(map(len, entries.values())) / 1e6:.1f} MB unzipped)")
        return

    tmp = a.out + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, b in entries.items():
            info = zipfile.ZipInfo(p, date_time=(2026, 11, 23, 0, 0, 0))
            info.external_attr = 0o644 << 16                 # rw-r--r-- when unzipped
            z.writestr(info, b, compress_type=zipfile.ZIP_DEFLATED)
    size = os.path.getsize(tmp)
    with zipfile.ZipFile(tmp) as z:
        part = {}
        for i in z.infolist():
            k = i.filename.split("/")[0]
            part[k] = part.get(k, 0) + i.compress_size
    for k, v in sorted(part.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v / 1e6:6.1f} MB")
    print(f"  {'total':<18} {size / 1e6:6.1f} MB of {a.limit_mb:g} ({len(entries)} files, {a.rev} = "
          f"{git('rev-parse', '--short', a.rev).decode().strip()})")
    if size > a.limit_mb * 1e6:
        os.remove(tmp)
        sys.exit(f"not written: {size / 1e6:.1f} MB is over --limit-mb {a.limit_mb:g}")
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
