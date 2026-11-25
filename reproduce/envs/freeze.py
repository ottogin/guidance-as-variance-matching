#!/usr/bin/env python
"""Print every installed distribution as name==version (works in environments without pip, e.g. uv venvs).

    python reproduce/envs/freeze.py > reproduce/envs/<env>.lock.txt

A venv layered over another (--system-site-packages) can see two copies of a package; the first on sys.path is
the one imported, so only that one is listed. A distribution installed from git or a URL (PEP 610 direct_url.json)
is printed as `name @ <url>` so that the lock installs it from there; one installed from a local path is printed
commented out, since no other machine has it. The header comment and the torch index line
(--extra-index-url https://download.pytorch.org/whl/cu124 for +cu124 wheels) are added by hand.
"""
import importlib.metadata as md
import json
import re


def line(d):
    name, pin = d.metadata["Name"], f"{d.metadata['Name']}=={d.version}"
    try:
        u = json.loads(d.read_text("direct_url.json") or "null")
    except (ValueError, OSError):
        u = None
    if not u:
        return pin
    if u["url"].startswith("file:"):
        return f"# {pin}  (installed from a local path: {u['url']})"
    if "vcs_info" in u:
        return f"{name} @ {u['vcs_info']['vcs']}+{u['url']}@{u['vcs_info']['commit_id']}"
    return f"{name} @ {u['url']}"


seen = {}
for d in md.distributions():                     # in sys.path order
    seen.setdefault(re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(), line(d))
print("\n".join(sorted(seen.values(), key=lambda s: s.lstrip("# ").lower())))
