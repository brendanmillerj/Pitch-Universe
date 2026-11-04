"""Publish the built atlas to the Hugging Face Space as a standalone page.

analysis/pitch_atlas.html is a page fragment (no doctype, charset or viewport):
the claude.ai artifact host supplies those. A static Space serves the file as-is
with a bare `text/html` content type, so without them browsers fall back to
windows-1252 (UTF-8 "·" renders as "Â·"), render in quirks mode, and lay the
page out at desktop width on phones. This adds the document head and uploads.

Why the data travels in separate files: the Hub stores any file over 10 MB as
a Git LFS object, and the static server then answers with a redirect to a CDN
instead of the page -- browsers offer a download of index.html rather than
rendering it (hit on 2026-11-03 when Triple-A took the inline page to 12 MB).
So the embedded JSON is lifted out of the page into data/atlas-<hash>-<n>.js
chunks, each well under the limit, loaded by plain <script> tags before the
page script; the page reads `window.__ATLAS_DATA__` when its inline literal is
null. analysis/pitch_atlas.html itself stays self-contained for local viewing.

Usage (after `python scripts/atlas_build.py ...`; needs `huggingface-cli login`):
  python scripts/deploy_space.py              # upload to the Space
  python scripts/deploy_space.py --dry-run    # write the files locally, upload nothing
"""
import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "analysis" / "pitch_atlas.html"
SPACE = "brmiller/Pitch-Universe"
HEAD = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
)
HUB_PLAIN_LIMIT = 10_000_000   # bytes; larger files become LFS objects and are no longer served inline
CHUNK_BUDGET = 8_000_000       # bytes per data file, with headroom under the limit
DATA_RE = re.compile(r"const D = (\{.*?\}) \|\| window\.__ATLAS_DATA__;", re.S)
PAGE_SCRIPT = '<script>\n"use strict";'


def standalone(fragment: str) -> str:
    if fragment.lstrip().lower().startswith("<!doctype"):
        return fragment  # already a full document
    return HEAD + fragment


def split_data(page: str) -> tuple[str, dict[str, str]]:
    """Lift the inline JSON out of the page into data/*.js chunks (each under CHUNK_BUDGET).

    Returns the page with the literal replaced by null plus the script tags, and
    {relative path: file text}. Chunks are named by content hash so a redeploy
    never serves a stale cached data file with a new page.
    """
    m = DATA_RE.search(page)
    if not m:
        raise SystemExit("could not find the inline data literal (`const D = {...} || window.__ATLAS_DATA__;`)")
    data = json.loads(m.group(1))
    parts = {k: json.dumps(v, separators=(",", ":"), allow_nan=False) for k, v in data.items()}
    for k, v in parts.items():
        if len(v.encode()) > CHUNK_BUDGET:
            raise SystemExit(f"data key {k!r} alone is {len(v.encode()) / 1e6:.1f} MB; split it in the build first")

    chunks, cur, size = [], {}, 0
    for k, v in parts.items():   # keep the build's key order; greedy fill
        if cur and size + len(v.encode()) > CHUNK_BUDGET:
            chunks.append(cur); cur, size = {}, 0
        cur[k] = v; size += len(v.encode())
    if cur:
        chunks.append(cur)

    tag = hashlib.sha256(m.group(1).encode()).hexdigest()[:10]
    files, tags = {}, []
    for n, chunk in enumerate(chunks, 1):
        body = "Object.assign(window.__ATLAS_DATA__ = window.__ATLAS_DATA__ || {}, {" \
            + ",".join(f"{json.dumps(k)}:{v}" for k, v in chunk.items()) + "});\n"
        path = f"data/atlas-{tag}-{n}.js"
        files[path] = body
        tags.append(f'<script src="{path}"></script>')
    page = page[: m.start(1)] + "null" + page[m.end(1):]
    assert page.count(PAGE_SCRIPT) == 1, "page script tag not found exactly once"
    page = page.replace(PAGE_SCRIPT, "\n".join(tags) + "\n" + PAGE_SCRIPT)
    return page, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", help="write the deployable folder here (default: a temp dir)")
    a = ap.parse_args()

    page, files = split_data(standalone(SRC.read_text(encoding="utf-8")))
    files["index.html"] = page
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="atlas-space-"))
    if (out / "data").exists():
        shutil.rmtree(out / "data")
    for rel, text in files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        n = p.stat().st_size
        print(f"  {rel:<28} {n / 1e6:5.1f} MB")
        if n >= HUB_PLAIN_LIMIT:
            raise SystemExit(f"{rel} is {n / 1e6:.1f} MB: the Hub would store it as LFS and the Space would not serve it inline")
    print(f"deployable folder: {out}")
    if a.dry_run:
        return

    from huggingface_hub import HfApi
    c = HfApi().upload_folder(folder_path=str(out), repo_id=SPACE, repo_type="space",
                              commit_message="Update atlas page", delete_patterns=["data/*"])
    print(f"uploaded: {c.commit_url}")


if __name__ == "__main__":
    main()
