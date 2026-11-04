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
So the embedded JSON is lifted out of the page into data files:

  data/atlas-<hash>-main.json.gz   MLB rows and everything shared (fetched on load)
  data/atlas-<hash>-aaa.json.gz    Triple-A rows, its comps, movement and flight parts,
                                   fetched only when the viewer first switches to Triple-A

The static server sends files as they are (no Content-Encoding), so the page
gunzips them itself with DecompressionStream; a plain .json copy of each sits
beside it for browsers without one. The page script is shipped inert
(type="text/plain") and started by a small loader once the main data is in
`window.__ATLAS_DATA__`; the Triple-A part is announced to it as
`D.lazy = {"aaa": [base path]}`. analysis/pitch_atlas.html itself stays
self-contained (every level inline) for local viewing.

Usage (after `python scripts/atlas_build.py ...`; needs `huggingface-cli login`):
  python scripts/deploy_space.py              # upload to the Space
  python scripts/deploy_space.py --dry-run    # write the files locally, upload nothing
"""
import argparse
import base64
import gzip
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
DATA_RE = re.compile(r"const D = (\{.*?\}) \|\| window\.__ATLAS_DATA__;", re.S)
PAGE_SCRIPT = '<script>\n"use strict";'


def standalone(fragment: str) -> str:
    if fragment.lstrip().lower().startswith("<!doctype"):
        return fragment  # already a full document
    return HEAD + fragment


def split_level(data: dict, level: str = "aaa") -> tuple[dict, dict]:
    """Cut `level`'s rows out of the data. The build writes every MLB row before any row of another level
    (types, pitchers, and the per-row comps, movement and flight blobs follow the types' order), and MLB
    comps only point at MLB rows, so the cut is a slice. Returns (main, part)."""
    lv = data["types"]["level"]
    if level not in lv:
        return data, {}
    k = lv.index(level)
    assert all(x == level for x in lv[k:]), "rows of the lazy level are not contiguous at the end"
    kp = data["pitchers"]["level"].index(level)
    assert all(x == level for x in data["pitchers"]["level"][kp:])
    assert all(c < k for row in data["tcomps"][:k] for c, _ in row), "an MLB comp points past the MLB rows"
    main, part = dict(data), {}
    main["types"] = {c: v[:k] for c, v in data["types"].items()}
    part["types"] = {c: v[k:] for c, v in data["types"].items()}
    main["pitchers"] = {c: v[:kp] for c, v in data["pitchers"].items()}
    part["pitchers"] = {c: v[kp:] for c, v in data["pitchers"].items()}
    main["tcomps"], part["tcomps"] = data["tcomps"][:k], data["tcomps"][k:]
    if "tcomps_aaa" in data:   # its indices point at Triple-A rows, which only exist once the part is loaded
        part["tcomps_aaa"] = main.pop("tcomps_aaa")
    # movement: counts per row, then every x then every z (int8)
    mv = data["movement"]; raw = base64.b64decode(mv["b64"]); n = len(raw) // 2
    x, z = raw[:n], raw[n:]; off = sum(mv["counts"][:k])
    main["movement"] = {"counts": mv["counts"][:k], "b64": base64.b64encode(x[:off] + z[:off]).decode()}
    part["movement"] = {"counts": mv["counts"][k:], "b64": base64.b64encode(x[off:] + z[off:]).decode()}
    # flight: len(cols) int16 per row
    fl = data["flight"]; raw = base64.b64decode(fl["b64"]); cut = k * len(fl["cols"]) * 2
    assert len(raw) == len(lv) * len(fl["cols"]) * 2
    main["flight"] = {**fl, "b64": base64.b64encode(raw[:cut]).decode()}
    part["flight"] = {**{c: fl[c] for c in ("cols", "scale", "na")}, "b64": base64.b64encode(raw[cut:]).decode()}
    return main, part


LOADER = """<div id="atlasLoading" style="position:fixed;inset:0;display:grid;place-items:center;font:500 17px 'Libre Franklin',Arial,sans-serif;color:#3d4756;background:#f0f3f7;z-index:100">Loading the pitch map…</div>
<script>
(function () {
  var gz = typeof DecompressionStream === "function", ext = gz ? ".json.gz" : ".json";
  function get(u) {
    return fetch(u + ext).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status + " for " + u + ext); return r.arrayBuffer(); })
      .then(function (b) {
        var a = new Uint8Array(b);
        if (a[0] === 0x1f && a[1] === 0x8b) return new Response(new Blob([a]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
        return new TextDecoder().decode(a);   // plain, or already un-gzipped by the server
      }).then(JSON.parse);
  }
  get(__MAIN__).then(function (d) {
    window.__ATLAS_DATA__ = d;
    var s = document.createElement("script"); s.textContent = document.getElementById("atlasMain").textContent;
    document.body.appendChild(s);
    var l = document.getElementById("atlasLoading"); if (l) l.remove();
  }).catch(function (e) {
    var l = document.getElementById("atlasLoading"); if (l) l.textContent = "The pitch map could not load (" + e.message + "). Try reloading the page.";
  });
})();
</script>
"""


def split_data(page: str) -> tuple[str, dict[str, bytes]]:
    """Lift the inline JSON out of the page into gzipped data files (see the module docstring).

    Returns the page (literal replaced by null, script made inert, loader added) and {relative path: bytes}.
    Files are named by content hash so a redeploy never serves a stale cached data file with a new page.
    """
    m = DATA_RE.search(page)
    if not m:
        raise SystemExit("could not find the inline data literal (`const D = {...} || window.__ATLAS_DATA__;`)")
    data = json.loads(m.group(1))
    tag = hashlib.sha256(m.group(1).encode()).hexdigest()[:10]
    main, part = split_level(data, "aaa")
    files = {}
    if part:
        base = f"data/atlas-{tag}-aaa"
        main["lazy"] = {"aaa": [base]}
        files[base] = part
    files = {f"data/atlas-{tag}-main": main, **files}
    out = {}
    for base, obj in files.items():
        text = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()
        out[base + ".json"] = text
        out[base + ".json.gz"] = gzip.compress(text, compresslevel=9, mtime=0)
    page = page[: m.start(1)] + "null" + page[m.end(1):]
    assert page.count(PAGE_SCRIPT) == 1, "page script tag not found exactly once"
    page = page.replace(PAGE_SCRIPT, LOADER.replace("__MAIN__", json.dumps(f"data/atlas-{tag}-main"))
                        + PAGE_SCRIPT.replace("<script>", '<script type="text/plain" id="atlasMain">'))
    return page, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", help="write the deployable folder here (default: a temp dir)")
    a = ap.parse_args()

    page, files = split_data(standalone(SRC.read_text(encoding="utf-8")))
    files["index.html"] = page.encode("utf-8")
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="atlas-space-"))
    if (out / "data").exists():
        shutil.rmtree(out / "data")
    for rel, body in files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        n = p.stat().st_size
        print(f"  {rel:<36} {n / 1e6:5.1f} MB")
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
