"""Publish the built atlas to the Hugging Face Space as a standalone page.

analysis/pitch_atlas.html is a page fragment (no doctype, charset or viewport):
the claude.ai artifact host supplies those. A static Space serves the file as-is
with a bare `text/html` content type, so without them browsers fall back to
windows-1252 (UTF-8 "·" renders as "Â·"), render in quirks mode, and lay the
page out at desktop width on phones. This adds the document head and uploads.

Usage (after `python scripts/atlas_build.py ...`; needs `huggingface-cli login`):
  python scripts/deploy_space.py              # upload to the Space
  python scripts/deploy_space.py --dry-run    # show what would be uploaded
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "analysis" / "pitch_atlas.html"
SPACE = "brmiller/Pitch-Universe"
HEAD = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
)


def standalone(fragment: str) -> str:
    if fragment.lstrip().lower().startswith("<!doctype"):
        return fragment  # already a full document
    return HEAD + fragment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    body = standalone(SRC.read_text(encoding="utf-8")).encode("utf-8")
    print(f"index.html from {SRC.name}: {len(body) / 1e6:.1f} MB, starts {body[:60]!r}")
    if a.dry_run:
        return

    from huggingface_hub import HfApi
    c = HfApi().upload_file(path_or_fileobj=body, path_in_repo="index.html", repo_id=SPACE,
                            repo_type="space", commit_message="Update atlas page")
    print(f"uploaded: {c.commit_url}")


if __name__ == "__main__":
    main()
