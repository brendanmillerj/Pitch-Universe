#!/bin/bash
# Publish the built atlas to Cloudflare Pages (project "pitchomics", served at pitchomics.com).
#
# Same files as the Hugging Face Space: deploy_space.py --dry-run --out builds the folder (index.html with the head
# added, data/atlas-<hash>-{main,aaa}.json[.gz]); wrangler uploads it. Ships whatever analysis/pitch_atlas.html holds
# right now, exactly as deploy_space.py would.
#
#   bash scripts/deploy_pages.sh               build + deploy
#   bash scripts/deploy_pages.sh --build-only  build the folder and stop (no token needed)
#
# Needs ~/.cloudflare.env (chmod 600), which wrangler reads from the environment:
#   export CLOUDFLARE_API_TOKEN=...      token with Account > Cloudflare Pages > Edit
#   export CLOUDFLARE_ACCOUNT_ID=...
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# the build step (deploy_space.py --dry-run) needs only the standard library: any Python 3 will do
PY="${PITCHOMICS_PYTHON:-python3}"
[ -x "$REPO/.venv/bin/python" ] && [ -z "${PITCHOMICS_PYTHON:-}" ] && PY="$REPO/.venv/bin/python"
PROJECT=pitchomics
NODE_V=v22.22.1
WRANGLER_V=4.156.0
TOOLS="${PITCHOMICS_TOOLS:-/scratch1/$USER/tools}"
[ -d /scratch1 ] || TOOLS="$HOME/.pitchomics-tools"           # off CARC

# wrangler (and Node with npm) live in scratch: reinstall them if scratch was purged
NODE_DIR="$TOOLS/node-$NODE_V-linux-x64"
WRANGLER="$TOOLS/wrangler/node_modules/.bin/wrangler"
if command -v npx >/dev/null 2>&1 && [ "$(uname)" = Darwin ]; then
  WRANGLER="npx --yes wrangler@$WRANGLER_V"                   # the Mac: its own Node
elif [ ! -x "$WRANGLER" ]; then
  echo "installing Node $NODE_V and wrangler $WRANGLER_V in $TOOLS"
  mkdir -p "$TOOLS/dl" && cd "$TOOLS/dl"
  curl -sSfO "https://nodejs.org/dist/$NODE_V/node-$NODE_V-linux-x64.tar.xz"
  curl -sSfO "https://nodejs.org/dist/$NODE_V/SHASUMS256.txt"
  grep " node-$NODE_V-linux-x64.tar.xz\$" SHASUMS256.txt | sha256sum -c -
  tar -xJf "node-$NODE_V-linux-x64.tar.xz" -C "$TOOLS" && cd "$TOOLS" && rm -rf "$TOOLS/dl"
  mkdir -p "$TOOLS/wrangler" && cd "$TOOLS/wrangler"
  PATH="$NODE_DIR/bin:$PATH" npm_config_cache="$TOOLS/npm-cache" npm init -y >/dev/null
  PATH="$NODE_DIR/bin:$PATH" npm_config_cache="$TOOLS/npm-cache" npm install --no-fund --no-audit "wrangler@$WRANGLER_V"
  rm -rf "$TOOLS/npm-cache"
fi
[ -d "$NODE_DIR/bin" ] && export PATH="$NODE_DIR/bin:$PATH"

# build the folder
SCRATCH="${TMPDIR:-/tmp}"; [ -d "/scratch1/$USER" ] && SCRATCH="/scratch1/$USER/tmp"
mkdir -p "$SCRATCH"
OUT="$(mktemp -d "$SCRATCH/pitchomics-pages.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT
cd "$REPO"
# PITCHOMICS_PAYWALL=1 moves the paid Evolution data under p/ (deploy_space.py --premium). Leave it off until the page
# fetches p/ (ensurePremium) and functions/p/_middleware.js guards it, or Evolution would vanish for everyone.
PREMIUM=""; [ "${PITCHOMICS_PAYWALL:-0}" = "1" ] && PREMIUM="--premium"
"$PY" scripts/deploy_space.py --dry-run --out "$OUT" $PREMIUM
# caching: data files carry a content hash in their names, so browsers may keep them; the page itself is always fresh
cat > "$OUT/_headers" <<'EOF'
/data/*
  Cache-Control: public, max-age=31536000, immutable
/
  Cache-Control: no-cache
/index.html
  Cache-Control: no-cache
EOF
if [ "${1:-}" = "--build-only" ]; then
  trap - EXIT
  echo "built $OUT (not deployed)"
  exit 0
fi

[ -f "$HOME/.cloudflare.env" ] && source "$HOME/.cloudflare.env"
: "${CLOUDFLARE_API_TOKEN:?put CLOUDFLARE_API_TOKEN in ~/.cloudflare.env}"
: "${CLOUDFLARE_ACCOUNT_ID:?put CLOUDFLARE_ACCOUNT_ID in ~/.cloudflare.env}"
if ! [[ "$CLOUDFLARE_ACCOUNT_ID" =~ ^[0-9a-f]{32}$ ]]; then
  echo "CLOUDFLARE_ACCOUNT_ID in ~/.cloudflare.env should be the 32-character Account ID (letters a-f and digits)," >&2
  echo "not your login. It's the code right after dash.cloudflare.com/ in the address bar when you're signed in." >&2
  exit 1
fi
export CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID

if ! $WRANGLER pages project list 2>/dev/null | grep -qw "$PROJECT"; then
  echo "creating the Cloudflare Pages project $PROJECT"
  $WRANGLER pages project create "$PROJECT" --production-branch main
fi
$WRANGLER pages deploy "$OUT" --project-name "$PROJECT" --branch main \
  --commit-hash "$(git rev-parse --short HEAD)" --commit-message "Atlas $(git rev-parse --short HEAD)" --commit-dirty=true
echo "live at https://$PROJECT.pages.dev (and https://pitchomics.com once the domain is attached)"
