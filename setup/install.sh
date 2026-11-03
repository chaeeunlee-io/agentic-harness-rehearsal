#!/usr/bin/env bash
# Install the environments used by the experiments:
#   1. Harbor from the vendored `third_party/harbor`.
#   2. SkyRL at the pinned commit, with the local `training/rl` overlay.
#   3. The SFT environment in `setup/sft`.
#   4. The root project providing `uv run exp`.
#
# Requires git, uv, and network access.
#
# Usage:
#   setup/install.sh                # everything
#   SKYRL=0 setup/install.sh        # skip SkyRL
#   SKYRL_SYNC=1 setup/install.sh   # sync SkyRL's dependencies on any system
#   SFT=0 setup/install.sh          # skip the SFT environment
#
# Pins:
#   Harbor  (vendored fork) @ 9233373feb3fce2f4e7351cd75f3af064ff45643
#   SkyRL   NovaSky-AI/SkyRL @ 4bdf9766629dd54850c8885617aec5d483a5ee57
#
# SkyRL dependencies are synced on Linux by default. `SKYRL_SYNC=1` forces the
# sync on other systems. The SFT environment requires Linux and CUDA.
#
# On EIDF, these environments are shared over CephFS. Only run this when no pod
# is using them; see `setup/eidf/README.md`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARBOR_DIR="$ROOT/third_party/harbor"
HARBOR_PIN="9233373feb3fce2f4e7351cd75f3af064ff45643"
SKYRL_DIR="$ROOT/third_party/SkyRL"
SKYRL_UPSTREAM="https://github.com/NovaSky-AI/SkyRL.git"
SKYRL_PIN="4bdf9766629dd54850c8885617aec5d483a5ee57"
OVERLAY="$ROOT/training/rl"

SKYRL="${SKYRL:-1}"
SFT="${SFT:-1}"
# SkyRL's dependencies are CUDA wheels, so by default they are synced on Linux only
if [ "$(uname -s)" = Linux ]; then
  SKYRL_SYNC="${SKYRL_SYNC:-1}"
else
  SKYRL_SYNC="${SKYRL_SYNC:-0}"
fi

if ! command -v uv > /dev/null; then
  echo "ERROR: uv not found (https://docs.astral.sh/uv/)"
  exit 1
fi


# --- 1. harbor -------------------------------------------------------------------------------

echo "== harbor: vendored copy of <owner>/harbor @ $HARBOR_PIN"
echo "   syncing $HARBOR_DIR/.venv from its uv.lock (--frozen: exactly the resolver state the fork pinned)"
( cd "$HARBOR_DIR" && uv sync --frozen --no-dev )
echo "   harbor $(cd "$HARBOR_DIR" && uv run --no-sync python -c 'import importlib.metadata as m; print(m.version("harbor"))') installed"


# --- 2. SkyRL --------------------------------------------------------------------------------

if [ "$SKYRL" = "1" ]; then
  echo "== SkyRL: upstream $SKYRL_UPSTREAM @ $SKYRL_PIN"

  # Fetch exactly the pinned commit (shallow), unless SkyRL is already cloned
  if [ ! -d "$SKYRL_DIR/.git" ]; then
    mkdir -p "$SKYRL_DIR" && (
      cd "$SKYRL_DIR" &&
        git init -q &&
        git remote add origin "$SKYRL_UPSTREAM" &&
        git fetch -q --depth 1 origin "$SKYRL_PIN" &&
        git checkout -q FETCH_HEAD
    )
  fi

  # Refuse to go on if the clone is at any other commit
  if ! ( cd "$SKYRL_DIR" && [ "$(git rev-parse HEAD)" = "$SKYRL_PIN" ] ); then
    echo "ERROR: $SKYRL_DIR is not at $SKYRL_PIN"
    exit 1
  fi

  # The overlay (training/rl/) replaces upstream's harbor integration, pyproject.toml and
  # uv.lock with the fork's
  echo "   overlaying training/rl/ (the fork's examples/train_integrations/harbor + pyproject.toml + uv.lock)"
  rm -rf "$SKYRL_DIR/examples/train_integrations/harbor"
  cp -R "$OVERLAY/examples/train_integrations/harbor" "$SKYRL_DIR/examples/train_integrations/harbor"
  cp "$OVERLAY/pyproject.toml" "$SKYRL_DIR/pyproject.toml"
  cp "$OVERLAY/uv.lock" "$SKYRL_DIR/uv.lock"
  if [ -f "$OVERLAY/.gitignore" ]; then
    cp "$OVERLAY/.gitignore" "$SKYRL_DIR/.gitignore"
  fi

  # The fork's dependency on harbor is a path (${PROJECT_ROOT}/harbor): point it at the
  # vendored copy instead
  python3 - "$SKYRL_DIR" "$HARBOR_DIR" <<'PY'
import pathlib, sys

skyrl, harbor = pathlib.Path(sys.argv[1]), sys.argv[2]
for name in ("pyproject.toml", "uv.lock"):
    path = skyrl / name
    text = path.read_text()
    new_text = text.replace("${PROJECT_ROOT}/harbor", harbor)
    if new_text != text:
        path.write_text(new_text)
        print(f"   {name}: harbor path -> {harbor}")
PY

  if [ "$SKYRL_SYNC" = "1" ]; then
    echo "   syncing SkyRL deps: uv sync --extra fsdp --extra harbor (CUDA wheels; Linux + GPU host expected)"
    ( cd "$SKYRL_DIR" && uv sync --extra fsdp --extra harbor )
  else
    echo "   SkyRL dependency sync SKIPPED on $(uname -s) (CUDA wheels). On the GPU host run:"
    echo "     cd $SKYRL_DIR && uv sync --extra fsdp --extra harbor"
  fi
fi


# --- 3. SFT environment ----------------------------------------------------------------------

if [ "$SFT" = "1" ]; then
  if [ "$(uname -s)" = "Linux" ]; then
    echo "== SFT environment: setup/sft (TRL 0.27.1, transformers 4.57.6, torch 2.9.1, liger-kernel 0.6.4)"
    ( cd "$ROOT/setup/sft" && uv sync )
  else
    echo "== SFT environment SKIPPED on $(uname -s) (liger-kernel/triton are Linux-only); on the GPU host: cd setup/sft && uv sync"
  fi
fi


# --- 4. the exp CLI --------------------------------------------------------------------------

echo "== CLI: root project (uv run exp)"
( cd "$ROOT" && uv sync )


cat <<EON

== done. Entry point: uv run exp (uv run exp --help; uv run exp list experiments; uv run exp plan specialists-vs-3h)
 * next:  setup/smoke.sh (no-GPU check), then the README example or docs/COMMANDS.md
 * run an experiment: uv run exp run <experiment> (uv run exp plan <experiment> shows its commands)
 * fill env.example -> .env (cluster settings: PROJECT_ROOT, sandbox backend, model store, ...)
 * on EIDF: setup/eidf/README.md (once per pod: setup/eidf/bootstrap_pod.sh; every shell: . setup/eidf/session.sh)
EON
