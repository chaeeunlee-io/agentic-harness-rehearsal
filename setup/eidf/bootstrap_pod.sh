#!/bin/bash
# Prepare a fresh EIDF pod to use this checkout. Run once per pod; safe to run again.
#
#   bash setup/eidf/bootstrap_pod.sh
#
# Only needed on EIDF (or any setup where several machines share one checkout). On a single
# machine of your own, use setup/install.sh instead and ignore this script.
#
# THE PROBLEM IT SOLVES
# The repo and its four Python environments live on CephFS, which every pod mounts:
#   .venv                      the exp CLI
#   third_party/harbor/.venv   harbor (evaluation, trace collection)
#   setup/sft/.venv            SFT training
#   third_party/SkyRL/.venv    GRPO training
# But an environment is not self-contained: its bin/python is only a link to a Python
# installed somewhere else. Ours point at:
#   - uv's Python 3.13.18, in ~/.local/share/uv/python/   (first three environments)
#   - the image's /usr/bin/python3.12                     (SkyRL)
# The home folder is NOT shared: each pod has its own, and a new pod's is empty. So on a new
# pod those links point nowhere and nothing runs. Installing the same uv and the same Python
# at the same paths makes the links valid again, and the shared environments then work
# unchanged, with no rebuild.
#
# WHAT IT DOES NOT DO
# It never rebuilds ("re-syncs") an environment. They are shared, so a rebuild would change
# them for every pod at once, including jobs running there. Use setup/install.sh for that,
# deliberately, when no pod is using them.
#
# Overridable: UV_VERSION (default 0.13.6), PY_VERSION (default 3.13.18), UV_CACHE_DIR.

# -u: stop with an error if the script uses a variable that was never set (catches typos)
set -u

# The exact versions the shared environments were built with. They must match: a different
# Python version would live at a different path, and the environments' links would still
# point nowhere.
UV_VERSION=${UV_VERSION:-0.13.6}     # the uv the shared environments were built with
PY_VERSION=${PY_VERSION:-3.13.18}    # their interpreter (uv-managed)

# The repo's root folder: two levels up from this script (setup/eidf/ -> repo root)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Where uv installs the Pythons it manages, in this pod's own home folder
PYTHON_DIR="$HOME/.local/share/uv/python"

# Report helpers: each check prints one line. A FAIL makes the whole script fail at the end;
# a WARN is reported but does not.
failed=0
ok()   { echo "  OK    $*"; }
warn() { echo "  WARN  $*"; }
bad()  { echo "  FAIL  $*"; failed=1; }

# Prints the version number of the uv on PATH ("uv 0.13.6 (...)" -> "0.13.6"), or nothing if
# there is no uv
installed_uv_version() {
  uv --version 2>/dev/null | awk '{print $2}'
}


# ---- Step 1: uv ----------------------------------------------------------------------------
# uv is the tool that installs Python and runs the environments. The pod image does not have
# it, so install the pinned version into ~/.local/bin.

echo "== uv $UV_VERSION in ~/.local/bin"

# Put ~/.local/bin first on PATH, so the uv found below is the one installed here (an older uv
# elsewhere on PATH would otherwise win)
export PATH="$HOME/.local/bin:$PATH"

if [ "$(installed_uv_version)" = "$UV_VERSION" ]; then
  # Already there (e.g. this script ran on this pod before): nothing to do
  ok "uv $UV_VERSION already installed"
else
  # Download and run uv's official installer for exactly this version.
  # UV_NO_MODIFY_PATH=1 stops it editing the pod's shell profile files.
  # Its output is hidden; the check after it reports whether it worked.
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh > /dev/null 2>&1
  if [ "$(installed_uv_version)" = "$UV_VERSION" ]; then
    ok "installed uv $UV_VERSION"
  else
    bad "could not install uv $UV_VERSION (network?)"
  fi
fi


# ---- Step 2: Python ------------------------------------------------------------------------
# Install the Python the first three environments point at.

echo "== Python $PY_VERSION (uv-managed)"

# uv keeps a download cache. Put it on the pod's local disk: CephFS is slow for its many small
# files.
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv_cache}

# Download and install Python $PY_VERSION into ~/.local/share/uv/python/ (does nothing if it
# is already installed)
if uv python install "$PY_VERSION" > /dev/null 2>&1; then
  ok "uv python $PY_VERSION"
else
  bad "uv python install $PY_VERSION"
fi

# uv installs Python as cpython-3.13.18-linux-x86_64-gnu, but the environments recorded the
# shorter name cpython-3.13-linux-x86_64-gnu. Recent uv versions create that short name
# themselves; if it is missing, make it a link to the full install.
full_python=$(ls -d "$PYTHON_DIR"/cpython-$PY_VERSION-linux-x86_64-gnu 2>/dev/null)
minor_python="$PYTHON_DIR/cpython-${PY_VERSION%.*}-linux-x86_64-gnu"   # 3.13.18 -> 3.13
if [ -n "$full_python" ] && [ ! -e "$minor_python" ]; then
  ln -s "$full_python" "$minor_python" && ok "linked $minor_python"
fi

# The SkyRL environment uses the pod image's own Python 3.12 instead. This script cannot
# install that one; it only checks the image has it.
if [ -x /usr/bin/python3.12 ]; then
  ok "/usr/bin/python3.12 (SkyRL environment)"
else
  warn "/usr/bin/python3.12 missing: the SkyRL environment will not run on this pod"
fi


# ---- Step 3: check the shared environments -------------------------------------------------
# Now that uv and Python are in place, try to start each environment's Python. This is the
# real test that steps 1 and 2 worked.

echo "== shared environments"
for venv in .venv third_party/harbor/.venv setup/sft/.venv third_party/SkyRL/.venv; do
  python="$ROOT/$venv/bin/python"
  if [ ! -d "$ROOT/$venv" ]; then
    # The environment has never been built in this checkout (a fresh clone): not this
    # script's job, so only a warning
    warn "$venv not built yet (setup/install.sh, then 'uv sync' at the repo root)"
  elif "$python" -c "import sys" 2>/dev/null; then
    # Python started: report which real interpreter the link leads to
    ok "$venv -> $(readlink -f "$python")"
  else
    # Python did not start: its link points at something this pod does not have
    bad "$venv: interpreter $(readlink "$python") not on this pod"
  fi
done


# ---- Step 4: other things the experiments need on the pod ----------------------------------
# Warnings only: the environments work without them, but runs will not.

echo "== cluster access"

# GPUs: list them, counted by model (e.g. "2 NVIDIA H100 80GB HBM3")
if command -v nvidia-smi > /dev/null; then
  gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | sed 's/^ *//' | tr '\n' ' ')
  ok "GPUs: $gpus"
else
  warn "no nvidia-smi (CPU-only pod)"
fi

# tmux: long runs must be started inside tmux so they survive a dropped connection
if command -v tmux > /dev/null; then
  ok "tmux"
else
  warn "tmux missing: long runs must survive a dropped connection"
fi

# .env: the cluster settings (namespace, paths, image names, ...) that session.sh loads.
# It lives in the shared checkout, so it is set up once, not per pod.
if [ -f "$ROOT/.env" ]; then
  ok ".env present"
else
  warn ".env missing: cp env.example .env and fill it in (see setup/eidf/README.md)"
fi


# ---- Result --------------------------------------------------------------------------------
# Exit with an error if any step above printed FAIL (warnings do not count)

echo
if [ $failed = 0 ]; then
  echo "BOOTSTRAP OK: in every new shell, source setup/eidf/session.sh"
else
  echo "BOOTSTRAP FAILED"
  exit 1
fi
