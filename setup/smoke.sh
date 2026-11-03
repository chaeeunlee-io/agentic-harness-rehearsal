#!/usr/bin/env bash
# CPU-only installation check. Run after `setup/install.sh`.
#
# Checks:
#   1. Harbor imports and its CLI runs.
#   2. Harness modules referenced by the benchmark configs import correctly.
#   3. SkyRL and the GRPO entrypoint import when the SkyRL environment is installed.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARBOR_DIR="$ROOT/third_party/harbor"
SKYRL_DIR="$ROOT/third_party/SkyRL"
GRPO_ENTRYPOINT=examples.train_integrations.harbor.entrypoints.main_harbor_xh

failed=0
ok()  { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; failed=1; }

# in_harbor <command...>: run a command in harbor's environment
in_harbor() {
  ( cd "$HARBOR_DIR" && uv run --no-sync "$@" )
}

# in_skyrl <command...>: run a command in SkyRL's environment, with this repo importable
in_skyrl() {
  ( cd "$SKYRL_DIR" && PYTHONPATH="$ROOT" uv run --no-sync "$@" )
}


echo "== 1. harbor"
if in_harbor python -c "import harbor, importlib.metadata as m; print('harbor', m.version('harbor'))"; then
  ok "import harbor"
else
  bad "import harbor"
fi

if in_harbor harbor --help > /dev/null; then
  ok "harbor --help"
else
  bad "harbor --help"
fi

if in_harbor python -c "from harbor.environments.eidf import EIDFEnvironment; from harbor.agents.terminus_2.terminus_2 import Terminus2"; then
  ok "EIDF environment + Terminus-2 import"
else
  bad "EIDF environment + Terminus-2 import"
fi


echo "== 2. harnesses (the import_path values in configs/benchmarks/*.yaml), PYTHONPATH=$ROOT"
HARNESS_MODULES="
  benchmarks.medagentbench.harnesses.harness_dispatch
  benchmarks.appworld.harnesses.harness_dispatch
  benchmarks.bfcl_multiturn.harnesses.bfclmt_agent
  benchmarks.bfcl_singleturn.harnesses.harness_dispatch
  benchmarks.reasoning_gym.harnesses.harness_dispatch"
for module in $HARNESS_MODULES; do
  # Prints the module's harness names (its HARNESS_REGISTRY), or else whether it defines DispatchAgent
  if out=$( cd "$HARBOR_DIR" && PYTHONPATH="$ROOT" uv run --no-sync python -c "import importlib; mod=importlib.import_module('$module'); print(getattr(mod,'HARNESS_REGISTRY',None) and sorted(mod.HARNESS_REGISTRY) or 'DispatchAgent' in dir(mod))" 2>&1 ); then
    ok "$module -> $out"
  else
    bad "$module: $(echo "$out" | tail -1)"
  fi
done


echo "== 3. SkyRL (upstream @ 4bdf9766 + training/rl overlay)"
if [ -d "$SKYRL_DIR/.venv" ]; then
  if in_skyrl python -c "import skyrl, importlib; importlib.import_module('$GRPO_ENTRYPOINT'); print('skyrl + main_harbor_xh import OK')"; then
    ok "skyrl + GRPO entrypoint import"
  else
    bad "skyrl + GRPO entrypoint import"
  fi
  if in_skyrl python -m $GRPO_ENTRYPOINT --help > /dev/null 2>&1; then
    ok "main_harbor_xh --help (hydra config loads)"
  else
    echo "  SKIP main_harbor_xh --help (entrypoint has no --help / needs ray; see setup/README.md)"
  fi
elif [ -d "$SKYRL_DIR" ]; then
  echo "  SKIP SkyRL env not synced (run on a Linux GPU host: cd $SKYRL_DIR && uv sync --extra fsdp --extra harbor)"
  if [ -f "$SKYRL_DIR/examples/train_integrations/harbor/entrypoints/main_harbor_xh.py" ]; then
    ok "training/rl overlay present in $SKYRL_DIR"
  else
    bad "overlay missing"
  fi
else
  echo "  SKIP third_party/SkyRL absent (setup/install.sh with SKYRL=1)"
fi


echo
if [ $failed = 0 ]; then
  echo "SMOKE OK"
else
  echo "SMOKE FAILED"
  exit 1
fi
