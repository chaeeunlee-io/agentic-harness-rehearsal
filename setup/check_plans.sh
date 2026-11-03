#!/bin/bash
# Check every experiment plan without running it.
#
# For each experiment:
#   1. Save `uv run exp plan <experiment>` to <out dir>/<experiment>.sh.
#   2. Check the generated script's bash syntax.
#   3. Verify that `exp run <experiment> --dry-run` produces the same script.
#   4. Dry-run each supported `exp` command in the script to check arguments
#      and config resolution without generating tasks, training, or serving.
#
# Step 3 writes runs/plans/<experiment>.sh, the script `exp run` executes: do not
# run this check while an experiment started with `exp run` is running.
#
# No GPU is required. The smoke test runs this as its `plans` stage.
#
# Usage:
#   setup/check_plans.sh [out dir]
#
# Default output directory: runs/plans-check
# Uses BACKEND (default: docker) and SANDBOX_API_BASE.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=${1:-runs/plans-check}
mkdir -p "$OUT"

BACKEND_OPTS="--backend ${BACKEND:-docker}"
[ -n "${SANDBOX_API_BASE:-}" ] && BACKEND_OPTS+=" --sandbox-api-base $SANDBOX_API_BASE"

# The variables a plan script sets for its commands
export PORT=8000
export API=http://127.0.0.1:$PORT/v1
export SERVER=0      # stands in for the server's process id

failed=0

for experiment in $(uv run exp list experiments | awk '{print $1}'); do
  script=$OUT/$experiment.sh

  if ! uv run exp plan $experiment $BACKEND_OPTS > $script; then
    echo "FAIL  $experiment: exp plan failed"
    failed=1
    continue
  fi
  if ! bash -n $script; then
    echo "FAIL  $experiment: the script is not valid bash"
    failed=1
    continue
  fi
  if ! uv run exp run $experiment $BACKEND_OPTS --dry-run > /dev/null || ! cmp -s $script runs/plans/$experiment.sh; then
    echo "FAIL  $experiment: exp run --dry-run did not write the same script as exp plan"
    failed=1
    continue
  fi

  # Dry-run each exp command that supports it. A plan line looks like
  #   setsid uv run exp serve --benchmark ... --port $PORT & SERVER=$!
  #   uv run exp eval --benchmark ... || true
  # so drop the "setsid", "& SERVER=$!" and "|| true" around the command first.
  commands=0 bad=0
  while read -r line; do
    command=${line#setsid }
    command=${command% & SERVER=\$!}
    command=${command% || true}
    read -r -a words <<< "$command"           # words[3] is the exp subcommand
    [ "${words[*]:0:3}" = "uv run exp" ] || continue
    case ${words[3]} in
      tasks|sft|grpo|serve|eval) ;;
      *) continue ;;
    esac
    commands=$((commands + 1))
    if ! eval "$command --dry-run" > /dev/null 2>> $OUT/$experiment.errors; then
      echo "      $experiment: dry run failed: $command"
      bad=$((bad + 1))
    fi
  done < $script

  if [ $bad = 0 ]; then
    echo "ok    $experiment ($commands commands dry-run)"
    rm -f $OUT/$experiment.errors
  else
    echo "FAIL  $experiment: $bad of $commands commands failed (errors in $OUT/$experiment.errors)"
    failed=1
  fi
done

exit $failed
