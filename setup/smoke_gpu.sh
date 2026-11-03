#!/bin/bash
# GPU smoke test for the full pipeline at small scale. It checks that every stage
# runs; it does not check the numbers.
#
# Usage (inside tmux):
#   setup/smoke_gpu.sh one-gpu [benchmark ...]
#       task generation, base eval, short SFT + eval, rehearsal traces,
#       Gemma SFT + eval
#
#   setup/smoke_gpu.sh two-gpu [benchmark ...]
#       Qwen3-32B teacher traces, dataset build, short GRPO
#
# Default benchmarks:
#   medagentbench reasoning_gym bfcl_singleturn bfcl_multiturn appworld
#
# Environment variables:
#   BACKEND            sandbox backend: docker (default) or eidf
#   SANDBOX_API_BASE   model-server URL reachable from sandboxes; required for eidf
#   CONCURRENCY        concurrent trials (default: 5)
#   GRPO_INIT_DIR      SFT checkpoints from one-gpu (default: runs/smoke/ckpt)
#   WAIT_ONE_GPU_MIN   minutes two-gpu waits for one-gpu outputs (default: 600)
#
# Each stage uses the same user-facing commands documented elsewhere in the repo.
# Model servers are started, checked, and stopped with `exp serve`, `exp wait`,
# and `exp stop`.
#
# Outputs under `runs/smoke/`:
#   status.tsv            stage status, runtime, and log path
#   commands-<mode>.sh    commands run by this mode
#   logs/<benchmark>/     stage and model-server logs
#   jobs/<benchmark>/     evaluation and trace jobs
#   ckpt/ pools/ rl/      checkpoints, datasets, and GRPO outputs
#   plans/                checked experiment plans from one-gpu
#   done                  progress markers
#
# Failed stages are recorded and the script continues.
#
# The two modes may run concurrently on machines sharing the checkout, or
# sequentially on one machine. two-gpu waits where it needs outputs from one-gpu.
#
# The script is wrapped in one `{ ... }` block so bash reads the full file before
# execution; editing it does not affect a run already in progress.
{
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE=${1:?one-gpu or two-gpu}
shift

SMOKE_DIR=runs/smoke
STATUS=$SMOKE_DIR/status.tsv
DONE=$SMOKE_DIR/done
COMMANDS=$SMOKE_DIR/commands-$MODE.sh
API=http://127.0.0.1:8000/v1       # the model server every stage talks to
QWEN=qwen3-4b
GEMMA=gemma-3-4b-it
FIVE_TASKS="--limit 5 --n-attempts 1"
THREE_TASKS="--limit 3 --n-attempts 1"
# Traces that feed a pool (teacher and rehearsal) get more tasks: only correct traces are kept,
# and the pool build stops when a harness has none, which 3 tasks with 1 attempt each often gave
POOL_TASKS="--limit 8 --n-attempts 1"


# ============================================================================================
# Helpers
# ============================================================================================

# record <command line>
# Appends a command to commands-<mode>.sh, the list of everything this run did.
record() {
  echo "$*" >> $COMMANDS
}

# run_stage <benchmark> <stage> <command...>
# Runs one stage (3-hour limit), writes its output to logs/<benchmark>/<stage>.log and appends
# its result to status.tsv. Returns the stage's exit code. The command is recorded in
# commands-<mode>.sh, unless the caller set SKIP_RECORD=1 because it records the command in
# the form a user would type it.
run_stage() {
  local bench=$1 stage=$2
  shift 2
  local log=$SMOKE_DIR/logs/$bench/$stage.log
  local start=$(date +%s)
  mkdir -p $SMOKE_DIR/logs/$bench
  echo "[$(date +%H:%M:%S)] $bench / $stage: $*"
  [ -n "${SKIP_RECORD:-}" ] || record "$*"

  timeout 3h "$@" > $log 2>&1
  local rc=$?

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date +%F_%T)" "$bench" "$stage" "$rc" "$(( $(date +%s) - start ))" "$log" >> $STATUS
  echo "    rc=$rc ($(( $(date +%s) - start ))s)"
  return $rc
}

# run_opts <benchmark>
# The options every `exp eval` and `exp collect` gets: sandbox backend, model server,
# concurrency, and this benchmark's own jobs folder.
run_opts() {
  local bench=$1
  local opts="--backend ${BACKEND:-docker} --api-base $API --concurrency ${CONCURRENCY:-5}"
  opts+=" --jobs-dir $SMOKE_DIR/jobs/$bench"
  [ -n "${SANDBOX_API_BASE:-}" ] && opts+=" --sandbox-api-base $SANDBOX_API_BASE"
  echo "$opts"
}

# start_server <benchmark> <name> <exp serve options...>
# Starts a model server in the background, in its own process group (setsid) so stop_server
# can stop all of it, then runs the stage "serve-<name>": `exp wait`, which returns once the
# server answers, or fails as soon as the server exits (up to 30 minutes).
# The server's own output goes to logs/<benchmark>/server-<name>.log.
start_server() {
  local bench=$1 name=$2
  shift 2
  local server_log=$SMOKE_DIR/logs/$bench/server-$name.log
  mkdir -p $SMOKE_DIR/logs/$bench

  record "setsid uv run exp serve -b $bench $* --port 8000 > $server_log 2>&1 & SERVER=\$!"
  setsid uv run exp serve -b $bench "$@" --port 8000 > $server_log 2>&1 &
  SERVER=$!

  record "uv run exp wait --api-base $API --pid \$SERVER"
  SKIP_RECORD=1 run_stage $bench serve-$name uv run exp wait --api-base $API --pid $SERVER
}

# stop_server
# `exp stop`: stops the server's process group and waits until the GPUs are free (it also
# cleans up vLLM engine processes that outlive the server). Without a server it only waits.
stop_server() {
  if [ -n "${SERVER:-}" ]; then
    record "uv run exp stop --pid \$SERVER"
    uv run exp stop --pid $SERVER
  else
    record "uv run exp stop"
    uv run exp stop
  fi
  SERVER=
}

# fetch_tasks <benchmark> <split>
# Runs the stage "tasks-<split>": `exp tasks` for that split, unless this smoke run already
# generated it (fetch_once, below, does the generating under a lock).
fetch_tasks() {
  record "uv run exp tasks -b $1 --split $2"
  SKIP_RECORD=1 run_stage $1 tasks-$2 bash "$ROOT/setup/smoke_gpu.sh" fetch $1 $2
}

# fetch_once <benchmark> <split>                 (runs as the internal mode "fetch")
# Generates the split under a lock, so the two modes never generate the same split at once,
# and leaves a marker in fetched/ so it is generated only once per smoke run.
fetch_once() {
  local bench=$1 split=$2
  mkdir -p $SMOKE_DIR/fetched
  exec 9> $SMOKE_DIR/fetched/.lock
  flock 9
  if [ -f $SMOKE_DIR/fetched/$bench-$split ]; then
    echo "already fetched by this smoke run"
    return 0
  fi
  uv run exp tasks -b $bench --split $split && touch $SMOKE_DIR/fetched/$bench-$split
}

# wait_for_one_gpu <benchmark>
# Waits until one-gpu has finished this benchmark (its line "one-gpu <benchmark>" in done).
wait_for_one_gpu() {
  local bench=$1
  record "# wait until one-gpu has finished $bench (its SFT checkpoint and rehearsal traces)"
  for i in $(seq ${WAIT_ONE_GPU_MIN:-600}); do
    grep -qx "one-gpu $bench" $DONE 2>/dev/null && return 0
    [ $i = 1 ] && echo "[$(date +%H:%M:%S)] $bench: waiting for one-gpu (SFT checkpoint, rehearsal traces)"
    sleep 60
  done
  echo "    gave up waiting for one-gpu $bench; the next stages will fail"
  return 1
}


# ============================================================================================
# one-gpu: everything that needs a single GPU
# ============================================================================================

one_gpu_stages() {
  local bench=$1
  local opts=$(run_opts $bench)
  local qwen_ckpt=$SMOKE_DIR/ckpt/$bench/$QWEN/3h

  # Tasks: the test split, plus a training split for three of the benchmarks
  fetch_tasks $bench test
  case $bench in
    reasoning_gym)                fetch_tasks $bench train-easy ;;
    medagentbench|bfcl_multiturn) fetch_tasks $bench train ;;
  esac

  # Base model: serve it and evaluate 5 test tasks on every harness (held-out one included)
  start_server $bench base --pool base &&
    run_stage $bench eval-base \
      uv run exp eval -b $bench --pool base --harness all --split test $FIVE_TASKS $opts
  stop_server

  # Short SFT: 10 steps on the shipped 3H pool
  run_stage $bench sft-3h \
    uv run exp sft -b $bench --pool 3h --max-steps 10 --out $qwen_ckpt

  # Serve the SFT checkpoint: evaluate it, then use it as the demonstrator for rehearsal traces
  if start_server $bench sft-3h --pool 3h --checkpoint $qwen_ckpt/final; then
    run_stage $bench eval-sft-3h \
      uv run exp eval -b $bench --pool 3h --harness all --split test $FIVE_TASKS $opts

    case $bench in
      medagentbench|reasoning_gym)
        fetch_tasks $bench rehearsal-t2
        run_stage $bench collect-rehearsal-t2 \
          uv run exp collect -b $bench --rehearsal --job rehearsal-t2 $POOL_TASKS $opts
        ;;
      bfcl_multiturn)
        fetch_tasks $bench rehearsal
        for job in rehearsal-fc rehearsal-xml; do
          run_stage $bench collect-$job \
            uv run exp collect -b $bench --rehearsal --job $job $POOL_TASKS $opts
        done
        ;;
    esac
  fi
  stop_server

  # Gemma (MedAgentBench and Reasoning Gym): short SFT, then serve it and evaluate 3 tasks.
  # On MedAgentBench Gemma was served differently for fc than for the other harnesses, so each
  # serving profile gets its own server. On Reasoning Gym one server covers all harnesses.
  case $bench in
    medagentbench|reasoning_gym)
      local gemma_ckpt=$SMOKE_DIR/ckpt/$bench/$GEMMA/3h
      run_stage $bench sft-gemma-3h \
        uv run exp sft -b $bench -m $GEMMA --pool 3h --max-steps 10 --out $gemma_ckpt

      local profiles=trained
      [ $bench = medagentbench ] && profiles="parity fc"
      for harness in $profiles; do
        local harness_opt=
        [ $harness != trained ] && harness_opt="--harness $harness"
        start_server $bench gemma-sft-3h-$harness \
            -m $GEMMA --pool 3h --checkpoint $gemma_ckpt/final $harness_opt &&
          run_stage $bench eval-gemma-sft-3h-$harness \
            uv run exp eval -b $bench -m $GEMMA --pool 3h --harness $harness --split test $THREE_TASKS $opts
        stop_server
      done
      ;;
  esac

  # Tell two-gpu that this benchmark's SFT checkpoint and rehearsal traces are ready
  echo "one-gpu $bench" >> $DONE
}


# ============================================================================================
# two-gpu: the 32B teacher and GRPO
# ============================================================================================

two_gpu_stages() {
  local bench=$1
  local opts=$(run_opts $bench)

  # Tasks: the training split the teacher runs on (Reasoning Gym: both difficulty bands)
  case $bench in
    reasoning_gym) fetch_tasks $bench train-hard; fetch_tasks $bench train-easy ;;
    *)             fetch_tasks $bench train ;;
  esac

  # Teacher traces: Qwen3-32B on 2 GPUs, 8 training tasks per harness
  if start_server $bench teacher --teacher --tensor-parallel-size 2; then
    run_stage $bench collect-teacher \
      uv run exp collect -b $bench $POOL_TASKS $opts
  fi
  stop_server

  # Pool build. A few teacher traces per harness cannot reach a full-size minimum (Reasoning Gym
  # asks for 30), so accept any number above zero.
  # Three benchmarks' pools also use one-gpu's rehearsal traces, so wait for those first.
  case $bench in
    medagentbench|reasoning_gym|bfcl_multiturn) wait_for_one_gpu $bench ;;
  esac
  run_stage $bench build-pool \
    uv run exp build-pool -b $bench --min-trajectories 1 --jobs-dir $SMOKE_DIR/jobs/$bench --out-dir $SMOKE_DIR/pools/$bench

  # Short GRPO (MedAgentBench and Reasoning Gym) from one-gpu's SFT checkpoint: 2 updates of
  # 6 prompts each (the recipe's batch is 24), once rolling out on all three trained harnesses
  # (3h) and once on a single harness.
  case $bench in
    medagentbench|reasoning_gym)
      local init=${GRPO_INIT_DIR:-$SMOKE_DIR/ckpt}/$bench/$QWEN/3h/final
      local single=parity
      if [ $bench = reasoning_gym ]; then
        single=miniswe
        fetch_tasks $bench train     # the RL split (MedAgentBench's was fetched above)
      fi
      wait_for_one_gpu $bench
      for rollout in 3h $single; do
        run_stage $bench grpo-$rollout \
          uv run exp grpo -b $bench --rollout $rollout --init $init --max-steps 2 --batch-size 6 \
            --out $SMOKE_DIR/rl/$bench/grpo-$rollout --backend ${BACKEND:-docker}
      done
      ;;
  esac
}


# ============================================================================================
# Main
# ============================================================================================

# Internal mode: the script calls itself for this, so it runs as a stage with its own log
if [ "$MODE" = fetch ]; then
  fetch_once "$@"
  exit
fi

BENCHMARKS=${*:-medagentbench reasoning_gym bfcl_singleturn bfcl_multiturn appworld}
mkdir -p $SMOKE_DIR/logs
[ -f $STATUS ] || printf "time\tbenchmark\tstage\trc\tseconds\tlog\n" > $STATUS

# The command list starts with the settings every command below ran with
cat >> $COMMANDS <<EOF
# setup/smoke_gpu.sh $MODE $BENCHMARKS, started $(date +%F_%T).
# Commands in the order they ran, from the repo root, after: . setup/eidf/session.sh (on EIDF)
# Settings: BACKEND=${BACKEND:-docker} SANDBOX_API_BASE=${SANDBOX_API_BASE:-} CONCURRENCY=${CONCURRENCY:-5}
EOF

# one-gpu also checks every experiment's reproduction script (no GPU needed)
if [ "$MODE" = one-gpu ]; then
  run_stage all plans bash setup/check_plans.sh $SMOKE_DIR/plans
fi

for bench in $BENCHMARKS; do
  case $MODE in
    one-gpu) one_gpu_stages $bench ;;
    two-gpu) two_gpu_stages $bench ;;
    *)       echo "mode: one-gpu or two-gpu"; exit 2 ;;
  esac
done
# A stage passes when its command exits cleanly, which harbor does even when every attempt in a
# job failed. So also list every job with its errored attempts (ALL ERRORED marks a broken one).
uv run python setup/summarize_jobs.py $SMOKE_DIR/jobs > $SMOKE_DIR/jobs-$MODE.tsv
grep "ALL ERRORED" $SMOKE_DIR/jobs-$MODE.tsv
echo "SMOKE-$MODE-DONE $(date +%F_%T)" | tee -a $DONE
exit
}
