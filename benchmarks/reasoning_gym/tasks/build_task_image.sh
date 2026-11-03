#!/bin/bash
# Build and push the Reasoning Gym task image, for sandbox backends that cannot build one
# themselves (eidf).
#
#   benchmarks/reasoning_gym/tasks/build_task_image.sh <image>     e.g. docker.io/<you>/reasoning-gym:v1
#
# Every Reasoning Gym task has the same environment/Dockerfile (reasoning-gym on the t-bench
# Python 3.13 base), so the image is built once, from the first task's copy. Steps:
#   1. uv run exp tasks -b reasoning_gym --split test     (fetches the tasks)
#   2. this script
#   3. set RG_TASK_IMAGE=<image> in .env and run `exp tasks` again: it names the image in
#      every task.toml
# Uses docker where available, else kaniko on EIDF (setup/eidf/kaniko_build.sh).
set -euo pipefail

IMAGE="${1:?image to push, e.g. docker.io/<you>/reasoning-gym:v1}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

TASK_ENV=$(ls -d "$ROOT"/runs/tasks/reasoning_gym/*/*/environment 2>/dev/null | head -1)
if [ -z "$TASK_ENV" ]; then
  echo "no fetched tasks: run exp tasks -b reasoning_gym --split test first" >&2
  exit 1
fi

# The build context: a folder holding only that Dockerfile
CONTEXT="$ROOT/runs/image_ctx/reasoning_gym"
rm -rf "$CONTEXT"
mkdir -p "$CONTEXT"
cp "$TASK_ENV/Dockerfile" "$CONTEXT/"

if command -v docker > /dev/null; then
  docker build -t "$IMAGE" "$CONTEXT" && docker push "$IMAGE"
else
  bash "$ROOT/setup/eidf/kaniko_build.sh" reasoning-gym "$CONTEXT" "$IMAGE"
fi
