#!/bin/bash
# Build and push the AppWorld task image (AppWorld 0.1.3.post1 with its data baked in), from
# the task template's Dockerfile.
#
#   benchmarks/appworld/tasks/build_appworld_image.sh <image>     e.g. docker.io/<you>/appworld:v1
#
# Uses docker where available, else kaniko on EIDF (setup/eidf/kaniko_build.sh; CPU only).
set -euo pipefail

IMAGE="${1:?image to push, e.g. docker.io/<you>/appworld:v1}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONTEXT="$ROOT/benchmarks/appworld/tasks/src/appworld_adapter/task-template/environment"

if command -v docker > /dev/null; then
  docker build -t "$IMAGE" "$CONTEXT" && docker push "$IMAGE"
else
  bash "$ROOT/setup/eidf/kaniko_build.sh" appworld "$CONTEXT" "$IMAGE"
fi
