# Shell setup for an EIDF pod. Source it in every new shell or tmux window:
#
#     . setup/eidf/session.sh
#
# It exports every variable in .env, then sets:
#   PATH             uv (~/.local/bin) first; kubectl's folder appended if kubectl is not on PATH
#   UV_CACHE_DIR     /tmp/uv_cache: pod-local, because CephFS is slow for uv's many small files
#   UV_LINK_MODE     copy: the cache and the environments are on different filesystems
#   UV_HTTP_TIMEOUT  120 seconds
#   KUBECONFIG       $PROJECT_ROOT/kube/kubeconfig, unless already set
#   POD_IP           this pod's address: how sandboxes reach a model server running here
# and changes to the repo root. Run setup/eidf/bootstrap_pod.sh once per pod first.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ ! -f "$REPO/.env" ]; then
  echo "session: $REPO/.env missing; cp env.example .env and fill it in (setup/eidf/README.md)" >&2
  return 1 2>/dev/null || exit 1      # return when sourced, exit when run as a script
fi

# Export every variable .env defines
set -a
. "$REPO/.env"
set +a

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv_cache}
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120}
export KUBECONFIG=${KUBECONFIG:-$PROJECT_ROOT/kube/kubeconfig}

# kubectl (the eidf sandbox backend calls it): the one on PATH, else the one kept next to the
# kubeconfig. Appended, so uv in ~/.local/bin stays first.
command -v kubectl >/dev/null || export PATH="$PATH:$(dirname "$KUBECONFIG")"

export POD_IP=$(hostname -i | awk '{print $1}')
cd "$REPO"

echo "session: repo=$REPO ns=${K8S_NAMESPACE:-?} pod_ip=$POD_IP" \
     "uv=$(uv --version 2>/dev/null | awk '{print $2}')" \
     "kubectl=$(command -v kubectl >/dev/null && echo ok || echo MISSING)" \
     "models=${SHARED_MODEL_STORE:-hub}"
