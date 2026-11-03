#!/bin/bash
# List, or delete, harbor sandbox jobs your runs left behind in the namespace.
#
#   setup/eidf/reap_sandboxes.sh            list them
#   setup/eidf/reap_sandboxes.sh --delete   delete them (and their pods)
#
# Each trial's sandbox is a Kubernetes Job that runs `sleep infinity` with no deadline. Harbor
# deletes it when the trial ends, but one left behind by a crash holds its CPU quota forever.
#
# Only jobs labelled eidf/user=$EIDF_USER and app=harbor-sandbox are selected, so nobody
# else's are touched. Use --delete only when none of your runs is in progress, on any pod: it
# cannot tell a leftover from a live trial. Source setup/eidf/session.sh first.
set -u
: "${EIDF_USER:?source setup/eidf/session.sh first}" "${K8S_NAMESPACE:?}"

SELECTOR="eidf/user=$EIDF_USER,app=harbor-sandbox"

kube() {
  kubectl -n "$K8S_NAMESPACE" "$@"
}

count=$(kube get jobs -l "$SELECTOR" --no-headers 2>/dev/null | wc -l)
echo "sandbox jobs ($SELECTOR): $count"
kube get jobs -l "$SELECTOR" --no-headers \
  -o custom-columns=NAME:.metadata.name,SESSION:.metadata.labels.harbor-session,CREATED:.metadata.creationTimestamp \
  2>/dev/null

if [ "${1:-}" = --delete ] && [ "$count" -gt 0 ]; then
  kube delete jobs -l "$SELECTOR" --cascade=foreground --wait=true && echo "deleted $count"
fi
