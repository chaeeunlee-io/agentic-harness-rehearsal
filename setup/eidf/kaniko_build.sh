#!/bin/bash
# Build a Docker image and push it to a registry, from EIDF, where pods have no Docker.
# It starts a CPU-only kaniko pod in the namespace (4 CPU, 16 GiB, counted against the CPU
# quota), which builds the image and pushes it.
#
#   setup/eidf/kaniko_build.sh <name> <context dir> <destination image>
#
#   <name>                names the pod: kaniko-<name>-<random suffix>
#   <context dir>         holds the Dockerfile; must be on the shared filesystem (CephFS), which
#                         the kaniko pod mounts at the same path
#   <destination image>   e.g. docker.io/<you>/<image>:<tag>
#
# Source setup/eidf/session.sh first. Needs, in .env:
#   KANIKO_DOCKER_SECRET  a Kubernetes secret (type dockerconfigjson) with push access to the
#                         registry; create it with
#                         kubectl create secret docker-registry <name> --docker-username=... --docker-password=...
#   KANIKO_PVC            the PersistentVolumeClaim of the shared filesystem
#   KANIKO_PVC_MOUNT      where our pods mount it (default /ceph_big)
#   KANIKO_IMAGE          optional: the kaniko executor image (default gcr.io/kaniko-project/executor:latest)
#
# The pod's name is printed. Follow the build with:
#   kubectl -n $K8S_NAMESPACE logs -f <pod>
set -euo pipefail

NAME="${1:?name}"
CONTEXT="$(realpath "${2:?context dir}")"
DESTINATION="${3:?destination image}"

: "${K8S_NAMESPACE:?source setup/eidf/session.sh}" "${EIDF_USER:?}" "${KUEUE_QUEUE:?}"
: "${KANIKO_DOCKER_SECRET:?set in .env (see this script)}" "${KANIKO_PVC:?set in .env (see this script)}"
MOUNT="${KANIKO_PVC_MOUNT:-/ceph_big}"

case "$CONTEXT" in
  "$MOUNT"/*) ;;
  *) echo "context $CONTEXT is not under $MOUNT (the shared filesystem)" >&2; exit 1 ;;
esac
if [ ! -f "$CONTEXT/Dockerfile" ]; then
  echo "no Dockerfile in $CONTEXT" >&2
  exit 1
fi

# The pod. The eidf/* annotations and labels and the Kueue queue label are the ones our other
# pods carry. The registry secret is mounted as /kaniko/.docker/config.json (kaniko's login),
# and the shared filesystem at the same path as on our pods, so the context path works as is.
kubectl -n "$K8S_NAMESPACE" create -f - <<YAML
apiVersion: v1
kind: Pod
metadata:
  generateName: kaniko-${NAME}-
  namespace: ${K8S_NAMESPACE}
  annotations:
    eidf/user: ${EIDF_USER}
    groups: ${EIDF_GROUPS:-}
    home: ${EIDF_HOME:-}
    login_user: ${EIDF_USER}
  labels:
    eidf/user: ${EIDF_USER}
    app: kaniko-build
    kueue.x-k8s.io/queue-name: ${KUEUE_QUEUE}
spec:
  restartPolicy: Never
  containers:
  - name: kaniko
    image: ${KANIKO_IMAGE:-gcr.io/kaniko-project/executor:latest}
    args:
    - --dockerfile=Dockerfile
    - "--context=dir://${CONTEXT}"
    - "--destination=${DESTINATION}"
    - --snapshot-mode=redo
    - --compressed-caching=false
    - --cache=false
    resources:
      requests: {cpu: '4', memory: 16Gi}
      limits:   {cpu: '4', memory: 16Gi}
    volumeMounts:
    - name: docker-config
      mountPath: /kaniko/.docker
    - name: shared
      mountPath: ${MOUNT}
  volumes:
  - name: docker-config
    secret:
      secretName: ${KANIKO_DOCKER_SECRET}
      items:
      - key: .dockerconfigjson
        path: config.json
  - name: shared
    persistentVolumeClaim:
      claimName: ${KANIKO_PVC}
YAML
