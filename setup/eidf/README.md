# Running on the EIDF GPU service

How this repo was set up and run on EIDF (the cluster the MedAgentBench and Reasoning Gym
experiments, and the GPU smoke tests, ran on). Everything here is a script in this folder
or a step below; nothing depends on an outside helper.

## What you need

- An EIDF project namespace with a kubeconfig and a `kubectl` binary (we keep both in
  `$PROJECT_ROOT/kube/`; `session.sh` finds kubectl there if it is not on `PATH`), and pods you create yourself (label them
  `eidf/user=<you>`; act on pods only by exact name or by that label, never on others').
- Pods that mount the project's CephFS (where the repo is checked out) and the shared model
  store (`/nfs_inf/public/hf/models`, laid out as `<store>/<hf id>`), from an image with git,
  tmux, CUDA and `/usr/bin/python3.12`.
- GPUs: 1 for serving, SFT and evaluating the 4B students; 2 for the Qwen3-32B teacher
  (tensor-parallel 2) and for GRPO.
- Sandboxes run as Kubernetes jobs in the same namespace (`--backend eidf`), so they count
  against the namespace's CPU quota.

## Once per checkout

```bash
git clone <repo> $PROJECT_ROOT/harness-competence && cd $PROJECT_ROOT/harness-competence
cp env.example .env              # (or a symlink to a copy kept outside the checkout) fill in: PROJECT_ROOT, K8S_NAMESPACE, KUEUE_QUEUE, EIDF_*, SHARED_MODEL_STORE, WANDB_*
bash setup/eidf/bootstrap_pod.sh # on the pod you build from (see below)
. setup/eidf/session.sh
setup/install.sh                 # harbor, SkyRL + training/rl overlay, the SFT environment, the root environment (the `exp` CLI)
setup/smoke.sh                   # no-GPU check
```

## Once per pod

```bash
bash setup/eidf/bootstrap_pod.sh
```

It installs uv 0.13.6 in `~/.local/bin` and uv's Python 3.13.18, then checks that the shared
environments run on this pod. They live on CephFS, so every pod uses the same copy, but their
interpreters are pod-local paths; installing the same uv and Python at the same path is what
makes them usable, with no re-sync. It never re-syncs an environment itself: that rewrites it
for every pod at once, including runs in progress. Re-run `setup/install.sh` / `uv sync` only
when no pod is using them.

## Every shell

```bash
. setup/eidf/session.sh
```

It exports `.env`, puts uv on `PATH`, keeps uv's cache pod-local (`/tmp/uv_cache`; CephFS is
slow for its many small files), sets `KUBECONFIG` (default `$PROJECT_ROOT/kube/kubeconfig`),
finds `kubectl`, and sets `POD_IP`. Sandboxes reach a model server on this pod at `http://$POD_IP:8000/v1`:

```bash
EV="--backend eidf --api-base http://127.0.0.1:8000/v1 --sandbox-api-base http://$POD_IP:8000/v1"
```

## Rules that matter on this cluster

- **Long runs in tmux**, logs on CephFS (connections drop). Start tmux first, then run
  `. setup/eidf/session.sh` inside it: a new tmux window does not inherit what an outer shell
  exported when a tmux server is already running.
- **CPU quota.** Each concurrent trial is a sandbox pod. Keep `--concurrency` so that at
  least 20 CPUs of the namespace quota stay free (the smoke tests use 5 per pod).
- **Never kill a `harbor run` mid-trial.** It leaves sandbox jobs behind, and reusing that
  job name later mixes the two runs' trials. Give every run its own job name or jobs folder.
- **Leftover sandboxes.** Each trial's sandbox is a Job running `sleep infinity` with no
  deadline; harbor deletes it when the trial ends, but one left by a crash holds its CPU quota
  until deleted. `setup/eidf/reap_sandboxes.sh` lists yours (labels `eidf/user=$EIDF_USER`,
  `app=harbor-sandbox`); `--delete` removes them. Delete only when none of your runs is in
  progress on any pod: it cannot tell a leftover from a live trial.
- **Stopping a model server.** Send SIGTERM to its process group. vLLM's `VLLM::EngineCore`
  processes can outlive their parent and keep the GPU's memory; if `nvidia-smi` still shows
  memory in use, stop those processes too. The pod itself does not need restarting.
- **Models are never downloaded.** With `SHARED_MODEL_STORE` set, `exp` loads
  `<store>/<hf id>` (e.g. `Qwen/Qwen3-4B`) from the shared store.

## Images

EIDF sandboxes cannot build images, so tasks that need one name a prebuilt image:

- Reasoning Gym: `benchmarks/reasoning_gym/tasks/build_task_image.sh <image>`, then
  `RG_TASK_IMAGE=<image>` in `.env`; `exp tasks -b reasoning_gym` writes it into each task.
- AppWorld: `benchmarks/appworld/tasks/build_appworld_image.sh <image>`, then
  `APPWORLD_IMAGE=<image>` in `.env`; `exp tasks -b appworld` writes it into each task.

Without Docker on the pod, both build with kaniko (`setup/eidf/kaniko_build.sh`): a CPU-only
pod in the namespace (4 CPU) that pushes with a registry secret. It needs, in `.env`,
`KANIKO_DOCKER_SECRET` (a `docker-registry` secret in the namespace), `KANIKO_PVC` (the claim
of the shared filesystem) and `KANIKO_PVC_MOUNT` (where pods mount it, `/ceph_big` here).

## Then

- GPU smoke tests: `setup/smoke_gpu.sh` (see setup/README.md).
- Every experiment's commands: `docs/COMMANDS.md`.
