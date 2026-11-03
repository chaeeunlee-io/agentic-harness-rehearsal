# Single container, no sidecar, no server process needed at all -- unlike
# tau3-bench (which needed a merge to work around EIDFEnvironment's
# single-container limit) and even MedAgentBench (which runs a background
# FHIR server), AppWorld's agent/verifier split is done entirely via disk:
# the agent process opens `AppWorld(task_id, experiment_name=X)`, calls
# `.execute()`/`.close()`, which persists state+logs under
# `$APPWORLD_ROOT/experiments/outputs/X/`; the verifier independently calls
# `evaluate_task(task_id, experiment_name=X)`, which reloads that same
# on-disk state fresh. No HTTP, no MCP, no entrypoint.sh background process.
# Confirmed live against the real appworld==0.1.3 package (see
# adapters/appworld/README.md for the exact commands run to verify this).
FROM python:3.11-slim

ENV APPWORLD_ROOT=/opt/appworld
# Explicit, not left to default (~/.appworld) -- `appworld install` unpacks a
# second bundle ("tests source code") to $APPWORLD_CACHE, which PathStore
# defaults to a HOME-relative path. kaniko builds as one user/HOME;
# EIDFEnvironment's exec() may run agent/verifier commands as a different
# user with a different HOME -- pin this so build-time and run-time always
# agree regardless of which user runs later.
ENV APPWORLD_CACHE=/opt/appworld/.cache
WORKDIR /opt/appworld

# Pinned to 0.1.3.post1 SPECIFICALLY, not 0.1.3 -- bare 0.1.3's metadata
# declares `typer<0.13.0,>=0.12.3`; the .post1 release loosens that to
# `typer>=0.16.0` with no upper bound. Hit two real, different failures
# getting here: (1) an unpinned `pip install appworld` first grabbed some
# newer typer that crashed the CLI outright ("Secondary flag is not valid
# for non-boolean flag"), (2) pinning typer==0.16.1 explicitly then conflicted
# with plain 0.1.3's tight range. 0.1.3.post1 is the exact version verified
# working end-to-end live (install, download data, load_task_ids,
# AppWorld().execute()/.evaluate() all exercised -- see README.md) with
# typer/click left to resolve naturally within ITS declared range.
RUN pip install --no-cache-dir "appworld==0.1.3.post1" \
    && appworld install \
    && appworld download data

# Baked once at image-build time -- no network needed at task runtime.
