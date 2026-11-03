"""Sandbox backends for harbor trials (eval and RL rollouts).

Public default is Docker. Daytona, Modal, E2B, Runloop and GKE use harbor's own backends
with no extra settings here. EIDF (the Kubernetes environment every reported run used)
reads its settings from environment variables, see env.example.
"""
from __future__ import annotations

import os
import platform
import subprocess
from urllib.parse import urlparse, urlunparse

BACKENDS = ["docker", "daytona", "modal", "e2b", "runloop", "gke", "eidf"]

EIDF_VARS = {  # job-config key -> environment variable
    "namespace": "K8S_NAMESPACE",
    "kueue_queue": "KUEUE_QUEUE",
    "eidf_user": "EIDF_USER",
    "eidf_groups": "EIDF_GROUPS",
    "eidf_home": "EIDF_HOME",
}


def _var(name: str, placeholders: bool) -> str:
    val = os.environ.get(name)
    if val:
        return val
    if placeholders:
        return "${" + name + "}"
    raise SystemExit(f"backend eidf needs ${name} (see env.example), or pass --dry-run to render with placeholders")


def environment(bench, backend: str, placeholders: bool = False) -> dict:
    if backend not in BACKENDS:
        raise SystemExit(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
    if backend != "eidf":
        return {"type": backend}
    kwargs = {k: _var(v, placeholders) for k, v in EIDF_VARS.items()}
    kwargs["image_pull_policy"] = "IfNotPresent"
    image = (bench.eval.get("environment") or {}).get("image")
    if image:
        kwargs["image"] = image  # EIDF has no build step; cluster 1 passed this image explicitly
    return {"type": "eidf", "kwargs": kwargs}


def _docker_host_alias() -> str:
    """How a Docker container reaches the host: host.docker.internal on Docker Desktop
    (macOS, Windows); the default bridge gateway on a Linux engine, where that alias is not
    defined unless the compose file maps it (harbor's does not)."""
    if platform.system() != "Linux":
        return "host.docker.internal"
    try:
        gw = subprocess.run(["docker", "network", "inspect", "bridge", "-f", "{{(index .IPAM.Config 0).Gateway}}"],
                            capture_output=True, text=True, timeout=10).stdout.strip()
        if gw:
            return gw
    except Exception:
        pass
    return "172.17.0.1"


def sandbox_endpoint(backend: str, api_base: str, sandbox_api_base: str | None) -> str | None:
    """URL that agents running *inside* a sandbox use to reach the model server.

    Order: --sandbox-api-base, $SANDBOX_API_BASE, then (docker only) api_base with a local
    host name replaced by the host alias. The model server must listen on 0.0.0.0
    (`exp serve` does). Other backends run sandboxes elsewhere and need an explicit URL.
    """
    sandbox_api_base = sandbox_api_base or os.environ.get("SANDBOX_API_BASE")
    if sandbox_api_base:
        return sandbox_api_base
    if backend == "docker":
        u = urlparse(api_base)
        if u.hostname in ("127.0.0.1", "localhost", "0.0.0.0"):
            netloc = _docker_host_alias() + (f":{u.port}" if u.port else "")
            return urlunparse(u._replace(netloc=netloc))
        return api_base
    return None


def host_of(url: str | None) -> str | None:
    if not url:
        return None
    u = urlparse(url)
    return urlunparse(u._replace(path="", params="", query="", fragment=""))
