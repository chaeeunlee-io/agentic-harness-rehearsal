"""exp command line. Run `uv run exp --help` or `uv run exp <command> --help`."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from exp import config as C, render as R
from exp.backends import BACKENDS


# --------------------------------------------------------------------------- helpers
def _show(argv, env=None, cwd=None):
    pre = "".join(f"{k}={shlex.quote(str(v))} " for k, v in (env or {}).items())
    where = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
    print(where + pre + " ".join(shlex.quote(str(a)) for a in argv), flush=True)


def _run(argv, env=None, cwd=None, dry=False) -> int:
    _show(argv, env, cwd)
    if dry:
        return 0
    full = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}}
    return subprocess.call([str(a) for a in argv], env=full, cwd=cwd)


def _experiment_options(p):
    """Options shared by `exp plan` and `exp run`."""
    p.add_argument("experiment", help="see `exp list experiments`")
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--sandbox-api-base")
    p.add_argument("--port", type=int, default=8000)


def _setting(a, b):
    """Which model to serve / name the run by: base, a pool, a GRPO run, or the teacher."""
    if getattr(a, "rl", None):
        return f"grpo-{a.rl}"
    if getattr(a, "teacher", False):
        return "teacher"
    return a.pool or "base"


# --------------------------------------------------------------------------- commands
def _benchmarks_of(e: dict) -> str:
    """The benchmarks an experiment runs on, e.g. "MAB, RG (easy, hard)", from its settings."""
    splits: dict[str, list[str]] = {}
    for s in e["settings"]:
        b = C.benchmark(s["benchmark"])
        name = b.cfg.get("short", s["benchmark"])
        split = s.get("split", "test")
        splits.setdefault(name, [])
        if split not in splits[name]:
            splits[name].append(split)
    parts = []
    for name, sp in splits.items():
        if sp == ["test"]:
            parts.append(name)
        else:   # a benchmark with more than one test split: name them (test -> easy on Reasoning Gym)
            parts.append(f"{name} ({', '.join('easy' if x == 'test' else x.removeprefix('test-') for x in sp)})")
    return ", ".join(parts)


def cmd_list(a):
    if a.what == "benchmarks":
        for n in C.benchmark_names():
            b = C.benchmark(n)
            print(f"{n:16s} {b.cfg['display']:18s} trained={b.harnesses('trained')} heldout={b.harnesses('heldout')} splits={list(b.cfg['splits'])}")
    elif a.what == "pools":
        for n in ([a.benchmark] if a.benchmark else C.benchmark_names()):
            b = C.benchmark(n)
            for pn, p in b.pools().items():
                print(f"{n:16s} {pn:22s} {p.get('paper',''):30s} {p.get('file') or '(base model)'}")
    elif a.what == "models":
        for k, m in C.models().items():
            print(f"{k:16s} {m['hf']:24s} {m.get('role','')}")
    elif a.what == "experiments":
        rows = [(k, _benchmarks_of(e), e["title"]) for k, e in C.experiments().items()]
        width = max(len(b) for _, b, _ in rows)
        for k, b, title in rows:
            print(f"{k:32s} {b:{width}s}  {title}")
    return 0


def cmd_serve(a):
    b = C.benchmark(a.benchmark)
    model = "qwen3-32b" if a.teacher else a.model
    path = a.checkpoint or R.model_path(b, model, None if a.teacher else a.pool, a.rl, a.rl_step)
    return _run(R.serve_cmd(b, model, path, a.port, a.served_model, a.tensor_parallel_size, a.harness), dry=a.dry_run)


def cmd_wait(a):
    """Wait until the model server answers. With --pid, give up as soon as that process exits
    (a server that crashed at startup), instead of waiting out the whole timeout."""
    url = a.api_base.rstrip("/") + "/models"
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"server up: {url}")
                    return 0
        except Exception:
            pass
        if a.pid and not _process_running(a.pid):
            print(f"server process {a.pid} exited before answering: see the server's log", file=sys.stderr)
            return 1
        time.sleep(5)
    print(f"timed out waiting for {url}", file=sys.stderr)
    return 1


def _process_running(pid: int) -> bool:
    """True while process `pid` is running."""
    try:
        os.kill(pid, 0)  # signal 0 sends nothing; it only checks that the process exists
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists but belongs to another user
        return True
    # On Linux, a process that has exited but whose parent has not collected it yet still
    # exists as a "zombie" (state Z). Count that as exited.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state != "Z"


def cmd_stop(a):
    """Stop a model server and wait until the GPUs are free again.

    Start the server in its own process group, so it can be stopped as a whole:
        setsid uv run exp serve ... & SERVER=$!
        uv run exp stop --pid $SERVER
    This sends SIGTERM to that group, then waits (up to 3 minutes) until every GPU uses less
    than 1 GB. vLLM's engine processes (VLLM::EngineCore) can outlive their parent and keep
    holding GPU memory; if memory is still in use after the wait, they are stopped too. It
    assumes the machine's GPUs are yours (as on a pod of your own). Without --pid it only
    does the waiting and the cleanup.
    """
    if a.pid:
        try:
            os.killpg(a.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone

    for _ in range(60):
        if _busiest_gpu_mb() < 1000:
            break
        time.sleep(3)
    else:
        print("GPU memory still in use: stopping leftover vLLM engine processes")
        subprocess.call(["pkill", "-TERM", "-f", "VLLM::EngineCore"])
        time.sleep(10)
        subprocess.call(["pkill", "-KILL", "-f", "VLLM::EngineCore"])
        time.sleep(5)

    used = _busiest_gpu_mb()
    if used >= 1000:
        print(f"GPU memory still in use ({used} MiB on the busiest GPU)", file=sys.stderr)
        return 1
    print("server stopped; GPUs free")
    return 0


def _busiest_gpu_mb() -> int:
    """Memory in use on the busiest GPU, in MiB (0 if there is no nvidia-smi)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return max((int(x) for x in out.split()), default=0)


# --- Reusing finished training ----------------------------------------------------------------
# Experiments share checkpoints (the same benchmark, model and pool), so `exp sft` and `exp grpo`
# skip a run that has already finished in full. Each successful run writes a record of how it was
# trained next to its output; only a full-scale run (no --max-steps / --batch-size) with the same
# inputs is reused. --force always retrains.

SFT_RECORD = "exp_sft.json"     # in <checkpoint>/final/
GRPO_RECORD = "exp_grpo.json"   # in the GRPO output folder


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def _reusable(record_path: Path, wanted: dict) -> tuple[bool, str]:
    """Whether a finished run can be reused: its record says it ran in full with these inputs."""
    if not record_path.exists():
        return False, "no record of a finished run"
    record = json.loads(record_path.read_text())
    if record.get("max_steps") or record.get("batch_size"):
        return False, "it was a short test run"
    for key, value in wanted.items():
        if record.get(key) != value:
            return False, f"it was trained with a different {key}"
    return True, ""


def cmd_sft(a):
    b = C.benchmark(a.benchmark)
    out = Path(a.out) if a.out else R.checkpoint_dir(b.name, a.model, a.pool)
    final = out / "final"
    pool_file = b.pool(a.pool).get("file")
    data = str(Path(a.data).resolve()) if a.data else (str(C.ROOT / pool_file) if pool_file else None)
    wanted = {"model": a.model, "data": data}
    complete = (final / "config.json").exists() and any(final.glob("*.safetensors"))
    if complete and not (a.force or a.max_steps or a.dry_run):
        record = final / SFT_RECORD
        if not record.exists():
            # Trained before records existed: reuse it, but say so
            print(f"{C.rel(final)} exists (no training record): reusing it; --force retrains it")
            return 0
        ok, why = _reusable(record, wanted)
        if ok:
            print(f"{C.rel(final)} exists: skipping training (--force retrains it)")
            return 0
        print(f"{C.rel(final)} exists but {why}: training again")
    argv = R.sft_cmd(b, a.model, a.pool, Path(a.out) if a.out else None, a.wandb_run, data=a.data, max_steps=a.max_steps)
    argv = ["uv", "run", "--project", str(C.ROOT / "setup" / "sft")] + argv
    # wandb (only with --wandb-run) keeps its local files in runs/wandb/, not the repo root
    env = {"PYTHONPATH": str(C.ROOT), "WANDB_DIR": str(R.RUNS)}
    # SFT trains on one GPU, as the paper did. With several visible, the trainer would spread the
    # model over all of them, and the fused loss kernel fails on tensors from different GPUs.
    # Set CUDA_VISIBLE_DEVICES yourself to pick another GPU.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    rc = _run(argv, env=env, cwd=C.ROOT, dry=a.dry_run)
    if rc == 0 and not a.dry_run:
        _write_record(final / SFT_RECORD, {**wanted, "max_steps": a.max_steps,
                                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return rc


def cmd_eval(a):
    b = C.benchmark(a.benchmark)
    model = "qwen3-32b" if a.teacher else a.model
    setting = _setting(a, b)
    proto = R.teacher_protocol(b) if a.teacher else {}
    rc = 0
    for h in b.harnesses(a.harness):
        name = a.job_name or R.job_name(b, model, setting, h, a.split)
        job = R.eval_job(b, h, job_name=name, split=a.split, backend=a.backend, api_base=a.api_base,
                         sandbox_api_base=a.sandbox_api_base, served=a.served_model,
                         jobs_dir=Path(a.jobs_dir) if a.jobs_dir else None, concurrency=a.concurrency,
                         n_attempts=a.n_attempts or proto.get("n_attempts"), temperature=proto.get("temperature"),
                         job_extra=proto.get("job"), placeholders=a.dry_run,
                         tasks=Path(a.tasks_dir) if a.tasks_dir else None, limit=a.limit)
        cfg_dir = R.RUNS / "job_configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        path = cfg_dir / f"{name}.yaml"
        path.write_text(R.dump_job(job))
        print(f"# {b.name} / {setting} / {h} / {a.split}  ->  {C.rel(path)}")
        argv = ["uv", "run", "--project", str(C.ROOT / "third_party" / "harbor"), "harbor", "run", "-c", str(path), "-y"]
        env = {"PYTHONPATH": str(C.ROOT), "OPENAI_API_BASE": a.api_base, "OPENAI_API_KEY": "EMPTY"}
        rc |= _run(argv, env=env, cwd=C.ROOT, dry=a.dry_run)
    return rc


def cmd_oracle(a):
    """Run harbor's oracle agent (each task's reference solution) on a few tasks: checks the
    task directories, the sandbox backend and the verifier without any model or GPU."""
    b = C.benchmark(a.benchmark)
    tasks = Path(a.tasks_dir) if a.tasks_dir else R.tasks_root()
    split_dir = tasks / b.name / a.split
    names = b.split_tasks(a.split) or (sorted(p.name for p in split_dir.iterdir() if p.is_dir()) if split_dir.is_dir() else [])
    job = {"job_name": f"oracle-{b.name}-{a.split}", "jobs_dir": str(R.RUNS / "jobs" / "oracle"), "n_attempts": 1,
           "n_concurrent_trials": a.n, "environment": __import__("exp.backends", fromlist=["x"]).environment(b, a.backend, placeholders=a.dry_run),
           "agents": [{"name": "oracle"}], "datasets": [{"path": str(split_dir), "task_names": names[: a.n]}]}
    path = R.RUNS / "job_configs" / f"{job['job_name']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(R.dump_job(job))
    print(f"# oracle on {min(a.n, len(names))} {b.name}/{a.split} tasks -> {C.rel(path)}; every trial should score 1")
    argv = ["uv", "run", "--project", str(C.ROOT / "third_party" / "harbor"), "harbor", "run", "-c", str(path), "-y"]
    return _run(argv, env={"PYTHONPATH": str(C.ROOT)}, cwd=C.ROOT, dry=a.dry_run)


def cmd_collect(a):
    b = C.benchmark(a.benchmark)
    hv = b.cfg.get("collect")
    if not hv:
        raise SystemExit(f"{b.name}: no trace collection configured")
    entries = (b.cfg.get("rehearsal") or []) if a.rehearsal else hv["jobs"]
    if a.rehearsal and not entries:
        raise SystemExit(f"{b.name}: no rehearsal trace collection configured")
    rc = 0
    for e in entries:
        if a.harness and e["harness"] != a.harness:
            continue
        if a.job and e["job"] != a.job:
            continue
        if e.get("demonstrator"):
            print(f"# {e['job']}: serve {e['demonstrator']} first; tasks: exp tasks -b {b.name} --split {e['split']}")
        job = R.collect_job(b, e, backend=a.backend, api_base=a.api_base, sandbox_api_base=a.sandbox_api_base,
                            served=a.served_model, jobs_dir=Path(a.jobs_dir) if a.jobs_dir else None,
                            placeholders=a.dry_run, tasks=Path(a.tasks_dir) if a.tasks_dir else None,
                            limit=a.limit, n_attempts=a.n_attempts, concurrency=a.concurrency)
        path = R.RUNS / "job_configs" / f"{e['job']}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(R.dump_job(job))
        print(f"# collect {b.name} / {e['harness']} / {e.get('split', hv.get('split', 'train'))} as {e['job']} -> {C.rel(path)}")
        argv = ["uv", "run", "--project", str(C.ROOT / "third_party" / "harbor"), "harbor", "run", "-c", str(path), "-y"]
        rc |= _run(argv, env={"PYTHONPATH": str(C.ROOT), "OPENAI_API_BASE": a.api_base, "OPENAI_API_KEY": "EMPTY"},
                   cwd=C.ROOT, dry=a.dry_run)
    return rc


def cmd_build_pool(a):
    from exp import pools
    return pools.main(a)


def cmd_grpo(a):
    b = C.benchmark(a.benchmark)
    out = Path(a.out).resolve() if a.out else R.rl_dir(b.name, a.model, a.rollout)
    init = str(Path(a.init).resolve()) if a.init else str(R.checkpoint_dir(b.name, a.model, b.rl["init_pool"]) / "final")
    wanted = {"model": a.model, "rollout": a.rollout, "init": init}
    if not (a.force or a.max_steps or a.batch_size or a.dry_run):
        ok, _ = _reusable(out / GRPO_RECORD, wanted)
        if ok:
            print(f"{C.rel(out)} has a finished GRPO run: skipping training (--force retrains it)")
            return 0
    argv, env, cwd = R.grpo(b, a.model, a.rollout, init=a.init, backend=a.backend, run_name=a.run_name,
                            placeholders=a.dry_run, max_steps=a.max_steps, out=a.out, batch_size=a.batch_size)
    rc = _run(argv, env=env, cwd=cwd, dry=a.dry_run)
    if rc == 0 and not a.dry_run:
        _write_record(out / GRPO_RECORD, {**wanted, "max_steps": a.max_steps, "batch_size": a.batch_size,
                                          "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return rc


def cmd_plan(a):
    """Print the bash script that reproduces one experiment from the shipped pools."""
    print(_plan_script(a))
    return 0


def _check_backend(a):
    """Stop before a run starts if its sandboxes could not work, instead of hours later at the
    first evaluation (the plan scripts carry on past a failed evaluation)."""
    from urllib.parse import urlparse
    problems = []
    if a.backend == "eidf":
        from exp.backends import EIDF_VARS
        problems += [f"${v} is not set (see env.example)" for v in EIDF_VARS.values() if not os.environ.get(v)]
    if a.backend != "docker":
        url = a.sandbox_api_base or ""
        if not urlparse(url).hostname:
            problems.append(f"--sandbox-api-base {url!r} has no host: give a URL the sandboxes can reach, "
                            "e.g. http://<this machine\'s IP>:8000/v1")
    if problems:
        raise SystemExit("not starting:\n  " + "\n  ".join(problems) +
                         "\n(on EIDF: run . setup/eidf/session.sh in the shell you start the run from, inside tmux)")


def cmd_run(a):
    """Run one experiment: write its script (the one `exp plan` prints) to
    runs/plans/<experiment>.sh, then run it. With --dry-run, only write it."""
    script = C.ROOT / "runs" / "plans" / f"{a.experiment}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_plan_script(a) + "\n")
    script.chmod(0o755)
    print(f"wrote {C.rel(script)}")
    if a.dry_run:
        return 0
    _check_backend(a)
    print(f"running: bash {C.rel(script)}", flush=True)
    return subprocess.call(["bash", str(script)], cwd=C.ROOT)


def _plan_script(a) -> str:
    """The bash script for one experiment: every command, in order."""
    e = C.experiments().get(a.experiment)
    if not e:
        raise SystemExit(f"unknown experiment {a.experiment!r}; see `exp list experiments`")
    exe = "uv run exp"
    be = f"--backend {a.backend}" + (f" --sandbox-api-base {a.sandbox_api_base}" if a.sandbox_api_base else "")
    out = ["#!/usr/bin/env bash", f"# {e['title']}",
           f"# generated by: uv run exp plan {a.experiment} (uv run exp run {a.experiment} writes this script and runs it)",
           "# Needs: setup/install.sh done, GPUs, a sandbox backend (on EIDF: . setup/eidf/session.sh).",
           "# Edit PORT / GPU placement as needed.",
           "set -euo pipefail",
           f"cd {shlex.quote(str(C.ROOT))}   # the checkout this script was generated from",
           f"PORT={a.port}",
           f"API=http://127.0.0.1:$PORT/v1", ""]
    done_tasks, done_sft, done_rl = set(), set(), set()
    for s in e["settings"]:
        bn, model, split = s["benchmark"], s["model"], s.get("split", "test")
        hs = s.get("harnesses", "trained")
        hs = hs if isinstance(hs, str) else ",".join(hs)
        for sp in [split] + (["train"] if s.get("rl") else []):
            if (bn, sp) not in done_tasks:
                out.append(f"{exe} tasks --benchmark {bn} --split {sp}")
                done_tasks.add((bn, sp))
        runs = []
        if s.get("teacher"):
            runs.append(("--teacher", "teacher"))
        for p in s.get("pools", []):
            if p != "base" and (bn, model, p) not in done_sft:
                out.append(f"{exe} sft --benchmark {bn} --model {model} --pool {p}")
                done_sft.add((bn, model, p))
            runs.append((f"--pool {p}", p))
        for r in s.get("rl", []):
            init_pool = C.benchmark(bn).rl["init_pool"]
            if (bn, model, init_pool) not in done_sft:
                out.append(f"{exe} sft --benchmark {bn} --model {model} --pool {init_pool}")
                done_sft.add((bn, model, init_pool))
            # one GRPO run per rollout, even when several settings evaluate it (e.g. two test splits)
            if (bn, model, r) not in done_rl:
                out.append(f"{exe} grpo --benchmark {bn} --model {model} --rollout {r} --backend {a.backend}")
                done_rl.add((bn, model, r))
            runs.append((f"--rl {r}", f"grpo-{r}"))
        for flag, label in runs:
            out.append(f"# --- {bn} / {label} ---")
            served_model = "qwen3-32b" if flag == "--teacher" else model
            for harnesses, serve_harness in _serving_groups(C.benchmark(bn), served_model, hs):
                out += [f"setsid {exe} serve --benchmark {bn} --model {model} {flag}{serve_harness} --port $PORT & SERVER=$!",
                        f"{exe} wait --api-base $API --pid $SERVER",
                        f"{exe} eval --benchmark {bn} --model {model} {flag} --harness {harnesses} --split {split} --api-base $API {be} || true",
                        f"{exe} stop --pid $SERVER"]
            out.append("")
    out.append(f"{exe} table {a.experiment}")
    return "\n".join(out)


def _serving_groups(b, model: str, harness_spec: str) -> list[tuple[str, str]]:
    """Split a harness set into groups that one model server can serve.

    Returns (harnesses to evaluate, extra `exp serve` option) pairs. Normally that is one pair:
    the whole set, served once. A model served differently per harness (a `per_harness` entry
    under serve.per_model in the benchmark config, e.g. Gemma on MedAgentBench: fc with a
    tool parser, the rest without) gets one server per distinct serving setup, started with
    --harness so `exp serve` picks that setup.
    """
    per_model = (b.cfg["serve"].get("per_model") or {}).get(model) or {}
    if not per_model.get("per_harness"):
        return [(harness_spec, "")]
    groups: dict[str, list[str]] = {}
    for h in b.harnesses(harness_spec):
        setup = json.dumps(R.serve_settings(b, model, h), sort_keys=True)
        groups.setdefault(setup, []).append(h)
    return [(",".join(hs), f" --harness {hs[0]}") for hs in groups.values()]


def cmd_tasks(a):
    from exp import tasks
    return tasks.main(a)


def cmd_table(a):
    from exp import table
    return table.main(a)


# --------------------------------------------------------------------------- parser
def _common_setting(p, teacher=True):
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--model", "-m", default="qwen3-4b")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pool", help="SFT pool name (see `exp list pools`); default: base model")
    g.add_argument("--rl", help="GRPO run: 3h or the single rollout harness")
    if teacher:
        g.add_argument("--teacher", action="store_true", help="the Qwen3-32B teacher")


def build_parser():
    ap = argparse.ArgumentParser(prog="exp", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list benchmarks, pools, models or experiments")
    p.add_argument("what", choices=["benchmarks", "pools", "models", "experiments"])
    p.add_argument("--benchmark", "-b")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("tasks", help="generate harbor task directories for a split")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--split", default="test")
    p.add_argument("--out", help="default runs/tasks/<benchmark>/<split>")
    p.add_argument("--from", dest="from_dir", help="copy the split from an existing task directory instead of generating")
    p.add_argument("--source", choices=["adapter", "registry"], help="where tasks come from (default: what the clusters used)")
    p.add_argument("--source-commit", help="upstream commit to generate from (MedAgentBench, BFCL single-turn)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("serve", help="serve a model with vLLM, with the benchmark's as-run flags")
    _common_setting(p)
    p.add_argument("--checkpoint", help="serve this path instead of the default for the setting")
    p.add_argument("--rl-step", default="final")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model", default=R.SERVED)
    p.add_argument("--tensor-parallel-size", type=int)
    p.add_argument("--harness", help="serve for this harness, where a model's serving differs per harness (Gemma on MedAgentBench)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("wait", help="wait until a model server answers")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--pid", type=int, help="the server's process id: stop waiting if it exits")
    p.set_defaults(fn=cmd_wait)

    p = sub.add_parser("stop", help="stop a model server and wait until the GPUs are free")
    p.add_argument("--pid", type=int, help="the server's process id, started with setsid (see `exp stop --help`)")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("sft", help="supervised fine-tuning on one pool")
    p.add_argument("--max-steps", type=int, help="small test run: stop after N optimizer steps")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--model", "-m", default="qwen3-4b")
    p.add_argument("--pool", required=True)
    p.add_argument("--data", help="train on this JSONL instead of the shipped pool (e.g. one rebuilt by build-pool)")
    p.add_argument("--out", help="default runs/checkpoints/<benchmark>/<model>/<pool>")
    p.add_argument("--wandb-run")
    p.add_argument("--force", action="store_true", help="retrain even if the checkpoint already exists")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_sft)

    p = sub.add_parser("eval", help="evaluate a served model on one or more harnesses")
    _common_setting(p)
    p.add_argument("--limit", type=int, help="small test run: N tasks spread evenly over the split")
    p.add_argument("--n-attempts", type=int, help="small test run: attempts per task (default: the protocol's)")
    p.add_argument("--harness", default="all", help="a harness name, or trained / heldout / all")
    p.add_argument("--split", default="test")
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="model server URL, as seen from this machine")
    p.add_argument("--sandbox-api-base", help="model server URL as seen from inside a sandbox (default: derived for docker)")
    p.add_argument("--served-model", default=R.SERVED)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--jobs-dir")
    p.add_argument("--tasks-dir", help="default runs/tasks")
    p.add_argument("--job-name")
    p.add_argument("--dry-run", action="store_true", help="write the job config and print the command only")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("collect", help="collect teacher traces on the training split (serve the teacher first)")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--harness", help="only this harness (default: every job)")
    p.add_argument("--rehearsal", action="store_true",
                   help="collect the rehearsal traces instead (config `rehearsal:`; serve each one's demonstrator)")
    p.add_argument("--job", help="only the job with this name")
    p.add_argument("--limit", type=int, help="small test run: N tasks spread evenly over the split")
    p.add_argument("--n-attempts", type=int, help="small test run: attempts per task (default: the protocol's)")
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--sandbox-api-base")
    p.add_argument("--served-model", default=R.SERVED)
    p.add_argument("--concurrency", type=int, help="concurrent trials (default: the benchmark's eval concurrency)")
    p.add_argument("--jobs-dir")
    p.add_argument("--tasks-dir")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("build-pool", help="build SFT pools from collected traces (config `build:`)")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--pool", help="build only the step that writes this setting (default: every step, in order)")
    p.add_argument("--jobs-dir", help="default runs/jobs/<benchmark>")
    p.add_argument("--out-dir", help="default runs/pools/<benchmark>")
    p.add_argument("--tokenizer", help="student tokenizer for the builders' render check (default Qwen/Qwen3-4B; a local path works)")
    p.add_argument("--min-trajectories", type=int,
                   help="small test run: accept fewer solved trajectories per harness than the builder's minimum (Reasoning Gym: 30)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_build_pool)

    p = sub.add_parser("oracle", help="run reference solutions on a few tasks (no model, no GPU)")
    p.add_argument("--benchmark", "-b", required=True, choices=C.benchmark_names())
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--tasks-dir")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_oracle)

    p = sub.add_parser("grpo", help="GRPO from the 3H SFT checkpoint")
    p.add_argument("--max-steps", type=int, help="small test run: stop (and save) after N training steps")
    p.add_argument("--batch-size", type=int, help="small test run: prompts per update, a multiple of the rollout harnesses (default: the recipe's 24)")
    p.add_argument("--out", help="output folder (default runs/rl/<benchmark>/<model>/grpo-<rollout>)")
    p.add_argument("--benchmark", "-b", required=True, choices=["medagentbench", "reasoning_gym"])
    p.add_argument("--model", "-m", default="qwen3-4b")
    p.add_argument("--rollout", required=True, help="3h, or the single harness to roll out on")
    p.add_argument("--init", help="initial checkpoint (default: this model's 3H SFT checkpoint)")
    p.add_argument("--backend", default="docker", choices=BACKENDS)
    p.add_argument("--run-name")
    p.add_argument("--force", action="store_true", help="retrain even if a finished run is already there")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_grpo)

    p = sub.add_parser("plan", help="print the bash script that reproduces one experiment")
    _experiment_options(p)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="run one experiment: write its script to runs/plans/<experiment>.sh and run it")
    _experiment_options(p)
    p.add_argument("--dry-run", action="store_true", help="only write the script")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("table", help="rebuild a paper table from finished runs (or --shipped results)")
    p.add_argument("--format", choices=["table", "cells"], default="table",
                   help="table = markdown (default); cells = the unified schema of results/cells.csv")
    p.add_argument("--out", help="write cells to this file instead of stdout")
    p.add_argument("experiment")
    p.add_argument("--shipped", action="store_true", help="use the shipped results/ instead of runs/")
    p.add_argument("--jobs-dir", help="default runs/jobs")
    p.set_defaults(fn=cmd_table)

    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0
