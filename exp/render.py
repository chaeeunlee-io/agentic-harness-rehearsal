"""Render serve / SFT / eval / GRPO invocations from configs/.

Pure functions: nothing here runs anything. The CLI decides whether to print or execute.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import yaml

from exp import backends
from exp.config import ROOT, Benchmark, model as model_cfg, model_source

RUNS = ROOT / "runs"
SERVED = "policy"  # served model name used by `exp serve`; any name works


def fmt(obj, ctx: dict):
    """Fill {placeholders} in strings, recursively, leaving unknown braces alone."""
    if isinstance(obj, str):
        out = obj
        for k, v in ctx.items():
            if v is not None:
                out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(obj, dict):
        return {k: fmt(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fmt(v, ctx) for v in obj]
    return obj


# --------------------------------------------------------------------------- paths
def checkpoint_dir(bench: str, model: str, pool: str) -> Path:
    return RUNS / "checkpoints" / bench / model / pool


def rl_dir(bench: str, model: str, rollout: str) -> Path:
    return RUNS / "rl" / bench / model / f"grpo-{rollout}"


def tasks_root() -> Path:
    return RUNS / "tasks"


def model_path(b: Benchmark, model: str, pool: str | None, rl: str | None = None, rl_step: str = "final") -> str:
    """What `exp serve` serves for a setting: the base model, an SFT checkpoint, or a GRPO export."""
    if rl:
        d = rl_dir(b.name, model, rl) / "exports"
        if rl_step != "final":
            return str(d / f"global_step_{rl_step}" / "policy")
        steps = sorted((int(p.name.split("_")[-1]), p) for p in d.glob("global_step_*") if p.name.split("_")[-1].isdigit()) \
            if d.is_dir() else []
        # the paper evaluates the final export (step 12 on MAB, ~14 on RG) and step 5 (--rl-step 5)
        return str(steps[-1][1] / "policy") if steps else str(d / "global_step_<final>" / "policy")
    if pool in (None, "base"):
        return model_source(model)
    return str(checkpoint_dir(b.name, model, pool) / "final")


# --------------------------------------------------------------------------- serve
def serve_settings(b: Benchmark, model: str, harness: str | None = None) -> dict:
    """The benchmark's serve settings, overridden by `serve.per_model.<model>` and, within that, by
    `per_harness.<harness>` (Gemma was served differently per harness on MedAgentBench)."""
    s = {k: v for k, v in b.cfg["serve"].items() if k != "per_model"}
    pm = dict((b.cfg["serve"].get("per_model") or {}).get(model) or {})
    ph = pm.pop("per_harness", None) or {}
    s.update(pm)
    if ph:
        if harness is None:
            raise SystemExit(f"{b.name}: {model} was served differently per harness ({', '.join(sorted(ph))} vs the rest); "
                             "pass --harness to serve it for one harness")
        s.update(ph.get(harness) or {})
    return s


def _base_template(model: str) -> str:
    """The base model's own chat template, written to a file vLLM can load (`chat_template: base`)."""
    src = Path(model_source(model))
    tc = src / "tokenizer_config.json"
    if not tc.exists():
        raise SystemExit(f"chat_template: base needs the base model on disk ({src}); set SHARED_MODEL_STORE")
    out = RUNS / "serve" / f"{model}-base-template.jinja"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.load(open(tc))["chat_template"])
    return str(out)


def serve_cmd(b: Benchmark, model: str, path: str, port: int, served: str = SERVED,
              tensor_parallel_size: int | None = None, harness: str | None = None) -> list[str]:
    s = serve_settings(b, model, harness)
    m = model_cfg(model)
    tp = tensor_parallel_size or (m.get("serve") or {}).get("tensor_parallel_size")
    cmd = ["uv", "run", "--project", str(ROOT / "third_party" / "SkyRL"), "--extra", "fsdp", "--extra", "harbor",
           "python", "-m", "vllm.entrypoints.openai.api_server",
           "--model", path, "--served-model-name", served, "--host", "0.0.0.0", "--port", str(port),
           "--gpu-memory-utilization", "0.9", "--max-model-len", str(s["max_model_len"])]
    if tp and int(tp) > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    if s.get("reasoning_parser"):
        cmd += ["--reasoning-parser", s["reasoning_parser"]]
    if s.get("tool_call_parser"):   # null: served plain, tool calls stay in the text
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", s["tool_call_parser"]]
    tmpl = s.get("chat_template") or (m.get("serve") or {}).get("chat_template")
    if tmpl == "base":
        cmd += ["--chat-template", _base_template(model)]
    elif tmpl:
        cmd += ["--chat-template", str(ROOT / tmpl)]
    return cmd


# --------------------------------------------------------------------------- SFT
def chat_template(b: Benchmark, model: str, pool: str) -> str | None:
    """The chat template an arm trained with: configs/models.yaml, overridden per model by the
    benchmark's sft.chat_template, then by the pool's chat_template (null = the model's own)."""
    tmpl = model_cfg(model).get("chat_template")
    for over in ((b.cfg.get("sft") or {}).get("chat_template") or {}, b.pool(pool).get("chat_template") or {}):
        if model in over:
            tmpl = over[model]
    return tmpl


def sft_cmd(b: Benchmark, model: str, pool: str, out: Path | None = None, wandb_run: str | None = None,
            data: str | None = None, max_steps: int | None = None) -> list[str]:
    p = b.pool(pool)
    if not p.get("file"):
        raise SystemExit(f"{b.name}/{pool}: nothing to train (base model)")
    m = model_cfg(model)
    sft = b.cfg["sft"]
    a = sft["args"]
    out = out or checkpoint_dir(b.name, model, pool)
    tmpl = chat_template(b, model, pool)
    cmd = ["python", "-m", "exp.sft_launch"]
    if not sft.get("cudnn_sdp", True):
        cmd.append("--disable-cudnn-sdp")
    cmd += ["--", "--data", str(data or (ROOT / p["file"])), "--model", model_source(model), "--out", str(out),
            "--epochs", str(a["epochs"]), "--max-len", str(a["max_len"])]
    if a.get("liger"):
        cmd.append("--liger")
    if a.get("optim"):
        cmd += ["--optim", a["optim"]]
    cmd += ["--chat-template", str(ROOT / tmpl) if tmpl else ""]
    if wandb_run:
        cmd += ["--wandb-run", wandb_run]
    if max_steps:  # small test run
        cmd += ["--max-steps", str(max_steps)]
    return cmd


# --------------------------------------------------------------------------- eval
def eval_job(b: Benchmark, harness: str, *, job_name: str, split: str = "test", backend: str = "docker",
             api_base: str = "http://127.0.0.1:8000/v1", sandbox_api_base: str | None = None,
             served: str = SERVED, jobs_dir: Path | None = None, concurrency: int | None = None,
             n_attempts: int | None = None, temperature: float | None = None, placeholders: bool = False,
             tasks: Path | None = None, job_extra: dict | None = None, limit: int | None = None) -> dict:
    spec = b.agent_spec(harness)
    sandbox = backends.sandbox_endpoint(backend, api_base, sandbox_api_base)
    if spec.get("runs_in") == "sandbox" and not sandbox:
        if placeholders:
            sandbox = "${SANDBOX_API_BASE}"
        else:
            raise SystemExit(f"{b.name}/{harness} runs inside the sandbox: pass --sandbox-api-base "
                             f"(a URL the {backend} sandboxes can reach the model server at)")
    ctx = {"harness": harness, "api_base": api_base,
           "sandbox_api_base": sandbox, "sandbox_host": backends.host_of(sandbox), "served_model": served}
    agent: dict = {}
    if spec.get("name"):
        agent["name"] = spec["name"]
    if spec.get("import_path"):
        agent["import_path"] = spec["import_path"]
    agent["model_name"] = f"{spec['provider']}/{served}"
    kwargs = fmt(spec.get("kwargs") or {}, ctx)
    if temperature is not None and "temperature" in kwargs:
        kwargs["temperature"] = temperature
    if kwargs:
        agent["kwargs"] = kwargs
    env = fmt(spec.get("env") or {}, ctx)
    if env:
        agent["env"] = env
    sp = b.split(split)
    ds: dict = {"path": str((tasks or tasks_root()) / b.name / split)}
    names = b.split_tasks(split)
    if limit:  # small test run: `limit` tasks spread evenly over the split (sorted names group task families)
        if names is None:
            split_dir = Path(ds["path"])
            names = sorted(p.name for p in split_dir.iterdir() if p.is_dir()) if split_dir.is_dir() else []
        names = names[::max(1, len(names) // limit)][:limit]
    if names:
        ds["task_names"] = names
    job = {
        "job_name": job_name,
        "jobs_dir": str(jobs_dir or (RUNS / "jobs" / b.name)),
        "n_attempts": int(n_attempts or b.eval["n_attempts"]),
        "n_concurrent_trials": int(concurrency or b.eval["concurrency"]),
        "environment": backends.environment(b, backend, placeholders=placeholders),
        "agents": [agent],
        "datasets": [ds],
    }
    job.update(spec.get("job") or {})
    job.update(job_extra or {})
    return job


def collect_job(b: Benchmark, entry: dict, *, backend: str = "docker", api_base: str = "http://127.0.0.1:8000/v1",
                sandbox_api_base: str | None = None, served: str = SERVED, jobs_dir: Path | None = None,
                placeholders: bool = False, tasks: Path | None = None, limit: int | None = None,
                n_attempts: int | None = None, concurrency: int | None = None) -> dict:
    """One trace-collection job: the eval job for (harness, split) with the collection protocol, named as the
    config's `job` (the pool builders in benchmarks/<b>/build/ find their inputs by that name).
    Used for teacher traces (`collect.jobs`) and rehearsal traces (`rehearsal`)."""
    hv = b.cfg["collect"]
    h = entry["harness"]
    split = entry.get("split", hv.get("split", "train"))
    job = eval_job(b, h, job_name=entry["job"], split=split, backend=backend, api_base=api_base,
                   sandbox_api_base=sandbox_api_base, served=served, jobs_dir=jobs_dir,
                   n_attempts=n_attempts or entry.get("n_attempts", hv.get("n_attempts")), placeholders=placeholders,
                   tasks=tasks, limit=limit, concurrency=concurrency)
    over = (hv.get("per_harness") or {}).get(h) or {}
    agent = job["agents"][0]
    kw = dict(agent.get("kwargs") or {})
    if kw or hv.get("kwargs"):
        for k, v in {**(hv.get("kwargs") or {}), **(over.get("kwargs") or {}), **(entry.get("kwargs") or {})}.items():
            if k == "temperature" and "temperature" not in kw:
                continue  # agents that take sampling from their own config (e.g. mini-swe) stay that way
            kw[k] = v
        for k in over.get("drop_kwargs") or []:
            kw.pop(k, None)
        if kw:
            agent["kwargs"] = kw
    for k in over.get("job_drop") or []:
        job.pop(k, None)
    return job


def teacher_protocol(b: Benchmark) -> dict:
    return dict(((b.cfg.get("teacher") or {}).get("protocol")) or {})


def job_name(b: Benchmark, model: str, setting: str, harness: str, split: str = "test") -> str:
    tier = b.tier(split)
    parts = [b.name, model, setting] + ([tier] if tier else []) + [harness]
    return "-".join(parts).replace("+", "_")


def dump_job(job: dict) -> str:
    return yaml.safe_dump(job, sort_keys=False, width=1000)


# --------------------------------------------------------------------------- GRPO
def grpo(b: Benchmark, model: str, rollout: str, *, init: str | None = None, backend: str = "docker",
         run_name: str | None = None, placeholders: bool = False, max_steps: int | None = None,
         out: str | None = None, batch_size: int | None = None) -> tuple[list[str], dict, Path]:
    """Return (argv, env, cwd) for one GRPO run. rollout = '3h' or one harness name."""
    rl = b.rl
    if not rl:
        raise SystemExit(f"{b.name}: RL is reported for medagentbench and reasoning_gym only")
    recipe = yaml.safe_load(open(ROOT / "configs" / "rl.yaml"))
    harnesses = b.harnesses("trained") if rollout == "3h" else [rollout]
    if rollout != "3h" and rollout not in b.harnesses("trained"):
        raise SystemExit(f"{b.name}: rollout harness must be 3h or one of {b.harnesses('trained')}")
    m = model_cfg(model)
    out = Path(out).resolve() if out else rl_dir(b.name, model, rollout)
    init = init or str(checkpoint_dir(b.name, model, rl["init_pool"]) / "final")
    init = str(Path(init).resolve())   # the trainer runs from third_party/SkyRL; a relative path would not resolve
    if not placeholders and not Path(init).is_dir():
        raise SystemExit(f"GRPO init checkpoint not found: {init}")
    train = str(tasks_root() / b.name / "train")
    rm = rl["rollouts_matched"]
    ov: dict[str, str] = {
        "data.train_data": f"['{train}']",
        "trainer.policy.model.path": init,
        "generator.inference_engine.served_model_name": m["hf"].split("/")[-1],
        "harbor_trial_config.trials_dir": str(out / "trials"),
        "trainer.export_path": str(out / "exports"),
        "trainer.ckpt_path": str(out / "ckpts"),
        "trainer.log_path": str(out / "logs"),
        "generator.inference_engine.engine_init_kwargs.chat_template": str(ROOT / m["chat_template"]),
        "trainer.epochs": str(rm["multi_harness_epochs"] if len(harnesses) > 1 else rm["single_harness_epochs"]),
        "trainer.run_name": run_name or f"{b.name}-{model}-grpo-{rollout}",
        "harbor_trial_config.agent.import_path": rl["agent_import_path"],
        "xh.dataset_root": train,
        "xh.split_file": str(ROOT / rl["band"]),
        "xh.harnesses": "[" + ",".join(f"'{h}'" for h in harnesses) + "]",
    }
    ov.update({k: str(v) for k, v in recipe["overrides"].items()})
    ov.update({k: str(v) for k, v in (rl.get("extra_overrides") or {}).items()})
    if ov.get("trainer.logger") == "wandb" and not os.environ.get("WANDB_API_KEY"):
        print("note: WANDB_API_KEY not set; logging GRPO metrics to the console instead of wandb")
        ov["trainer.logger"] = "console"
    if max_steps:  # small test run: stop after max_steps updates and save then
        ov.update({"trainer.max_training_steps": str(max_steps), "trainer.ckpt_interval": str(max_steps),
                   "trainer.hf_save_interval": str(max_steps)})
    if batch_size:  # small test run: fewer prompts per update (the recipe's batch is 24)
        # Each task is rolled out once per harness, so the batch must split evenly between them
        if batch_size % len(harnesses):
            raise SystemExit(f"--batch-size {batch_size}: must be a multiple of the {len(harnesses)} rollout harnesses")
        ov.update({"trainer.train_batch_size": str(batch_size), "trainer.policy_mini_batch_size": str(batch_size)})
    for h, kw in (rl.get("harness_kwargs") or {}).items():
        if h not in harnesses:      # kwargs for harnesses that are not rolled out are inert; pass only the used ones
            continue
        for k, v in _flatten(kw).items():
            ov[f"xh.harness_kwargs.{h}.{k}"] = str(v)
    env_block = backends.environment(b, backend, placeholders=placeholders)
    ov["harbor_trial_config.environment.type"] = env_block["type"]
    for k, v in (env_block.get("kwargs") or {}).items():
        if k != "image":
            ov[f"harbor_trial_config.environment.kwargs.{k}"] = str(v)
    argv = ["uv", "run", "--isolated"] + sum((["--extra", e] for e in recipe["uv_extras"]), []) + \
           ["-m", recipe["entrypoint"]] + [f"{k}={v}" for k, v in ov.items()]
    env = {"PYTHONPATH": str(ROOT), **recipe.get("env", {})}
    return argv, env, ROOT / "third_party" / "SkyRL"


def _flatten(d: dict, p: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{p}.{k}" if p else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
