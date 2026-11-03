"""`exp tasks`: generate Harbor task directories for a benchmark split.

Runs the benchmark's existing task generator, stages its inputs, copies the
selected tasks to `runs/tasks/<benchmark>/<split>/`, and writes `.hc-tasks.json`
with the generation details.

BFCL multi-turn uses the shipped frozen data. Reasoning Gym uses the
harbor-datasets registry pinned by the vendored Harbor version (`--source adapter`
regenerates the tasks instead, with different task names). MedAgentBench and BFCL
single-turn run this repo's adapter on a clone of the upstream benchmark repository
by default; the commit the paper used was not recorded, and `--source-commit` pins
one. `--source registry` uses the pinned harbor-datasets version instead. `--from`
copies an existing generated split. Rehearsal splits are generated from the
training tasks by the benchmark's `build/gen_rehearsal.py`.

AppWorld uses its installed licensed data package.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from exp import config as C
from exp.render import RUNS, tasks_root

ROOT = C.ROOT

# harbor-datasets commits pinned by the vendored harbor framework's registry (<owner>/harbor @ 9233373f,
# registry.json). Reasoning Gym's as-run tasks are these (cluster 2 symlinked the registry; task names match).
HARBOR_DATASETS = "https://github.com/laude-institute/harbor-datasets.git"
REGISTRY = {
    "reasoning_gym": {"commit": "9119ca8a5a26868a12e42058321cc10daa74503d", "paths": {"easy": "datasets/reasoning-gym/easy", "hard": "datasets/reasoning-gym/hard"}},
    "medagentbench": {"commit": "9a7e5aea093bfdeeccc4ea319140f8ce351d2b63", "paths": {"all": "datasets/medagentbench"}},
    "bfcl_singleturn": {"commit": "6bedd7878dc5d6f3456b4d80b781eb3c2d84f262", "paths": {"all": "datasets/bfcl"}},
}


def _registry_fetch(bench: str, keys: list[str], dry: bool) -> dict[str, Path]:
    """Sparse-checkout the pinned harbor-datasets paths; returns key -> local dir."""
    pin = REGISTRY[bench]
    dest = Path(_staging_dir("hc-hd-")) / "harbor-datasets"
    paths = [pin["paths"][k] for k in keys]
    _sh(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse", HARBOR_DATASETS, dest], dry=dry)
    _sh(["git", "sparse-checkout", "set", "--no-cone", *paths], cwd=dest, dry=dry)
    _sh(["git", "checkout", "-q", pin["commit"]], cwd=dest, dry=dry)
    return {k: dest / pin["paths"][k] for k in keys}


def _sh(argv, cwd=None, env=None, dry=False):
    print("+ " + " ".join(str(a) for a in argv) + (f"   (in {C.rel(cwd)})" if cwd else ""), flush=True)
    if dry:
        return
    subprocess.run([str(a) for a in argv], cwd=cwd, env={**os.environ, **(env or {})}, check=True)


def _staging_dir(prefix: str) -> str:
    """A fresh folder under runs/staging/ for clones and generator output before the tasks are
    copied to runs/tasks/ (safe to delete afterwards)."""
    root = RUNS / "staging"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=root)


def _git_clone(url: str, dest: Path, commit: str | None, dry: bool) -> str | None:
    if commit:
        _sh(["git", "clone", "--filter=blob:none", url, dest], dry=dry)
        _sh(["git", "checkout", "-q", commit], cwd=dest, dry=dry)
    else:
        _sh(["git", "clone", "--depth", "1", url, dest], dry=dry)
    if dry:
        return commit
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()


def _select(src: Path, out: Path, names: list[str] | None, pred=None, dry=False) -> int:
    if dry:
        print(f"+ copy {'all' if names is None and pred is None else 'selected'} task dirs from {C.rel(src)} -> {C.rel(out)}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    have = {p.name: p for p in src.iterdir() if p.is_dir()}
    want = names if names is not None else [n for n in sorted(have) if (pred is None or pred(n))]
    missing = [n for n in want if n not in have]
    if missing:
        raise SystemExit(f"generator did not produce {len(missing)} expected tasks, e.g. {missing[:3]}")
    for n in want:
        dst = out / n
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(have[n], dst, symlinks=False)
    return len(want)


def _manifest(out: Path, info: dict, dry: bool):
    if dry:
        return
    (out / ".hc-tasks.json").write_text(json.dumps(info, indent=2) + "\n")


# --------------------------------------------------------------------------- per benchmark
def bfcl_multiturn(b, split, out, a):
    t = ROOT / "benchmarks" / "bfcl_multiturn" / "tasks"
    stage = Path(_staging_dir("hc-bfclmt-"))
    for name in ["multi_turn_func_doc", "possible_answer", "bfcl_env", "mt_split.json",
                 *[p.name for p in t.glob("BFCL_v4_multi_turn_*.json")]]:
        (stage / name).symlink_to(t / name) if not a.dry_run else None
    if not a.dry_run:
        (stage / "eval").mkdir()
        (stage / "eval" / "evaluate_template.py").symlink_to(t / "evaluate_template.py")
        (stage / "mt_split.json").unlink()
        shutil.copy(t / "mt_split.json", stage / "mt_split.json")  # the scripts rewrite split files in B
    for script in ["gen_mt_tasks.py", "gen_mt_tasks_ext.py"]:
        src = (t / script).read_text()
        line = 'B = "${PROJECT_ROOT}/<user>_bfcl_runs/mt"'
        if line not in src:
            raise SystemExit(f"{script}: cluster root line not found; generator changed?")
        print(f"+ python {C.rel(t / script)}  (cluster root B -> staged copy of benchmarks/bfcl_multiturn/tasks)")
        if not a.dry_run:
            code = src.replace(line, f"B = {str(stage)!r}")
            subprocess.run([sys.executable, "-c", code], check=True, cwd=stage)
    n = _select(stage / "tasks", out, b.split_tasks(split), dry=a.dry_run)
    _manifest(out, {"benchmark": b.name, "split": split, "tasks": n, "generator": "benchmarks/bfcl_multiturn/tasks/gen_mt_tasks{,_ext}.py",
                    "data": "benchmarks/bfcl_multiturn/tasks (shipped, frozen)"}, a.dry_run)


def reasoning_gym(b, split, out, a):
    tiers = {"test": ["easy"], "test-hard": ["hard"], "train": ["easy", "hard"], "train-easy": ["easy"], "train-hard": ["hard"]}[split]
    keep = {"2"} if split.startswith("test") else {"0", "1"}  # seeds 0-1 train, seed 2 test
    if a.source == "adapter":
        print("warning: regenerating with benchmarks/reasoning_gym/tasks gives different task names (config hash) from the"
              " as-run registry tasks; use the default --source registry to reproduce the paper")
        proj = ROOT / "benchmarks" / "reasoning_gym" / "tasks"
        stage = Path(_staging_dir("hc-rg-"))
        dirs = {}
        for tier in tiers:
            argv = ["uv", "run", "--project", proj, "reasoning-gym", "--config", proj / f"{tier}.yaml", "--size", "3",
                    "--output-dir", stage / tier] + (["--difficulty", "hard"] if tier == "hard" else [])
            _sh(argv, cwd=proj, dry=a.dry_run)
            dirs[tier] = stage / tier
        source = {"generator": "benchmarks/reasoning_gym/tasks (reasoning-gym CLI), NOT the as-run source"}
    else:
        dirs = _registry_fetch("reasoning_gym", tiers, a.dry_run)
        source = {"source": "harbor-datasets registry (as run)", "repo": HARBOR_DATASETS,
                  "commit": REGISTRY["reasoning_gym"]["commit"], "paths": [REGISTRY["reasoning_gym"]["paths"][t] for t in tiers]}
    n = 0
    for tier in tiers:
        n += _select(dirs[tier], out, None, pred=lambda nm: nm.rsplit("-", 1)[-1] in keep, dry=a.dry_run)
    image = os.environ.get("RG_TASK_IMAGE")
    if image and not a.dry_run:
        _set_docker_image(out, image)
        source["docker_image"] = image
    want = b.split_tasks(split)
    if want is not None and not a.dry_run:
        got = sorted(p.name for p in out.iterdir() if p.is_dir())
        if set(got) != set(want):
            raise SystemExit(f"{split} task names do not match {b.split(split).get('task_list')}: "
                             f"{len(set(want) - set(got))} missing, {len(set(got) - set(want))} extra")
    _manifest(out, {"benchmark": b.name, "split": split, "tasks": n, **source}, a.dry_run)


def _set_docker_image(out: Path, image: str):
    """Name a prebuilt image in every task.toml, where the as-run Reasoning Gym tasks had it (for
    backends with no build step, e.g. eidf). Built from the tasks' environment/Dockerfile by
    benchmarks/reasoning_gym/tasks/build_task_image.sh."""
    mark = "# Leave empty to build from Dockerfile in environment/ directory\n"
    for t in sorted(p for p in out.iterdir() if (p / "task.toml").exists()):
        s = (t / "task.toml").read_text()
        if "\ndocker_image = " in s:
            continue
        if mark not in s:
            raise SystemExit(f"{t.name}/task.toml: no place for docker_image")
        (t / "task.toml").write_text(s.replace(mark, mark + f'docker_image = "{image}"\n', 1))


def _from_registry(bench, b, split, out, a, note):
    d = _registry_fetch(bench, ["all"], a.dry_run)["all"]
    names = b.split_tasks(split)
    if names is None:  # medagentbench train: everything that is not test
        test = set(b.split_tasks("test"))
        n = _select(d, out, None, pred=lambda nm: nm not in test, dry=a.dry_run)
    else:
        n = _select(d, out, names, dry=a.dry_run)
    _manifest(out, {"benchmark": b.name, "split": split, "tasks": n, "source": "harbor-datasets registry",
                    "repo": HARBOR_DATASETS, "commit": REGISTRY[bench]["commit"], "note": note}, a.dry_run)


def medagentbench(b, split, out, a):
    if a.source == "registry":
        return _from_registry("medagentbench", b, split, out, a,
                              "pinned upstream generation; the clusters generated with benchmarks/medagentbench/tasks (default)")
    proj = ROOT / "benchmarks" / "medagentbench" / "tasks"
    stage = Path(_staging_dir("hc-mab-"))
    commit = _git_clone("https://github.com/stanfordmlgroup/MedAgentBench", stage / "MedAgentBench", a.source_commit, a.dry_run)
    test = b.split_tasks("test")
    argv = [sys.executable, proj / "run_adapter.py", "--medagentbench-repo", stage / "MedAgentBench", "--output-dir", stage / "tasks"]
    if split == "test":
        argv += ["--ids", *test]
    _sh(argv, cwd=proj, dry=a.dry_run)
    if split == "test":
        n = _select(stage / "tasks", out, test, dry=a.dry_run)
    else:
        n = _select(stage / "tasks", out, None, pred=lambda nm: nm not in set(test), dry=a.dry_run)
        print("note: the as-run training split had 209 of these 215 non-test tasks (6 removed by a filter whose"
              " list was not shipped); RL uses the shipped band (96 ids) and is unaffected.")
    _manifest(out, {"benchmark": b.name, "split": split, "tasks": n, "generator": "benchmarks/medagentbench/tasks/run_adapter.py",
                    "upstream": "stanfordmlgroup/MedAgentBench", "upstream_commit": commit,
                    "upstream_commit_note": "as-run commit not recorded"}, a.dry_run)


def bfcl_singleturn(b, split, out, a):
    if a.source == "registry":
        return _from_registry("bfcl_singleturn", b, split, out, a,
                              "pinned upstream generation; cluster 1 generated with benchmarks/bfcl_singleturn/tasks (default)")
    proj = ROOT / "benchmarks" / "bfcl_singleturn" / "tasks"
    stage = Path(_staging_dir("hc-bfcl-"))
    commit = _git_clone("https://github.com/ShishirPatil/gorilla.git", stage / "gorilla", a.source_commit, a.dry_run)
    _sh(["uv", "run", "--project", proj, "bfcl", "--bfcl-root", stage / "gorilla" / "berkeley-function-call-leaderboard",
         "--output-dir", stage / "tasks"], cwd=proj, dry=a.dry_run)
    n = _select(stage / "tasks", out, b.split_tasks(split), dry=a.dry_run)
    _manifest(out, {"benchmark": b.name, "split": split, "tasks": n, "generator": "benchmarks/bfcl_singleturn/tasks (bfcl CLI)",
                    "upstream": "ShishirPatil/gorilla", "upstream_commit": commit,
                    "upstream_commit_note": "as-run commit not recorded"}, a.dry_run)


def appworld(b, split, out, a):
    proj = ROOT / "benchmarks" / "appworld" / "tasks"
    stage = Path(_staging_dir("hc-aw-"))
    aw_split = {"test": "test_normal", "train": "train"}[split]
    print("note: needs the AppWorld package and its data (`appworld install && appworld download data`)")
    # Every task names its image; build it with benchmarks/appworld/tasks/build_appworld_image.sh
    image = os.environ.get("APPWORLD_IMAGE")
    if not image:
        if not a.dry_run:
            raise SystemExit("set APPWORLD_IMAGE (in .env) to the task image: build and push it with "
                             "benchmarks/appworld/tasks/build_appworld_image.sh <image>")
        image = "${APPWORLD_IMAGE}"
    _sh(["uv", "run", "--project", proj, "appworld-bench", "--output-dir", stage, "--splits", aw_split,
         "--docker-image", image], cwd=proj, dry=a.dry_run)
    n = _select(stage, out, b.split_tasks(split), dry=a.dry_run)
    _manifest(out, {"benchmark": b.name, "split": split, "tasks": n, "generator": "benchmarks/appworld/tasks (appworld-bench CLI)",
                    "appworld_split": aw_split, "docker_image": image}, a.dry_run)


HANDLERS = {"bfcl_multiturn": bfcl_multiturn, "reasoning_gym": reasoning_gym, "medagentbench": medagentbench,
            "bfcl_singleturn": bfcl_singleturn, "appworld": appworld}


def main(a) -> int:
    b = C.benchmark(a.benchmark)
    b.split(a.split)
    out = Path(a.out) if a.out else tasks_root() / b.name / a.split
    if getattr(a, "from_dir", None):
        n = _select(Path(a.from_dir), out, b.split_tasks(a.split), dry=a.dry_run)
        _manifest(out, {"benchmark": b.name, "split": a.split, "tasks": n, "from": str(a.from_dir)}, a.dry_run)
        print(f"{n} tasks -> {C.rel(out)}")
        return 0
    gen = b.split(a.split).get("generate")
    if gen:  # rehearsal tasks: built from the training tasks by the benchmark's own generator
        # The generators refuse to write into an existing folder. Like every other split,
        # `exp tasks` regenerates, so remove the old copy first.
        if out.exists() and not a.dry_run:
            shutil.rmtree(out)
        argv = [sys.executable, str(ROOT / "benchmarks" / b.name / gen[0]), *gen[1:], "--out", str(out)]
        _sh(argv, dry=a.dry_run)
        return 0
    HANDLERS[b.name](b, a.split, out, a)
    if not a.dry_run:
        print(f"{sum(1 for p in out.iterdir() if p.is_dir())} tasks -> {C.rel(out)}")
    return 0
