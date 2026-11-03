"""`exp table`: rebuild experiment results.

By default, reads trial rewards and task IDs from
`runs/jobs/<benchmark>/<job>/` (another folder with `--jobs-dir`) and applies the
benchmark's configured aggregation and bootstrap procedure. AppWorld bootstraps
over scenarios.

With `--shipped`, reads cell values from `results/cells.csv`; per-trial outcomes
are not included in the public release. `--format cells` writes results in the
same schema as `results/cells.csv`, so a re-run can be compared with them.
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from exp import config as C, render as R

ROOT = C.ROOT


def stats(per: dict, b, cluster_unit: str | None = None, estimator: str = "task_mean") -> dict | None:
    """per: task -> [reward]. Mirrors the as-run implementations exactly (seed, ordering, index).

    estimator: task_mean    = resample tasks, statistic = mean of per-task means
                              (the as-run per-task estimator)
               trial_pooled = resample tasks, statistic = pooled trial mean of the resample
                              (the 3H-SFT and 3H-GRPO rows of grpo-3h-vs-single)
    """
    if not per:
        return None
    bs = b.eval.get("bootstrap") or {"B": 1000, "seed": 20260923, "hi_index": 975, "binarize": True}
    if bs.get("binarize"):
        per = {t: [1 if float(r) in (1.0,) else 0 for r in v] for t, v in per.items()}
    tasks = sorted(per)
    trials = sum(len(per[t]) for t in tasks)
    pooled = 100 * sum(sum(per[t]) for t in tasks) / trials
    rates = {t: sum(per[t]) / len(per[t]) for t in tasks}
    task_level = 100 * sum(rates.values()) / len(rates)
    if (cluster_unit or b.eval.get("bootstrap_unit")) == "scenario":
        groups = defaultdict(list)
        for t, r in rates.items():
            groups[t.split("_")[0]].append(r)
        units = [sum(v) / len(v) for _, v in sorted(groups.items())]
    else:
        units = [rates[t] for t in tasks]
    rng = random.Random(bs["seed"])
    n, out = len(units), []
    if estimator == "trial_pooled":
        data = [per[t] for t in tasks]
        for _ in range(bs["B"]):
            idx = [rng.randrange(n) for _ in range(n)]
            out.append(100 * sum(sum(data[i]) for i in idx) / sum(len(data[i]) for i in idx))
    else:
        for _ in range(bs["B"]):
            out.append(100 * sum(units[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    lo, hi = out[int(0.025 * bs["B"])], out[bs["hi_index"]]
    mean = pooled if b.eval.get("aggregate") == "pooled" else task_level
    return {"mean": mean, "pooled": pooled, "task_level": task_level, "lo": lo, "hi": hi,
            "pm": (hi - lo) / 2, "tasks": len(tasks), "trials": trials}


def read_job(job_dir: Path) -> dict:
    per = defaultdict(list)
    if not job_dir.is_dir():
        return {}
    for t in sorted(job_dir.iterdir()):
        rp = t / "verifier" / "reward.txt"
        if not rp.is_file():
            continue
        try:
            r = float(rp.read_text().strip())
        except ValueError:
            continue
        task = t.name.split("__")[0]
        cfg = t / "config.json"
        if cfg.is_file():
            try:
                task = os.path.basename(json.load(open(cfg))["task"]["path"].rstrip("/"))
            except Exception:
                pass
        per[task].append(r)
    return dict(per)


_CELLS = None


def _cells():
    """results/cells.csv (reported rows) indexed by (benchmark, model, tier, layer, arm, harness, metric)."""
    global _CELLS
    if _CELLS is None:
        _CELLS = {}
        path = ROOT / "results" / "cells.csv"
        if path.is_file():
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    if r.get("status", "reported") != "reported":
                        continue
                    k = (r["benchmark"], r["model"], r["tier"], r["layer"], r["arm"], r["harness"], r["metric"])
                    assert k not in _CELLS, f"results/cells.csv: duplicate reported cell {k}"
                    _CELLS[k] = r
    return _CELLS


def _shipped_cell(b, model: str, setting: str, harness: str, split: str, metric: str = "mean"):
    """A published cell, from the unified results/cells.csv."""
    arm = _arm_label(b, setting)
    r = _cells().get((b.name, model, b.tier(split) or "", _layer(b, setting, harness), arm, harness, metric))
    if not r:
        return None
    f = lambda k: float(r[k]) if r[k] not in ("", None) else 0.0
    return {"mean": f("value"), "pooled": f("value"), "task_level": f("value"),
            "lo": f("ci_lo"), "hi": f("ci_hi"), "pm": f("plusminus"),
            "tasks": int(r["tasks"] or 0), "trials": int(r["trials"] or 0)}


def _per(b, model: str, setting: str, harness: str, split: str, shipped: bool, jobs_dir: Path):
    """task -> [reward] for one cell, plus the job ids it came from.

    Returns (None, []) with --shipped: per-trial outcomes are not part of the
    public release, so published values are read from results/cells.csv instead.
    """
    if shipped:
        return None, []
    name = R.job_name(b, model, _arm_label(b, setting), harness, split)
    return read_job(jobs_dir / b.name / name), [name]


def pass_at(per: dict, b, thresh: float = 0.99) -> dict | None:
    """pass@n over the benchmark's resampling unit, same bootstrap constants as stats()."""
    if not per:
        return None
    bs = b.eval.get("bootstrap") or {"B": 1000, "seed": 20260923, "hi_index": 974}
    tasks = sorted(per)
    solved = {t: 1 if max(per[t]) >= thresh else 0 for t in tasks}
    if b.eval.get("bootstrap_unit") == "scenario":
        groups = defaultdict(list)
        for t, v in solved.items():
            groups[t.split("_")[0]].append(v)
        units = [sum(v) / len(v) for _, v in sorted(groups.items())]
    else:
        units = [solved[t] for t in tasks]
    n = len(units)
    point = 100 * sum(units) / n
    rng = random.Random(bs["seed"])
    out = [100 * sum(units[rng.randrange(n)] for _ in range(n)) / n for _ in range(bs["B"])]
    out.sort()
    lo, hi = out[int(0.025 * bs["B"])], out[bs["hi_index"]]
    return {"value": point, "lo": lo, "hi": hi, "pm": (hi - lo) / 2,
            "tasks": len(tasks), "trials": sum(len(v) for v in per.values()), "units": n}


def cell(b, model: str, setting: str, harness: str, split: str, shipped: bool, jobs_dir: Path, estimator: str = "task_mean"):
    per, _ = _per(b, model, setting, harness, split, shipped, jobs_dir)
    if per is None:
        return _shipped_cell(b, model, setting, harness, split)
    return stats(per, b, estimator=estimator)


CELL_COLS = ["benchmark", "model", "tier", "layer", "arm", "harness", "metric", "value",
             "ci_lo", "ci_hi", "plusminus", "tasks", "trials", "bootstrap_unit",
             "bootstrap_n", "status", "note"]


def _layer(b, setting: str, harness: str) -> str:
    """The sampling protocol of a cell: the teacher's own protocol, else the benchmark's per-harness layer."""
    if setting == "teacher":
        return R.teacher_protocol(b).get("layer") or b.layer(harness)
    return b.layer(harness)


def _settings(b, s) -> list[str]:
    """Rows of one experiment setting, in order, without repeats (rl rows bring their init pool)."""
    st = (["teacher"] if s.get("teacher") else []) + ([b.rl["init_pool"]] if s.get("rl") else []) + \
         list(s.get("pools", [])) + [f"rl:{r}" for r in s.get("rl", [])]
    return list(dict.fromkeys(st))


def _arm_label(b, setting: str) -> str:
    if setting == "teacher":
        return "teacher"
    if setting.startswith("rl:"):
        return f"grpo-{setting[3:]}"
    return setting


def cells(a) -> int:
    """Emit this experiment's cells in the unified schema of results/cells.csv."""
    e = C.experiments().get(a.experiment)
    if not e:
        raise SystemExit(f"unknown experiment {a.experiment!r}; see `exp list experiments`")
    jobs_dir = Path(a.jobs_dir) if a.jobs_dir else R.RUNS / "jobs"
    rows = []
    for s in e["settings"]:
        b = C.benchmark(s["benchmark"])
        split = s.get("split", "test")
        tier = b.tier(split) or ""
        unit = b.eval.get("bootstrap_unit") or "task"
        for st in _settings(b, s):
            for h in b.harnesses(s.get("harnesses", "trained")):
                per, jobs = _per(b, s["model"], st, h, split, a.shipped, jobs_dir)
                layer = _layer(b, st, h)
                base = dict(benchmark=b.name, model=s["model"], tier=tier, layer=layer,
                            arm=_arm_label(b, st), harness=h, bootstrap_unit=unit,
                            status="reported")
                src = ("shipped" if a.shipped else "runs") + (": " + ",".join(jobs) if jobs else "")
                if per is None:            # no per-trial data shipped: echo results/cells.csv
                    for metric in ("mean", f"pass{b.eval.get('n_attempts', 1)}"):
                        c = _shipped_cell(b, s["model"], st, h, split, metric)
                        if not c:
                            continue
                        rows.append({**base, "metric": metric, "value": f"{c['mean']:.2f}",
                                     "ci_lo": f"{c['lo']:.2f}", "ci_hi": f"{c['hi']:.2f}",
                                     "plusminus": f"{c['pm']:.2f}", "tasks": c["tasks"],
                                     "trials": c["trials"] or "", "bootstrap_n": "",
                                     "note": "published value from results/cells.csv"})
                    continue
                if not per:
                    continue
                m = stats(per, b, estimator=(s.get("ci") or {}).get(st, "task_mean"))
                rows.append({**base, "metric": "mean", "value": f"{m['mean']:.2f}",
                             "ci_lo": f"{m['lo']:.2f}", "ci_hi": f"{m['hi']:.2f}",
                             "plusminus": f"{m['pm']:.2f}", "tasks": m["tasks"],
                             "trials": m["trials"], "bootstrap_n": m["tasks"], "note": src})
                if b.eval.get("n_attempts", 1) > 1:
                    pn = pass_at(per, b)
                    rows.append({**base, "metric": f"pass{b.eval['n_attempts']}",
                                 "value": f"{pn['value']:.2f}", "ci_lo": f"{pn['lo']:.2f}",
                                 "ci_hi": f"{pn['hi']:.2f}", "plusminus": f"{pn['pm']:.2f}",
                                 "tasks": pn["tasks"], "trials": pn["trials"],
                                 "bootstrap_n": pn["units"], "note": src})
    out = open(a.out, "w", newline="") if a.out else __import__("sys").stdout
    w = csv.DictWriter(out, fieldnames=CELL_COLS)
    w.writeheader()
    w.writerows(rows)
    if a.out:
        out.close()
        print(f"wrote {a.out}: {len(rows)} rows")
    return 0


def main(a) -> int:
    if getattr(a, "format", "table") == "cells":
        return cells(a)
    e = C.experiments().get(a.experiment)
    if not e:
        raise SystemExit(f"unknown experiment {a.experiment!r}; see `exp list experiments`")
    jobs_dir = Path(a.jobs_dir) if a.jobs_dir else R.RUNS / "jobs"
    print(f"# {e['title']}   ({'shipped results' if a.shipped else 'runs: ' + C.rel(jobs_dir)})\n")
    empties = False
    for s in e["settings"]:
        b = C.benchmark(s["benchmark"])
        split = s.get("split", "test")
        hs = b.harnesses(s.get("harnesses", "trained"))
        settings = _settings(b, s)
        tier = f" ({b.tier(split)})" if b.tier(split) else ""
        n_att = (R.teacher_protocol(b).get("n_attempts") if s.get("teacher") else None) or b.eval["n_attempts"]
        print(f"## {b.cfg['display']}{tier}   model={s['model']}   attempts per task={n_att}   mean ± 95% CI half-width")
        print("| setting | " + " | ".join(hs) + " |")
        print("|---|" + "---|" * len(hs))
        for st in settings:
            row = []
            for h in hs:
                c = cell(b, s["model"], st, h, split, a.shipped, jobs_dir, (s.get("ci") or {}).get(st, "task_mean"))
                row.append("—" if not c else f"{c['mean']:.1f} ± {c['pm']:.1f}")
            label = {"teacher": "Qwen3-32B teacher"}.get(st, st)
            if st.startswith("rl:"):
                label = f"GRPO, rollouts on {st[3:]}"
            elif st in b.pools():
                label = b.pool(st).get("paper", st)
            print(f"| {label} | " + " | ".join(row) + " |")
            if "—" in row:
                empties = True
        print()
    if empties:
        print("— : no result: " + ("the cell has not been run under runs/ yet." if not a.shipped else
              "no reported row in results/cells.csv for this cell (the paper does not report it)."))
    return 0
