"""Exchange bank: cross-harness success/stats store for multi-harness training.

One JSON file, single writer (the training driver), same persistence pattern
as PrimerStore. Keyed by (task, harness). Two consumers:

  * pooled-baseline statistics  -> ``stats(task)`` (recent rewards per harness;
    a slightly stale baseline is still a valid baseline);
  * demonstration selection     -> ``best_demo(task, target)``: the
    cost-minimal verified success FEASIBLE under the target harness
    (action types + budget only, per the pinned method), with at least one
    environment action (escape-hatch guard).

Design doc: docs/eidf/multi-harness-rollout-options.md (harbor repo).
"""

from __future__ import annotations

import json
from pathlib import Path

from .canonical import feasible_under

# Rolling window per (task, harness) for reward stats.
STATS_WINDOW = 64


class ExchangeBank:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.state: dict = {"cells": {}}
        if self.path.is_file():
            self.state = json.loads(self.path.read_text())

    # ------------------------------------------------------------------ util
    def _cell(self, task: str, harness: str) -> dict:
        return self.state["cells"].setdefault(f"{task}|{harness}", {
            "rewards": [],          # rolling window, newest last
            "successes": [],        # [{cost, actions, trial, step}]
        })

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)

    # ---------------------------------------------------------------- record
    def record(
        self,
        task: str,
        harness: str,
        *,
        reward: float,
        canonical_actions: list | None = None,
        cost: int | None = None,
        trial_ref: str = "",
        step: int = -1,
        save: bool = True,
    ) -> None:
        cell = self._cell(task, harness)
        cell["rewards"] = (cell["rewards"] + [float(reward)])[-STATS_WINDOW:]
        if reward > 0 and canonical_actions:
            env_actions = [a for a in canonical_actions if a[0] != "SUBMIT"]
            if len(env_actions) >= 1:  # escape-hatch guard: must touch the env
                cell["successes"].append({
                    "cost": cost if cost is not None else len(env_actions),
                    "actions": canonical_actions,
                    "trial": trial_ref,
                    "step": step,
                })
                # keep successes sorted by cost; cap storage
                cell["successes"].sort(key=lambda s: s["cost"])
                del cell["successes"][32:]
        if save:
            self._save()

    # ----------------------------------------------------------------- reads
    def stats(self, task: str) -> dict[str, dict]:
        """Per-harness reward stats for a task: {harness: {n, mean}}."""
        out = {}
        for key, cell in self.state["cells"].items():
            t, h = key.rsplit("|", 1)
            if t != task or not cell["rewards"]:
                continue
            r = cell["rewards"]
            out[h] = {"n": len(r), "mean": sum(r) / len(r)}
        return out

    def best_demo(self, task: str, target_harness: str) -> dict | None:
        """Cost-minimal verified success on `task` feasible under
        `target_harness`, across ALL source harnesses. None if the pool is
        empty. Source-agnostic by design: which side of the ladder supplies
        demos is an emergent, logged quantity (flow-reversal metric)."""
        best = None
        for key, cell in self.state["cells"].items():
            t, source = key.rsplit("|", 1)
            if t != task:
                continue
            for s in cell["successes"]:
                if not feasible_under(s["actions"], target_harness):
                    continue
                if best is None or s["cost"] < best["cost"]:
                    best = {**s, "source_harness": source}
        return best
