"""Semantic-task grouping: which realizations exist, and how rollouts are keyed.

A *semantic task* is the task identity shared by all its realizations
(`task4_17`), independent of which arm directory it was materialized into
(`medagentbench--answer_mode_mcq/task4_17`). A *realization* is one arm.

Two facts from the perturbation program drive this module:

1. **Arms are not universally applicable.** `missing_value_rule_*` exists for 90
   of 300 MedAgentBench tasks, `timestamp_granularity_stated` for 30, the
   write-payload arms for 150. Sampling a realization a task does not have would
   silently fall back to a copy of native — an R4 no-op that dilutes every
   measurement. Each arm therefore publishes `AFFECTED.json`, and we sample only
   from a task's *eligible* pool.
2. **The uid carries both group keys.** SkyRL hands the advantage estimator a
   single `index` array, so we encode both levels in it:

       uid = "<semantic_task_id>|<realization_arm>"

   within-realization group  = the full uid
   cross-realization group   = uid.split("|", 1)[0]

   No SkyRL plumbing change is needed for the grouping itself.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

UID_SEP = "|"


def make_uid(semantic_id: str, arm: str) -> str:
    if UID_SEP in semantic_id or UID_SEP in arm:
        raise ValueError(f"'{UID_SEP}' is reserved as the uid separator: {semantic_id} {arm}")
    return f"{semantic_id}{UID_SEP}{arm}"


def semantic_of(uid: str) -> str:
    """Cross-realization group key."""
    return uid.split(UID_SEP, 1)[0]


def realization_of(uid: str) -> str:
    parts = uid.split(UID_SEP, 1)
    return parts[1] if len(parts) == 2 else "native"


@dataclass(frozen=True)
class Realization:
    """One (semantic task, arm) pair, resolved to a concrete task directory."""

    semantic_id: str
    arm: str
    path: Path

    @property
    def uid(self) -> str:
        return make_uid(self.semantic_id, self.arm)


class RealizationIndex:
    """Which realizations each semantic task actually has.

    Built by scanning sibling arm directories `<dataset>--<arm>/` next to the
    stock dataset. A task is eligible for an arm iff the arm materialized it
    (inapplicable tasks are skipped by the generators, never copied).
    """

    def __init__(self, dataset_root: Path, arms: list[str] | None = None):
        self.dataset_root = Path(dataset_root)
        self.base_name = self.dataset_root.name
        self.parent = self.dataset_root.parent
        self._pools: dict[str, dict[str, Path]] = {}
        self._build(arms)

    def _arm_dirs(self, arms: list[str] | None) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for d in sorted(self.parent.glob(f"{self.base_name}--*")):
            if not d.is_dir():
                continue
            arm = d.name.split("--", 1)[1]
            if arms is not None and arm not in arms:
                continue
            found[arm] = d
        return found

    def _build(self, arms: list[str] | None) -> None:
        for arm, d in self._arm_dirs(arms).items():
            skipped: set[str] = set()
            manifest = d / "AFFECTED.json"
            if manifest.is_file():
                try:
                    data = json.loads(manifest.read_text())
                    skipped = set(data.get("skipped_inapplicable", []) or [])
                except Exception:  # noqa: BLE001 - a bad manifest must not hide tasks
                    skipped = set()
            for task_dir in d.iterdir():
                if not task_dir.is_dir() or not (task_dir / "task.toml").is_file():
                    continue
                if task_dir.name in skipped:
                    continue
                self._pools.setdefault(task_dir.name, {})[arm] = task_dir

    @property
    def semantic_ids(self) -> list[str]:
        return sorted(self._pools)

    def pool(self, semantic_id: str) -> dict[str, Path]:
        """Arms this semantic task actually has, arm -> task dir."""
        return dict(self._pools.get(semantic_id, {}))

    def coverage_report(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pool in self._pools.values():
            for arm in pool:
                counts[arm] = counts.get(arm, 0) + 1
        return dict(sorted(counts.items()))

    def sample(
        self,
        semantic_id: str,
        k: int,
        *,
        rng: random.Random,
        always_include: str | None = "native",
        seed_arms: Sequence[str] = (),
        exclude: set[str] | None = None,
    ) -> list[Realization]:
        """Sample K realizations of one semantic task.

        `always_include` pins an anchor arm (default `native`) in every
        super-group so cross-realization comparisons always share a reference
        point. Eval-only arms are excluded by the caller via `exclude`.

        `seed_arms` is the sparse-success device: a priority list of
        disclosure/easier arms, the first of which that this task has is pinned
        as the group's *success seed*. Seeds raise P(some realization succeeds),
        which activates the grounding ladder's strongest rung and lets credit
        flow from the seed's successful strategy to failed hard-arm rollouts.
        A task with no seed arm available just samples normally.

        Raises if the task cannot supply K distinct realizations — silently
        returning fewer would make super-group sizes vary in a way that
        contaminates the cross-realization normalization.
        """
        pool = self.pool(semantic_id)
        for arm in exclude or ():
            pool.pop(arm, None)
        if always_include and always_include not in pool:
            raise ValueError(f"{semantic_id}: anchor arm {always_include!r} not materialized")
        chosen: list[str] = []
        if always_include:
            chosen.append(always_include)
        for seed in seed_arms:
            if seed in pool and seed not in chosen and len(chosen) < k:
                chosen.append(seed)
                break  # one seed per group; the rest stays random
        others = sorted(a for a in pool if a not in chosen)
        need = k - len(chosen)
        if need > len(others):
            raise ValueError(
                f"{semantic_id}: need {k} realizations but only "
                f"{len(others) + len(chosen)} eligible ({sorted(pool)})"
            )
        chosen += rng.sample(others, need)
        return [Realization(semantic_id, arm, pool[arm]) for arm in chosen]


def eligible_semantic_ids(index: RealizationIndex, k: int, exclude: set[str] | None = None) -> list[str]:
    """Semantic tasks with at least K eligible realizations (anchor included)."""
    out = []
    for sid in index.semantic_ids:
        pool = set(index.pool(sid)) - set(exclude or ())
        if "native" in pool and len(pool) >= k:
            out.append(sid)
    return out
