"""Multi-harness super-group dataset (Option A / DispatchAgent path).

Rows: task x harness ladder; uid = "<task>|<harness>" so the existing
grouping helpers (semantic_of / realization_of) and the grpo_xr dual-level
estimator apply unchanged with harness as the realization axis:
within-(task,harness) centering + lambda * cross-harness centering.

Per-row harness selection rides env_extras["agent_kwargs"] -> merged into the
trial's agent kwargs by HarborGenerator (additive passthrough) -> consumed by
adapters.medagentbench.harness_dispatch:DispatchAgent.

NEW FILE: the existing XRSuperGroupDataset (arm axis) is untouched.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class HarnessGroupDataset:
    """Minimal torch-style dataset: one item per (task, harness) row.

    The trainer's sampler multiplies rows by n_samples_per_prompt (M), so a
    super-group is len(harnesses) x M trajectories, exactly like the arm
    design. Group-preserving shuffling is handled by the existing
    XRGroupSampler with k = len(harnesses).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        harnesses: list[str],
        *,
        split_file: str | Path | None = None,
        split_key: str = "train",
        drop_task_prefixes: tuple[str, ...] = (),
        harness_kwargs: dict[str, dict] | None = None,
        seed: int = 0,
    ):
        self.dataset_root = Path(dataset_root)
        if not harnesses:
            raise ValueError("harnesses must be a non-empty list")
        self.harnesses = list(harnesses)
        self.harness_kwargs = harness_kwargs or {}
        self._rng = random.Random(seed)

        tasks = sorted(p.name for p in self.dataset_root.iterdir() if p.is_dir())
        if split_file is not None:
            split = json.loads(Path(split_file).read_text())
            keep = set(split[split_key] if isinstance(split, dict) else split)
            tasks = [t for t in tasks if t in keep]
        if drop_task_prefixes:
            prefixes = tuple(p for p in drop_task_prefixes if p.endswith("_"))
            exact = {p for p in drop_task_prefixes if not p.endswith("_")}
            before = len(tasks)
            tasks = [t for t in tasks
                     if t not in exact and not any(t.startswith(p) for p in prefixes)]
            logger.warning("XH: dropped %d tasks (%s)", before - len(tasks),
                           list(drop_task_prefixes))
        if not tasks:
            raise ValueError("harness dataset resolved zero tasks")
        self.tasks = tasks
        logger.warning(
            "XH dataset: %d tasks x %d harnesses = %d rows (K=%d)",
            len(tasks), len(self.harnesses),
            len(tasks) * len(self.harnesses), len(self.harnesses),
        )

    @property
    def k(self) -> int:  # super-group width, for XRGroupSampler
        return len(self.harnesses)

    def __len__(self) -> int:
        return len(self.tasks) * len(self.harnesses)

    def __getitem__(self, idx: int):
        task = self.tasks[idx // len(self.harnesses)]
        harness = self.harnesses[idx % len(self.harnesses)]
        agent_kwargs = {"harness": harness, **self.harness_kwargs.get(harness, {})}
        return {
            "prompt": str(self.dataset_root / task),
            "env_class": None,
            "uid": f"{task}|{harness}",
            "env_extras": {"agent_kwargs": agent_kwargs},
        }

    def collate_fn(self, batch):
        # prepare_generator_input consumes a LIST of row dicts (see
        # skyrl.train.generators.utils) — do not repack into dict-of-lists.
        return batch
