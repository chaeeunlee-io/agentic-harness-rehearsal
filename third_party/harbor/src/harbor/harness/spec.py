"""The harness arm: a validated knob vector.

Every field's default is stock behaviour (identity transform). Unknown keys and
invalid values raise instead of being silently ignored — a declared-but-dead config
field is how sweeps go wrong without anyone noticing
(see pi-task-bind docs/HARNESS_IMPLEMENTATION_STATUS.md, "declared-but-partial").
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

# Allowed values per knob. int-typed knobs validate as non-negative ints instead.
_ENUMS: dict[str, tuple[str, ...]] = {
    "answer_contract": ("full", "code_only", "bare"),
    "sort_support": ("honored", "ignored"),
    "code_system_form": ("bare", "qualified"),
    "response_verbosity": ("full", "trimmed"),
    "long_output_mode": ("native", "head", "tail"),
    "issue_detail": ("full", "title_only"),
    "code_blocks": ("kept", "stripped"),
    # Task-agnostic harness knobs (docs/perturbation-candidates/task_agnostic_harness_knobs.md)
    "budget_disclosure": ("none", "at_start", "every_turn", "every_k"),
    "parse_failure_mode": ("native", "silent", "with_hint"),
    "thought_retention": ("kept", "stripped"),
    "history_compaction": ("native", "off", "at_k"),
    "workspace_tree": ("none", "listing", "tree_d2"),
    "repo_orientation": ("undisclosed", "path_only", "path_and_tree"),
    "entry_cwd": ("native", "home", "parent"),
}


@dataclass(frozen=True)
class HarnessArm:
    """One arm = one point in knob space. Defaults are all stock."""

    name: str = "native"

    # Offline knobs — consumed by the variant generator at task-materialization
    # time; carried here so runtime records (trial metadata) show full provenance.
    answer_contract: str = "full"

    # Runtime knobs — consumed by the tool-layer kernel (fhir.py).
    sort_support: str = (
        "honored"  # ignored: strip _sort; "most recent" needs client sort
    )
    code_system_form: str = (
        "bare"  # qualified: bare codes return an empty-but-valid bundle
    )
    response_verbosity: str = "full"  # trimmed: strip known-bloat FHIR fields
    max_fhir_chars: int = 0  # >0: hard-truncate response text (mid-JSON cuts allowed)
    bundle_count: int = 0  # >0: inject _count=n into searches (pagination pressure)
    link_stripped: bool = (
        False  # remove Bundle.link[rel=next]: page 1 poses as everything
    )

    # Runtime knobs — consumed by the terminal-text kernel (text.py; terminus-2 seam).
    long_output_mode: str = "native"  # which region survives a harness cut: head | tail
    max_output_chars: int = 0  # >0: cut terminal output to n chars per long_output_mode

    # Offline knobs — SWE-bench instruction materialization (make_arm_variants).
    issue_detail: str = "full"  # title_only: issue body reduced to its title line
    code_blocks: str = "kept"  # stripped: fenced blocks (repro/tracebacks) removed

    # ------------------------------------------------------------------
    # Task-agnostic harness knobs — consumed by orient.py at the terminus-2
    # seam. They perturb the agent's model of its own execution (horizon,
    # memory, error visibility, orientation), never the task. Stock values
    # reproduce current behaviour exactly.
    # Design: docs/perturbation-candidates/task_agnostic_harness_knobs.md
    # ------------------------------------------------------------------
    budget_disclosure: str = "none"  # at_start | every_turn | every_k: show turns left
    budget_every_k: int = 5  # cadence for budget_disclosure="every_k"
    parse_failure_mode: str = "native"  # silent: drop WARNINGS; with_hint: add format hint
    thought_retention: str = "kept"  # stripped: reasoning not carried in later history
    history_compaction: str = "native"  # off: disable summarization; at_k: force at turn k
    history_compaction_k: int = 10  # turn index for history_compaction="at_k"
    workspace_tree: str = "none"  # listing | tree_d2: inject a real listing at setup
    repo_orientation: str = "undisclosed"  # path_only | path_and_tree: name the repo root
    entry_cwd: str = "native"  # home | parent: where the shell session starts

    def is_stock(self) -> bool:
        """True iff every knob (name aside) is at its identity default."""
        default = HarnessArm(name=self.name)
        return self == default

    def non_stock_knobs(self) -> dict[str, Any]:
        """The knobs that differ from stock — what trial metadata should record."""
        default = HarnessArm()
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name == "name":
                continue
            value = getattr(self, f.name)
            if value != getattr(default, f.name):
                out[f.name] = value
        return out

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, source: str = "harness config"
    ) -> HarnessArm:
        """Build an arm from a plain dict, failing loudly on anything unrecognised."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - field_names
        if unknown:
            raise ValueError(
                f"{source}: unknown harness knob(s) {sorted(unknown)}; "
                f"allowed: {sorted(field_names)}"
            )
        for key, allowed in _ENUMS.items():
            if key in raw and raw[key] not in allowed:
                raise ValueError(f"{source}: {key}={raw[key]!r} not in {list(allowed)}")
        for key in (
            "max_fhir_chars",
            "bundle_count",
            "max_output_chars",
            "budget_every_k",
            "history_compaction_k",
        ):
            if key in raw and (not isinstance(raw[key], int) or raw[key] < 0):
                raise ValueError(f"{source}: {key} must be a non-negative int")
        if "link_stripped" in raw and not isinstance(raw["link_stripped"], bool):
            raise ValueError(f"{source}: link_stripped must be a bool")
        if "name" in raw and not isinstance(raw["name"], str):
            raise ValueError(f"{source}: name must be a string")
        return cls(**raw)

    def merged_with(
        self, override: dict[str, Any] | None, *, source: str = "override"
    ) -> HarnessArm:
        """Apply a partial override dict on top of this arm (override wins).

        Precedence rule (documented in the design doc): the per-run config override
        (TrialConfig agent.kwargs.harness_config) beats the task-dir arm — that is
        what makes global sweeps cheap.
        """
        if not override:
            return self
        # Validate the override in isolation so its errors name the right source.
        HarnessArm.from_dict(override, source=source)
        return dataclasses.replace(self, **override)
