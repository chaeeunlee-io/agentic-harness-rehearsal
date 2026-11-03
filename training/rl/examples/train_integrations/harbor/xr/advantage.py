"""Cross-realization GRPO advantage: two comparison sets, one advantage.

    A_{k,m} = A^out_{k,m} + lambda * A^ground_{k,m}

    A^out    normalized WITHIN realization k   ("better than peers under the
                                                 same knob?")  -> realization
                                                 difficulty cannot contaminate
                                                 ordinary credit assignment
    A^ground normalized ACROSS realizations of  ("did this reasoning survive a
             the same semantic task s            change of realization?")

Benchmark-agnostic: the estimator only reads uids (`<semantic>|<realization>`)
and a per-rollout grounding vector. It has no notion of FHIR, notebooks or
repositories — those live in `xr/extractors/`.

Registered with SkyRL's `AdvantageEstimatorRegistry` as **`grpo_xr`**, so it is
selected with `trainer.algorithm.advantage_estimator=grpo_xr` and requires no
change to SkyRL's estimator dispatch.

Ablation modes (`xr.mode`), matching the paper's four cells:

    off              K=1; plain GRPO. Identical to `grpo` (verified in tests).
    within           K>1, outcome normalized within realization, no grounding.
                     Isolates "is the gain just extra augmentation?"
    pooled           K>1, outcome normalized over the whole super-group.
                     Isolates "does realization-aware normalization matter?"
    full             within + lambda * grounding. The proposed objective.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .grouping import semantic_of

MODES = ("off", "within", "pooled", "full", "within_cross")


def _group_normalize(
    scores: torch.Tensor,
    keys: list[str],
    *,
    norm_by_std: bool,
    epsilon: float,
) -> torch.Tensor:
    """Mean-center (optionally std-scale) scores within each key's group.

    A singleton group yields 0 — with one sample there is no relative signal,
    and emitting a raw score would let group size leak into the gradient.
    """
    out = scores.clone()
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        buckets[k].append(i)
    for idxs in buckets.values():
        if len(idxs) == 1:
            out[idxs[0]] = 0.0
            continue
        vals = scores[idxs]
        centered = vals - vals.mean()
        if norm_by_std:
            centered = centered / (vals.std(unbiased=True) + epsilon)
        for pos, i in enumerate(idxs):
            out[i] = centered[pos]
    return out


def compute_xr_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    grpo_norm_by_std: bool = True,
    grounding_scores: np.ndarray | list[float] | None = None,
    xr_mode: str = "full",
    xr_lambda: float = 0.5,
    xr_norm_grounding_by_std: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """See module docstring. Signature mirrors SkyRL's `compute_grpo_outcome_advantage`.

    Args:
        index: uid per rollout, `"<semantic>|<realization>"`. Plain uids (no
            separator) degrade gracefully to single-realization behaviour.
        grounding_scores: G per rollout, aligned with `index`. Required for
            `xr_mode="full"`; ignored otherwise.
        xr_mode: one of MODES.
        xr_lambda: weight on the grounding advantage.
    """
    if xr_mode not in MODES:
        raise ValueError(f"xr_mode={xr_mode!r} not in {MODES}")

    scores = token_level_rewards.sum(dim=-1)
    uids = [str(u) for u in index]

    with torch.no_grad():
        if xr_mode == "within_cross":
            # v2 objective (design doc §2): verified outcome normalized twice.
            #   A = (R − μ_k)/(σ_k+ε)  +  λ·(R − μ_s)
            # Cross term is mean-centered ONLY — dividing by σ_s as well would
            # stack the small-σ difficulty blowup twice (Dr. GRPO critique).
            # Properties (unit-tested): per-cell affine in R with positive
            # slope; no failure ever positive, no success ever negative.
            within = _group_normalize(
                scores, uids, norm_by_std=grpo_norm_by_std, epsilon=epsilon
            )
            cross = _group_normalize(
                scores,
                [semantic_of(u) for u in uids],
                norm_by_std=False,
                epsilon=epsilon,
            )
            adv = within + xr_lambda * cross
            adv = adv.unsqueeze(-1) * response_mask
            return adv, adv

        if xr_mode == "pooled":
            outcome_keys = [semantic_of(u) for u in uids]
        else:
            # "off" and "within" both key on the full uid: with K=1 the uid IS
            # the realization, so "off" reduces to plain GRPO by construction.
            outcome_keys = uids
        adv = _group_normalize(
            scores, outcome_keys, norm_by_std=grpo_norm_by_std, epsilon=epsilon
        )

        if xr_mode == "full":
            if grounding_scores is None:
                raise ValueError(
                    "xr_mode='full' requires grounding_scores; none were passed. "
                    "Check the generator attached them and the trainer forwarded them."
                )
            g = torch.as_tensor(
                np.asarray(grounding_scores, dtype=np.float32), device=scores.device
            )
            if g.shape[0] != scores.shape[0]:
                raise ValueError(
                    f"grounding_scores has {g.shape[0]} entries for {scores.shape[0]} rollouts"
                )
            ground_adv = _group_normalize(
                g,
                [semantic_of(u) for u in uids],
                norm_by_std=xr_norm_grounding_by_std,
                epsilon=epsilon,
            )
            adv = adv + xr_lambda * ground_adv

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


def register(
    name: str = "grpo_xr",
    *,
    xr_mode: str = "within_cross",
    xr_lambda: float = 0.5,
    xr_norm_grounding_by_std: bool = True,
) -> None:
    """Register with SkyRL, binding the XR settings at registration time.

    Binding matters: SkyRL's trainer calls estimators with a fixed argument
    set (rewards/mask/index/config/...) and no xr kwargs — without a bound
    partial, `grpo_xr` would silently run at whatever the function defaults
    are rather than the run's config. Call BEFORE validate_cfg (which checks
    the estimator name against the registry).
    """
    import functools

    from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

    fn = functools.partial(
        compute_xr_advantage,
        xr_mode=xr_mode,
        xr_lambda=xr_lambda,
        xr_norm_grounding_by_std=xr_norm_grounding_by_std,
    )
    functools.update_wrapper(fn, compute_xr_advantage)
    if name in set(AdvantageEstimatorRegistry.list_available()):
        AdvantageEstimatorRegistry.unregister(name)
    AdvantageEstimatorRegistry.register(name, fn)
