"""Mixed-harness (XH) GRPO entrypoint — Option A (DispatchAgent path).

`main_harbor` with three substitutions (mirrors main_harbor_xr, minus primer/
gate/judge — there is no context machinery here, only harness grouping):

  dataset   -> HarnessGroupDataset: rows = task x harness, uid "<task>|<harness>";
               per-row harness selection rides env_extras["agent_kwargs"] into
               the DispatchAgent via HarborGenerator's additive passthrough.
  sampler   -> XRGroupSampler with k = len(harnesses): super-groups (all of a
               task's harness rows) stay contiguous within a batch.
  estimator -> grpo_xr reused unchanged with HARNESS as the realization axis:
               within-(task,harness) centering + lambda * cross-harness term.

New config block (plain dict on the CLI, like harbor_trial_config):
  xh: dataset_root, split_file (opt), split_key, harnesses (list of dispatch
      registry names), harness_kwargs (per-harness overrides), lambda_ground,
      drop_task_prefixes, norm_grounding_by_std

Design doc: harbor docs/eidf/multi-harness-rollout-options.md (Option A).
"""

import sys

import ray
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ..xr import advantage as xr_advantage
from ..xr.harness_dataset import HarnessGroupDataset
from .main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
)
from ..xr.group_trainer import XRTrainer


@dataclass
class HarborXHSkyRLConfig(HarborSkyRLConfig):
    xh: Dict[str, Any] = field(default_factory=dict)


class HarborXHExp(HarborExp):
    def __init__(self, cfg, *args, **kwargs):
        xh = dict(cfg.xh)
        for required in ("dataset_root", "harnesses"):
            if not xh.get(required):
                raise ValueError(f"xh.{required} is required")
        self._xh = xh
        xr_advantage.register(
            xr_mode=xh.get("mode", "within_cross"),
            xr_lambda=float(xh.get("lambda_ground", 0.5)),
            xr_norm_grounding_by_std=bool(xh.get("norm_grounding_by_std", False)),
        )
        print(f"XH: harnesses={list(xh['harnesses'])} lambda={xh.get('lambda_ground', 0.5)}")
        super().__init__(cfg, *args, **kwargs)

    def get_train_dataset(self):
        xh = self._xh
        ds = HarnessGroupDataset(
            dataset_root=xh["dataset_root"],
            harnesses=list(xh["harnesses"]),
            split_file=xh.get("split_file"),
            split_key=xh.get("split_key", "train"),
            drop_task_prefixes=tuple(xh.get("drop_task_prefixes") or ()),
            harness_kwargs=dict(xh.get("harness_kwargs") or {}),
        )
        bs = self.cfg.trainer.train_batch_size
        if bs % ds.k != 0:
            raise ValueError(f"train_batch_size={bs} not divisible by k={ds.k}")
        return ds

    def get_trainer(
        self, cfg, tracker, tokenizer, train_dataset, eval_dataset,
        inference_engine_client, generator, colocate_pg,
    ):
        return XRTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
            xr_k=train_dataset.k,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = HarborXHExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborXHSkyRLConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    xr_advantage.register(
        xr_mode=cfg.xh.get("mode", "within_cross"),
        xr_lambda=float(cfg.xh.get("lambda_ground", 0.5)),
    )
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be explicitly set")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
