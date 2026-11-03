"""Group-preserving dataloader machinery, shared by realization- and
harness-axis training (extracted from main_harbor_xr so the current
mixed-harness entrypoint does not depend on the legacy XR pipeline)."""

import torch
from skyrl.train.trainer import RayPPOTrainer
from torchdata.stateful_dataloader import StatefulDataLoader


class XRGroupSampler(torch.utils.data.Sampler):
    """Shuffle SUPER-GROUPS, never rows.

    SkyRL's stock dataloader shuffles at row level, which scatters a semantic
    task's K rows across batches — the gate loop then sees partial groups and
    the cross-realization normalization runs over fragments (observed live:
    a K=3 group re-rolled as 8 trajectories instead of 12). Rows stay in
    dataset order inside a group; group ORDER is re-permuted every epoch.
    """

    def __init__(self, n_rows: int, k: int, seed: int):
        if n_rows % k != 0:
            raise ValueError(f"{n_rows} rows not divisible by K={k}")
        self.n_rows, self.k, self.seed = n_rows, k, seed
        self.epoch = 0

    def __len__(self):
        return self.n_rows

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for group in torch.randperm(self.n_rows // self.k, generator=g).tolist():
            yield from range(group * self.k, (group + 1) * self.k)


class XRTrainer(RayPPOTrainer):
    """RayPPOTrainer with the group-aware dataloader.

    xr_k must be set before super().__init__ — the base constructor invokes the
    dataloader hook (trainer.py:126)."""

    def __init__(self, *args, xr_k: int = 1, **kwargs):
        self.xr_k = xr_k
        super().__init__(*args, **kwargs)

    def _build_train_dataloader_and_compute_training_steps(self):
        if self.train_dataset is None or self.xr_k <= 1:
            return super()._build_train_dataloader_and_compute_training_steps()
        cfg = self.cfg
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_size=cfg.trainer.train_batch_size,
            sampler=XRGroupSampler(len(self.train_dataset), self.xr_k, cfg.trainer.seed),
            collate_fn=self.train_dataset.collate_fn,
            num_workers=cfg.data.dataloader.num_workers,
            persistent_workers=cfg.data.dataloader.persistent_workers,
            drop_last=True,
            multiprocessing_context="spawn" if cfg.data.dataloader.num_workers > 0 else None,
        )
        self.total_training_steps = len(self.train_dataloader) * cfg.trainer.epochs
        if cfg.trainer.max_training_steps is not None:
            self.total_training_steps = min(self.total_training_steps, cfg.trainer.max_training_steps)


