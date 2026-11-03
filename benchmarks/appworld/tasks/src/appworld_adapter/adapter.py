"""AppWorldAdapter - converts AppWorld tasks into Harbor format.

Unlike reasoning-gym (which procedurally GENERATES entries from a library at
adapter-run time), AppWorld's tasks are static, pre-existing data shipped by
the `appworld download data` command -- this adapter's job is just to
enumerate task_ids (via appworld.load_task_ids, the package's own split
accessor) and materialize one Harbor task dir per task_id. The actual task
data (per-task DB fixtures, ground_truth/evaluation.py, the full apps engine)
is NOT copied per-task into the Harbor task dir -- it all ships baked into
the shared environment/Dockerfile image (appworld==0.1.3 + `appworld install`
+ `appworld download data`, all network-free at container runtime). Each
generated Harbor task only needs to know WHICH task_id to instantiate at
runtime, which reaches the sandbox via task.toml's
`[environment].env.APPWORLD_TASK_ID` (see task-template/task.toml).

Requires the `appworld` package (and its downloaded data) available on
whatever host RUNS this adapter, since `load_task_ids`/task specs are read
directly from the package: `pip install appworld==0.1.3 && appworld install
&& appworld download data` first (see README.md).

License note: AppWorld's ground_truth/evaluation.py content is licensed
Apache-2.0 "with the additional requirement that any public redistribution
...must also be done in an encrypted format" -- this adapter never copies
that content into a generated task dir (it stays inside the baked image,
read directly by evaluate_task() at verify time), and harbor's own
`/datasets/` is gitignored at the repo root regardless, so generated task
dirs are never committed. Do not change this adapter to inline any
ground_truth content into a task dir's plain files.

Source: https://github.com/StonyBrookNLP/appworld
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

TEMPLATE_DIR = Path(__file__).parent / "task-template"

VALID_SPLITS = ("train", "dev", "test_normal", "test_challenge")


class AppWorldAdapter:
    NAME = "appworld"

    def __init__(
        self,
        output_dir: Path,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.splits = self._parse_splits(kwargs.get("splits"))
        self.limit = kwargs.get("limit")
        self.task_ids_filter = kwargs.get("task_ids")
        # The task image, built with benchmarks/appworld/tasks/build_appworld_image.sh
        self.docker_image = kwargs.get("docker_image")
        if not self.docker_image:
            raise ValueError("docker_image is required: build the image with build_appworld_image.sh")

    @staticmethod
    def _parse_splits(raw: Any) -> list[str]:
        if raw is None:
            splits = ["train", "dev", "test_normal"]  # test_challenge opt-in only
        elif isinstance(raw, str):
            splits = [s.strip() for s in raw.split(",") if s.strip()]
        else:
            splits = [str(s).strip() for s in raw if str(s).strip()]
        invalid = [s for s in splits if s not in VALID_SPLITS]
        if invalid:
            raise ValueError(
                f"Unsupported split(s): {invalid}. Supported: {VALID_SPLITS}"
            )
        return splits

    def run(self) -> int:
        from appworld import load_task_ids

        selected = set(self.task_ids_filter) if self.task_ids_filter else None
        total = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split in self.splits:
            for task_id in load_task_ids(split):
                if selected and task_id not in selected:
                    continue
                self._generate_task(task_id, split)
                total += 1
                if self.limit is not None and total >= self.limit:
                    return total
        return total

    def _generate_task(self, task_id: str, split: str) -> None:
        import json

        local_task_id = f"appworld-{task_id}"
        output_dir = self.output_dir / local_task_id
        if output_dir.exists():
            if self.overwrite:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(f"Task directory already exists: {output_dir}")

        try:
            self._copy_template(output_dir)
            self._customize_task(output_dir, task_id, split)
        except Exception:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            raise

    def _copy_template(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in TEMPLATE_DIR.iterdir():
            dst = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)

    def _customize_task(self, output_dir: Path, task_id: str, split: str) -> None:
        import json

        from appworld.common.path_store import path_store

        specs_path = Path(path_store.data) / "tasks" / task_id / "specs.json"
        specs = json.loads(specs_path.read_text(encoding="utf-8"))

        # scenario_id: appworld task_ids are "{scenario}_{variation}" (e.g.
        # "82e2fac_1") -- the same underlying scenario reworded/reparameterized
        # across a few numbered tasks. Keeping this in task.toml's keywords
        # lets a later split/dedup step group by scenario if that turns out
        # to matter (unverified whether it does -- flagging, not assuming).
        scenario_id = task_id.rsplit("_", 1)[0]

        instruction_path = output_dir / "instruction.md"
        instruction_path.write_text(
            (TEMPLATE_DIR / "instruction.md")
            .read_text()
            .format(instruction=specs["instruction"]),
            encoding="utf-8",
        )

        # .replace(), not .format() -- task.toml's `authors = [{ name = ... }]`
        # is a literal TOML inline table; .format() chokes on those braces
        # (hit this live generating the first smoke batch).
        toml_text = (TEMPLATE_DIR / "task.toml").read_text()
        toml_text = (
            toml_text.replace("{task_id}", f"appworld-{task_id}")
            .replace("{appworld_task_id}", task_id)
            .replace("{scenario_id}", scenario_id)
            .replace("{difficulty}", split)
            .replace("{docker_image}", self.docker_image)
        )
        (output_dir / "task.toml").write_text(toml_text, encoding="utf-8")

        # Metadata only (never ground_truth content) -- safe to leave as a
        # plain file per the license note above.
        meta_path = output_dir / "tests" / "appworld_meta.json"
        meta_path.write_text(
            json.dumps(
                {"task_id": task_id, "split": split, "scenario_id": scenario_id},
                indent=2,
            ),
            encoding="utf-8",
        )
