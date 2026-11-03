# third_party/harbor — vendored harbor framework

A copy of the [harbor](https://github.com/harbor-framework/harbor) framework the experiments
ran with, from a fork at commit `9233373feb3fce2f4e7351cd75f3af064ff45643` (2026-09-10),
unchanged: `src/harbor/`, `packages/rewardkit/` (a uv workspace member the lockfile expects),
`pyproject.toml`, `uv.lock`, `README.md`, `LICENSE` (Apache-2.0) and `.python-version`. The
fork adds the `eidf` Kubernetes sandbox backend (`src/harbor/environments/eidf.py`).

Left out: harbor's own `adapters/` (the ones this paper uses are in this repo, under
`benchmarks/<benchmark>/harnesses/` and `tasks/`), `docs/`, `tests/`, `examples/`, `apps/`,
`scripts/`, `skills/`, `.github/`, `registry.json` and `rfcs/`.

`setup/install.sh` runs `uv sync --frozen --no-dev` here: exactly the dependency versions in
`uv.lock` (harbor 0.6.5, Python 3.13).
