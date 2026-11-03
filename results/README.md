# Result data

`cells.csv` contains every reported result, with one row per evaluation cell and metric.

```text
benchmark, model, tier, layer, arm, harness, metric, value,
ci_lo, ci_hi, plusminus, tasks, trials, bootstrap_unit, bootstrap_n, status, note
```

| column | meaning |
|---|---|
| `benchmark` | Benchmark name accepted by `--benchmark`: `medagentbench`, `reasoning_gym`, `appworld`, `bfcl_singleturn`, or `bfcl_multiturn` |
| `model` | Evaluated policy: a key from `configs/models.yaml` (`qwen3-4b`, `gemma-3-4b-it`, `qwen3-32b`), or `qwen3.8-27b` for the extra 27B teacher rows |
| `tier` | Reasoning Gym tier (`easy` or `hard`); empty for other benchmarks |
| `layer` | Sampling setup: `n4-T0.7`, `n4-T0.6` (4 attempts at that temperature), `n4-gencfg-default` (4 attempts using the released agent's generation config), `pass1-greedy` (1 greedy attempt), or `n4-T0.7-SUBSET61` (AppWorld, first 61 test tasks) |
| `arm` | Training setting. For SFT settings, names match `benchmarks/<benchmark>/pools/<arm>.jsonl` and the corresponding key in `configs/benchmarks/<benchmark>.yaml`; see below |
| `harness` | Harness used for evaluation, including held-out harnesses; empty for set-level counts |
| `metric` | `mean` (mean verifier reward, percent), `pass4` (percent of tasks solved in at least one of 4 attempts), or solve-set counts: `solved`, `unique-<set>`, `union-<set>`, and `common-<set>`. `<set>` is `trained` for the three training harnesses or `all` when the held-out harness is included |
| `value` | Percentage for `mean` and `pass4`; task count for solve-set metrics |
| `ci_lo`, `ci_hi`, `plusminus` | Percentile bootstrap interval and half-width. Some rows only report `plusminus` |
| `tasks`, `trials` | Number of tasks and trials in the cell |
| `bootstrap_unit`, `bootstrap_n` | Resampling unit and number of units: tasks, or AppWorld's 56 scenarios |
| `status` | `reported` (reported result), `superseded` (replaced by another result; see `note`), or `extra` (computed but not reported) |
| `note` | Provenance, such as the original job ID, original arm name, or how the row was produced |

## Setting names

The same naming scheme is used across benchmarks. Harness names come from `configs/benchmarks/<b>.yaml`.

| name | trained on |
|---|---|
| `base` | No training; the base model |
| `teacher` | The teacher model itself; `model` identifies which teacher |
| `<h>-only` | Teacher demonstrations from harness `h` |
| `3h` | Demonstrations from all three training harnesses |
| `<h1>+<h2>` | Demonstrations from two harnesses |
| `reh-<h>` | Rehearsal traces from harness `h` only |
| `<...>+reh-<h>` | The preceding dataset plus rehearsal traces from `h` |
| `-32b`, `-gemma`, `-fail` | Rehearsal traces from the 32B teacher, Gemma, or failed attempts; otherwise rehearsal uses the 4B student |
| `3h-n108`, `3h-n150` (`-cov`, `-s2`, `-s3`) | `3h` reduced to 108 or 150 rows using random sampling, coverage-maximising selection, or alternate seeds |
| `3h-matched` | AppWorld `3h` reduced to 74 rows to match `code-only` |
| `<h>-only-27b`, `mixed-27b` | Students distilled from the extra 27B teacher traces, using one harness or a mixture; these appear only as `extra` rows |
| `grpo-<x>` | GRPO starting from the `3h` SFT checkpoint, with rollouts on `x`; `grpo-3h` uses all three training harnesses. `-step5` denotes the step-5 checkpoint |

## Rebuilding and comparing

```bash
uv run exp table <experiment> --shipped
uv run exp table <experiment> --format cells --out mine.csv
```

The first command reads the shipped results. The second writes results from a local run in the same schema, so they can be compared directly with `cells.csv`.

## Reading the results

- Reward meaning differs by benchmark. MedAgentBench, BFCL, and AppWorld use binary verifier rewards. Reasoning Gym uses scores in `[0, 1]`, so its means include partial credit and `solved` means near-full credit.
- Means use completed trials. Cells with substantial infrastructure failures were rerun rather than reported.
- Per-trial outcomes are not included in the release. Rows whose notes say `computed from per-trial outcomes` were produced from those outcomes with `exp/table.py`.
- Confidence intervals use percentile bootstrap with `B=1000` and a fixed seed, resampling the unit named in `bootstrap_unit`. The `grpo-3h` rows use the trial-pooled statistic (marked in their `note`), and `exp table grpo-3h-vs-single` shows its `3h` rows the same way, as in the paper; other rows use the mean of task means.
