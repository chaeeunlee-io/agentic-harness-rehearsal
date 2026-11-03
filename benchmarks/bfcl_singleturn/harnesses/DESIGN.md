# BFCL single-turn harnesses

Each task contains one request and a list of function definitions. The model returns the calls to make, possibly none, in `/app/result.json`. Grading compares those calls with the ground truth by syntax tree; nothing is executed and no model judge is used.

Because there is no interactive environment, the harnesses mainly differ in the format of a single response. This makes BFCL single-turn a relatively clean test of output format, with enough tasks to give narrow confidence intervals. On the other benchmarks, harnesses also differ in how the agent interacts with the environment.

| harness | role | model output | implemented by |
|---|---|---|---|
| `fc` | trained | native tool calls | `bfcl_agent.py` + `run_bfcl_agent.py`, protocol `fc` |
| `json` | trained | `{"analysis": "...", "calls": [{"name": ..., "arguments": {...}}]}` as text | same, protocol `json` |
| `t2` | trained | shell commands that write `/app/result.json` | Harbor's Terminus-2 (JSON parser, up to 8 episodes) |
| `xml` | held out | `<reasoning>...</reasoning><calls><call name="...">{...}</call></calls>` | same as `fc`, protocol `xml` |

`configs/benchmarks/bfcl_singleturn.yaml` points each harness to `harness_dispatch.py`, which maps `bfcl-<harness>` to the corresponding agent. `bfcl_agent.py` runs on the Harbor side and starts `run_bfcl_agent.py` inside the task container. `bfcl_protocols.py` defines the three formats: their prompts, how functions are presented, and how responses are parsed.

`fc`, `json`, and `xml` differ only in how calls are represented, so comparisons between them isolate format more closely. `t2` differs in both output format and agent structure because it is a multi-turn shell agent, so gaps involving `t2` are not purely format effects.

## Details that affect results

- All tasks, including the Java and JavaScript categories, use the same `python:3.10-slim` image. Since calls are not executed, separate language-specific images are unnecessary.
- `fc` uses converted function schemas because the native tool interface requires valid JSON Schema. `bfcl_protocols.py` converts BFCL-specific types such as `"type": "dict"` and drops types it cannot map. A single invalid schema can cause the server to reject the request. `json` and `xml` use the original BFCL definitions, matching the task instruction.
- `fc` replaces `.` with `_` in dotted function names because tool names cannot contain periods. The grader applies the same normalisation, and no two functions within a task map to the same name. These assumptions should be rechecked if the BFCL data changes.
- `tool_choice` is `auto`, not `required`. Of the 355 test tasks, 80 (22.5%) are relevance or irrelevance cases where the correct answer may be to make no call. `required` would force a call and also suppress model reasoning.
- `t2` requires a `model_info` block in its settings because Harbor's client expects one for locally served models. The configuration provides it. `fc`, `json`, and `xml` call the model from inside the container and do not need this block.

## Splits and protocol

- Harbor's BFCL adapter provides 13 single-turn categories and 3,641 tasks. Capping each category at 150 leaves 1,406 tasks, split into 1,051 train and 355 test examples in `splits/split.json` using seed `20260907`.
- Evaluation uses one greedy attempt per task with temperature 0.
- Training uses `3h = fc + json + t2`. `xml` is held out and is never used for teacher traces or training.
