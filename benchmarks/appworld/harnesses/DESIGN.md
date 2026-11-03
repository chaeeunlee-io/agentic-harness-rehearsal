# AppWorld harnesses

AppWorld actions are Python programs executed against a stateful world with 457 APIs. Variables and state persist across turns, so all harnesses keep the underlying action the same and vary only the interface around it. Each turn still submits a complete Python program.

Exposing every API as a separate tool would change the action space by removing things like loops and persistent variables. For that reason, `fc` exposes one tool, `execute_python(code)`, rather than one tool per API.

| harness | role | model output | implemented by |
|---|---|---|---|
| `code` | trained | fenced ```` ```python ```` block | `appworld_code_agent.py` + `run_appworld_code_agent.py`, protocol `code` |
| `fc` | trained | `execute_python(code=...)` tool call | same, protocol `fc` |
| `mini` | trained | Python in mini-swe-agent's action block | `appworld_mini_swe_agent.py` (Harbor mini-swe-agent 2.4.6, adapted) |
| `xml` | held out | `<reasoning>...</reasoning><code>...</code>` | same as `code`, protocol `xml` |

`configs/benchmarks/appworld.yaml` points every harness at `harness_dispatch.py` (its `import_path`), which maps `appworld-<harness>` to the corresponding agent class.

## code, fc, and xml

`appworld_code_agent.py` uploads `run_appworld_code_agent.py` and `appworld_protocols.py` to the task container and starts the runner with the selected harness.

The runner repeatedly asks the model for an action, extracts the Python, executes it in AppWorld, and returns the result until the task finishes or the budget is exhausted. The configured limits are 50 steps and 40 errors.

`appworld_protocols.py` defines the three formats. Each has its own system prompt and parser.

## mini

mini-swe-agent normally runs each action in a separate process, while AppWorld requires state to persist across actions. AppWorld's environment server therefore holds the world state, and the agent sends actions to it over HTTP.

- `aw_env_init.py` waits for the server and loads the task world.
- `appworld_mswe_env.py` sends Python to the world server instead of running shell commands, and stops when AppWorld reports completion.
- `appworld_mswe.yaml` defines mini-swe-agent's prompt, parsing, sampling (`temperature: 0.6`), 50-step limit, and environment class.
- `aw_env_save.py` saves the final world state for verification.
- `appworld_mini_swe_agent.py` uploads these files, starts the server, and launches Harbor's mini-swe-agent.

The HTTP boundary also separates incompatible dependencies: mini-swe-agent requires pydantic v2, while AppWorld requires v1. The adapter stores the model endpoint in process-wide environment variables. This is safe for the reported runs because concurrent trials within a job use the same endpoint.

## Distillation dataset rules

1. Each harness has its own prompt and action-format instructions.
2. Distillation data keeps the model's literal replies, including the output envelope, rather than only extracted code.
3. Rows are checked for a non-empty render, reasoning, and the harness's action form in `build/build_pools.py`.
4. `xml` is held out and is not used for teacher traces or training.

## Splits and budgets

- Training uses AppWorld's `train` split: 90 tasks from 30 scenarios. Evaluation uses the 168 `test_normal` tasks in `splits/test.json`. `dev` and `test_challenge` are unused.
- Bootstrap resampling is over AppWorld's 56 test scenarios, not the 168 individual tasks.
- `code`, `fc`, and `xml` use limits of 50 steps and 40 errors. `mini` uses 50 steps.
