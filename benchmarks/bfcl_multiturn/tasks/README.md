# BFCL multi-turn: tasks

What is BFCL's and what is ours. BFCL is the Berkeley Function Calling Leaderboard
([gorilla](https://github.com/ShishirPatil/gorilla), `berkeley-function-call-leaderboard/`);
compared against its commit `6ea5797` (2026-03-23).

| path | origin |
|---|---|
| `BFCL_v4_multi_turn_{base,long_context,miss_func,miss_param}.json` | BFCL, unchanged: the 800 multi-turn tasks (`bfcl_eval/data/`) |
| `possible_answer/` | BFCL, unchanged: ground-truth call sequences |
| `multi_turn_func_doc/` | BFCL, unchanged: the function documentation the model is given |
| `bfcl_env/` | BFCL's API simulators and multi-turn checker (`bfcl_eval/eval_checker/multi_turn_eval/` and `bfcl_eval/constants/executable_backend_config.py`). Logic unchanged; the only edits are import paths (`bfcl_eval...` -> `bfcl_env...`) so it runs as a standalone package. It is the task environment and grader: the agent's calls execute against it in every run (teacher traces, rehearsal traces, evaluation), and the checker turns the final state into the reward. |
| `gen_mt_tasks.py`, `gen_mt_tasks_ext.py` | ours: turn BFCL's data into Harbor task directories (base category; the other three) and write the seeded train/test split |
| `mt_split.json`, `mt_split_ext.json` | ours: those splits (150 train / 50 test per category); `../splits/split.json` merges them |
| `evaluate_template.py` | ours: the verifier copied into each task; grades the agent's recorded calls with `bfcl_env`'s checker |

`bfcl_env/` is also shipped to the agent as `../harnesses/bfcl_env_bundle.tgz` (same files),
so the agent's calls run inside the sandbox.
