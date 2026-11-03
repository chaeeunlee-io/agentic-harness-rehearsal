"""Multi-harness (XH) component tests: canonicalizer, bank, dataset."""

import json

import pytest

from ..bank import ExchangeBank
from ..canonical import (
    canonicalize_atif,
    canonicalize_parity,
    cost,
    feasible_under,
)
from ..harness_dataset import HarnessGroupDataset


# ---------------------------------------------------------------- canonical
def test_parity_canonicalization():
    transcript = [
        {"role": "user", "content": "prompt"},
        {"role": "agent", "content": "GET http://localhost:8080/fhir/Observation?patient=S1&code=MG&_format=json"},
        {"role": "user", "content": "resp"},
        {"role": "agent", "content": "FINISH([2.1])"},
    ]
    acts = canonicalize_parity(transcript)
    assert acts == [
        ("GET", "http://localhost:8080/fhir/Observation?code=MG&patient=S1"),
        ("SUBMIT", ""),
    ]
    assert cost(acts) == 1


def test_atif_shell_http_recognized_and_param_order_normalized():
    steps = [{"tool_calls": [{"function": {"arguments": {
        "keystrokes": 'medagentbench_cli.py get "http://localhost:8080/fhir/Observation?code=MG&patient=S1"\n'}}}]},
        {"tool_calls": [{"function": {"arguments": {
            "keystrokes": "medagentbench_cli.py finish '[2.1]'\n"}}}]}]
    acts = canonicalize_atif(steps)
    assert acts[0] == ("GET", "http://localhost:8080/fhir/Observation?code=MG&patient=S1")
    assert acts[1] == ("SUBMIT", "")
    # same act spelled by parity == spelled by t2 shell (the lingua franca)
    parity = canonicalize_parity([
        {"role": "agent", "content": "GET http://localhost:8080/fhir/Observation?patient=S1&code=MG"}])
    assert parity[0] == acts[0]


def test_atif_non_http_shell_is_harness_action():
    steps = [{"tool_calls": [{"function": {"arguments": {"command": "python3 -c 'print(1)'"}}}]}]
    assert canonicalize_atif(steps)[0][0] == "SHELL"


def test_feasibility_types_and_budget():
    http_only = [("GET", "u"), ("SUBMIT", "")]
    with_shell = [("SHELL", "jq ..."), ("GET", "u"), ("SUBMIT", "")]
    assert feasible_under(http_only, "parity")
    assert feasible_under(http_only, "t2")
    assert not feasible_under(with_shell, "parity")     # type check
    assert feasible_under(with_shell, "t2")
    long_run = [("GET", f"u{i}") for i in range(9)] + [("SUBMIT", "")]
    assert not feasible_under(long_run, "parity")        # budget check (8)
    assert feasible_under(long_run, "t2")
    assert not feasible_under(http_only, "nope")         # unknown target: refuse


# --------------------------------------------------------------------- bank
def test_bank_roundtrip_demo_selection_and_guards(tmp_path):
    b = ExchangeBank(tmp_path / "bank.json")
    # zero-env-action success must NOT become a demo (escape-hatch guard)
    b.record("task4_1", "parity", reward=1.0, canonical_actions=[("SUBMIT", "")])
    assert b.best_demo("task4_1", "t2") is None
    # real successes: argmin cost wins, feasibility respected
    b.record("task4_1", "parity", reward=1.0,
             canonical_actions=[("GET", "u1"), ("SUBMIT", "")], trial_ref="p1")
    b.record("task4_1", "t2", reward=1.0,
             canonical_actions=[("SHELL", "x"), ("GET", "u1"), ("GET", "u2"), ("SUBMIT", "")],
             trial_ref="t1")
    demo_for_t2 = b.best_demo("task4_1", "t2")
    assert demo_for_t2["source_harness"] == "parity" and demo_for_t2["cost"] == 1
    # target=parity: the SHELL-using t2 success is infeasible; parity's own wins
    demo_for_parity = b.best_demo("task4_1", "parity")
    assert demo_for_parity["source_harness"] == "parity"
    # persistence
    b2 = ExchangeBank(tmp_path / "bank.json")
    assert b2.best_demo("task4_1", "t2")["trial"] == "p1"
    assert b2.stats("task4_1")["parity"]["n"] == 2


# ------------------------------------------------------------------ dataset
def _mk_tasks(tmp_path, names):
    root = tmp_path / "ds"
    for n in names:
        (root / n).mkdir(parents=True)
    return root


def test_dataset_rows_uids_and_kwargs(tmp_path):
    root = _mk_tasks(tmp_path, ["task1_1", "task2_1"])
    ds = HarnessGroupDataset(root, ["parity", "t2"],
                             harness_kwargs={"t2": {"temperature": 0.6}})
    assert len(ds) == 4 and ds.k == 2
    rows = [ds[i] for i in range(len(ds))]
    uids = {r["uid"] for r in rows}
    assert uids == {"task1_1|parity", "task1_1|t2", "task2_1|parity", "task2_1|t2"}
    t2_rows = [r for r in rows if r["uid"].endswith("|t2")]
    assert all(r["env_extras"]["agent_kwargs"] == {"harness": "t2", "temperature": 0.6}
               for r in t2_rows)
    # rows of one task are contiguous (super-group unit for the sampler)
    assert [r["uid"].split("|")[0] for r in rows] == ["task1_1", "task1_1", "task2_1", "task2_1"]


def test_dataset_split_and_exact_drop(tmp_path):
    root = _mk_tasks(tmp_path, ["task1_1", "task6_1", "task6_10"])
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["task1_1", "task6_1", "task6_10"]}))
    ds = HarnessGroupDataset(root, ["parity"], split_file=split,
                             drop_task_prefixes=("task6_1",))
    assert ds.tasks == ["task1_1", "task6_10"]  # exact drop must not eat task6_10


def test_dataset_refuses_empty(tmp_path):
    root = _mk_tasks(tmp_path, ["task1_1"])
    with pytest.raises(ValueError):
        HarnessGroupDataset(root, [])
