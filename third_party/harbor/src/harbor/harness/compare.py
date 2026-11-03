"""Manual sanity-check tool for harness arms: readable side-by-side comparisons.

Two subcommands, matching the verification plan's Level 0 and Level 1:

  variants  compare task DIRECTORIES (native vs knobbed arm) — what the offline
            generator changed, and nothing else.
  trials    compare two TRIAL result directories (native vs knobbed run) — where
            the transcripts diverge, plus the middleman's per-request audit log.

Design goals: stdlib only (runs as `python3 src/harbor/harness/compare.py ...` on a
bare checkout); output is plain text built for eyeballs — one header block, a file
table, focused diffs, a turn-by-turn table with the first divergence flagged.

Extending to a new task later = registering one extractor: a function that turns
that agent's trial dir into a TrialView (see EXTRACTORS at the bottom of the
trial section). Everything above the extractor is task-agnostic.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

WIDTH = 100
_COLOR = False


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _dim(t: str) -> str:
    return _c("2", t)


def _bold(t: str) -> str:
    return _c("1", t)


def _rule(title: str = "") -> str:
    if not title:
        return "═" * WIDTH
    pad = "═" * max(0, WIDTH - len(title) - 6)
    return f"════ {_bold(title)} {pad}"


def _sub(title: str) -> str:
    return f"\n  ── {_bold(title)} " + "─" * max(0, WIDTH - len(title) - 8)


def _clip(text: str, limit: int) -> str:
    text = text.replace("\n", "⏎")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ─────────────────────────────── Level 0: variants ───────────────────────────────

_TEXT_SUFFIXES = {".md", ".toml", ".json", ".txt", ".sh", ".py", ".yaml", ".yml"}


def _walk_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}


def compare_task_dirs(dir_a: Path, dir_b: Path, *, max_diff_lines: int = 40) -> bool:
    """Print a readable file-level comparison. Returns True iff dirs differ."""
    files_a, files_b = _walk_files(dir_a), _walk_files(dir_b)
    all_paths = sorted(set(files_a) | set(files_b))

    identical: list[str] = []
    changed: list[str] = []
    only_a: list[str] = []
    only_b: list[str] = []
    for rel in all_paths:
        if rel not in files_b:
            only_a.append(rel)
        elif rel not in files_a:
            only_b.append(rel)
        elif files_a[rel].read_bytes() == files_b[rel].read_bytes():
            identical.append(rel)
        else:
            changed.append(rel)

    print(f"\n  FILES  ({len(all_paths)} compared)")
    print(f"  {_green('✓')} {len(identical)} identical")
    for rel in changed:
        print(f"  {_yellow('~')} {rel:<50} changed")
    for rel in only_a:
        print(f"  {_red('-')} {rel:<50} only in A ({dir_a.name})")
    for rel in only_b:
        print(f"  {_green('+')} {rel:<50} only in B ({dir_b.name})")
    if not (changed or only_a or only_b):
        print(f"  {_green('→ directories are byte-identical')}")
        return False

    for rel in changed:
        if Path(rel).suffix not in _TEXT_SUFFIXES:
            print(_sub(f"{rel} · changed (binary — no diff shown)"))
            continue
        print(_sub(f"{rel} · diff (A=−  B=+)"))
        a_lines = files_a[rel].read_text(errors="replace").splitlines()
        b_lines = files_b[rel].read_text(errors="replace").splitlines()
        shown = 0
        for line in difflib.unified_diff(a_lines, b_lines, lineterm="", n=1):
            if line.startswith(("---", "+++")):
                continue
            if shown >= max_diff_lines:
                print(_dim("    … (diff truncated; run plain `diff` for the rest)"))
                break
            if line.startswith("-"):
                print("  " + _red(_clip(line, WIDTH - 2)))
            elif line.startswith("+"):
                print("  " + _green(_clip(line, WIDTH - 2)))
            elif line.startswith("@@"):
                print("  " + _dim(line))
            else:
                print("  " + _clip(line, WIDTH - 2))
            shown += 1

    for rel in only_b:
        path = files_b[rel]
        if Path(rel).suffix in _TEXT_SUFFIXES and path.stat().st_size < 2000:
            print(_sub(f"{rel} · new file content"))
            for line in path.read_text(errors="replace").splitlines():
                print("  " + _green(_clip(line, WIDTH - 2)))
    return True


def cmd_variants(args: argparse.Namespace) -> int:
    dir_a, dir_b = Path(args.a), Path(args.b)

    def is_task_dir(p: Path) -> bool:
        return (p / "instruction.md").exists()

    if is_task_dir(dir_a) and is_task_dir(dir_b):
        print(_rule(f"VARIANT DIFF  A={dir_a.name}  B={dir_b.name}"))
        print(f"  A: {dir_a}\n  B: {dir_b}")
        compare_task_dirs(dir_a, dir_b)
        return 0

    # Arm roots: iterate shared task subdirectories.
    tasks_a = {p.name for p in dir_a.iterdir() if is_task_dir(p)}
    tasks_b = {p.name for p in dir_b.iterdir() if is_task_dir(p)}
    shared = sorted(tasks_a & tasks_b)
    if not shared:
        print("No task directories in common — pass task dirs or arm roots.")
        return 1
    print(_rule(f"ARM DIFF  A={dir_a.name}  B={dir_b.name}  ({len(shared)} tasks)"))
    if args.task:
        print(f"  A: {dir_a}\n  B: {dir_b}")
        compare_task_dirs(dir_a / args.task, dir_b / args.task)
        return 0
    differing = 0
    for task in shared:
        files_a = _walk_files(dir_a / task)
        files_b = _walk_files(dir_b / task)
        n_changed = sum(
            1
            for rel in set(files_a) & set(files_b)
            if files_a[rel].read_bytes() != files_b[rel].read_bytes()
        )
        n_extra = len(set(files_a) ^ set(files_b))
        if n_changed or n_extra:
            differing += 1
            print(
                f"  {_yellow('~')} {task:<24} {n_changed} changed, "
                f"{n_extra} added/removed"
            )
    print(
        f"\n  {differing}/{len(shared)} tasks differ. Detail one with:  --task <name>"
    )
    a_only_tasks = sorted(tasks_a - tasks_b)
    if a_only_tasks:
        print(_dim(f"  tasks only in A: {a_only_tasks[:5]} …"))
    return 0


# ─────────────────────────────── Level 1: trials ───────────────────────────────


@dataclass
class TrialView:
    """Task-agnostic view of one trial, produced by a per-agent extractor."""

    trial_dir: Path
    agent: str = "?"
    arm_name: str = "?"
    arm_knobs: dict = field(default_factory=dict)
    reward: float | None = None
    status: str = "?"
    exception: str | None = None
    prompt: str = ""
    turns: list[tuple[str, str]] = field(default_factory=list)  # (role, text)
    harness_log: list[dict] = field(default_factory=list)


def _load_result_json(view: TrialView) -> None:
    result_path = view.trial_dir / "result.json"
    if not result_path.exists():
        return
    data = json.loads(result_path.read_text(errors="replace"))
    verifier = data.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    if "reward" in rewards:
        view.reward = rewards["reward"]
    exc = data.get("exception_info") or {}
    view.exception = exc.get("exception_type")
    agent_result = data.get("agent_result") or {}
    meta = agent_result.get("metadata") or {}
    view.status = meta.get("status", view.status)
    view.arm_name = meta.get("harness_arm", view.arm_name)
    view.arm_knobs = meta.get("harness_knobs", view.arm_knobs)


def _extract_medagentbench(view: TrialView) -> bool:
    """Extractor for medagentbench-parity-agent debug logs."""
    matches = list(view.trial_dir.rglob("medagentbench-parity-agent.json"))
    if not matches:
        return False
    data = json.loads(matches[0].read_text(errors="replace"))
    view.agent = "medagentbench-parity-agent"
    view.status = data.get("status", view.status)
    view.harness_log = data.get("harness_log", [])
    arm = data.get("harness_arm")
    if isinstance(arm, dict):
        view.arm_name = arm.get("name", view.arm_name)
        view.arm_knobs = {k: v for k, v in arm.items() if k != "name"}
    transcript = data.get("transcript", [])
    if transcript:
        view.prompt = transcript[0].get("content", "")
        view.turns = [
            (t.get("role", "?"), t.get("content", "")) for t in transcript[1:]
        ]
    return True


# Extend to a new task by appending (name, extractor) here. An extractor fills a
# TrialView from that agent's on-disk logs and returns True if it recognised them.
EXTRACTORS = [
    ("medagentbench", _extract_medagentbench),
]


def load_trial(trial_dir: Path) -> TrialView:
    view = TrialView(trial_dir=trial_dir)
    _load_result_json(view)
    for _name, extractor in EXTRACTORS:
        if extractor(view):
            break
    return view


def _summarise_turn(role: str, text: str, limit: int) -> str:
    first = text.strip().splitlines()[0] if text.strip() else "(empty)"
    if role == "user":
        kind = "reply"
        if text.startswith("Here is the response from the GET"):
            kind = f"GET reply · {len(text):,} chars"
        elif "POST request accepted" in text:
            kind = "POST ack"
        elif text.startswith("Error"):
            kind = "error reply"
        return _dim(_clip(f"← {kind}: {first}", limit))
    return _clip(f"→ {first}", limit)


def cmd_trials(args: argparse.Namespace) -> int:
    view_a, view_b = load_trial(Path(args.a)), load_trial(Path(args.b))
    print(_rule("TRIAL DIFF"))
    half = (WIDTH - 6) // 2
    print(f"  A: {view_a.trial_dir}")
    print(f"  B: {view_b.trial_dir}")

    def fmt_arm(v: TrialView) -> str:
        knobs = f" {v.arm_knobs}" if v.arm_knobs else ""
        return f"{v.arm_name}{knobs}"

    rows = [
        ("agent", view_a.agent, view_b.agent),
        ("arm", fmt_arm(view_a), fmt_arm(view_b)),
        ("reward", str(view_a.reward), str(view_b.reward)),
        ("status", view_a.status, view_b.status),
        ("exception", str(view_a.exception), str(view_b.exception)),
    ]
    print()
    for label, a_val, b_val in rows:
        marker = "  " if a_val == b_val else _yellow("≠ ")
        print(
            f"  {marker}{label:<10} {_clip(a_val, half):<{half}}  {_clip(b_val, half)}"
        )

    print(_sub("PROMPT"))
    if view_a.prompt == view_b.prompt:
        print(f"  {_green('identical')} ({len(view_a.prompt):,} chars)")
    else:
        a_lines = view_a.prompt.splitlines()
        b_lines = view_b.prompt.splitlines()
        shown = 0
        for line in difflib.unified_diff(a_lines, b_lines, lineterm="", n=0):
            if line.startswith(("---", "+++", "@@")):
                continue
            if shown >= args.max_lines:
                print(_dim("  … (more; rerun with --max-lines N)"))
                break
            colour = _red if line.startswith("-") else _green
            print("  " + colour(_clip(line, WIDTH - 2)))
            shown += 1

    print(_sub("TURNS  (A left · B right)"))
    n_turns = max(len(view_a.turns), len(view_b.turns))
    divergence_flagged = False
    for i in range(n_turns):
        role_a, text_a = view_a.turns[i] if i < len(view_a.turns) else ("-", "(absent)")
        role_b, text_b = view_b.turns[i] if i < len(view_b.turns) else ("-", "(absent)")
        cell_a = _summarise_turn(role_a, text_a, half)
        cell_b = _summarise_turn(role_b, text_b, half)
        flag = ""
        if not divergence_flagged and (role_a, text_a) != (role_b, text_b):
            flag = _yellow(" ◀ FIRST DIVERGENCE")
            divergence_flagged = True
            first_divergence = i
        print(f"  {i + 1:>2} {cell_a:<{half}}  {cell_b}{flag}")
    if not divergence_flagged:
        print(f"  {_green('→ transcripts are identical')}")
    elif args.full:
        i = first_divergence
        print(_sub(f"DIVERGING TURN {i + 1} · full text"))
        for label, view in (("A", view_a), ("B", view_b)):
            text = view.turns[i][1] if i < len(view.turns) else "(absent)"
            print(f"  [{label}] {view.arm_name}:")
            for line in text.splitlines()[: args.max_lines]:
                print("    " + _clip(line, WIDTH - 4))
            print()

    for label, view in (("A", view_a), ("B", view_b)):
        if view.harness_log:
            print(_sub(f"HARNESS LOG ({label} · {view.arm_name}) — what the shim did"))
            for row in view.harness_log:
                edited = row.get("model_url") != row.get("forwarded_url")
                mark = _yellow("✎") if edited else _dim("=")
                syn = (
                    _red(" SYNTHETIC (server not called)")
                    if row.get("synthetic")
                    else ""
                )
                print(
                    f"  {mark} turn {row.get('turn', '?')}: {_clip(row.get('model_url', ''), WIDTH - 14)}"
                )
                if edited:
                    print(f"      → {_clip(row.get('forwarded_url', ''), WIDTH - 8)}")
                chars_in, chars_out = row.get("chars_in"), row.get("chars_out")
                if chars_in is not None and chars_in != chars_out:
                    print(_dim(f"      reply {chars_in:,} → {chars_out:,} chars{syn}"))
                elif syn:
                    print(f"     {syn}")
    return 0


def main() -> int:
    global _COLOR, WIDTH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", action="store_true", help="force ANSI colours")
    parser.add_argument("--width", type=int, default=WIDTH)
    sub = parser.add_subparsers(dest="command", required=True)

    p_var = sub.add_parser("variants", help="Level 0: diff task dirs / arm trees")
    p_var.add_argument("a", help="native task dir or arm root")
    p_var.add_argument("b", help="knobbed task dir or arm root")
    p_var.add_argument("--task", help="detail one task when comparing arm roots")
    p_var.set_defaults(func=cmd_variants)

    p_tri = sub.add_parser("trials", help="Level 1: diff two trial result dirs")
    p_tri.add_argument("a", help="trial dir (e.g. the native run)")
    p_tri.add_argument("b", help="trial dir (the knobbed run)")
    p_tri.add_argument(
        "--full", action="store_true", help="print the diverging turn in full"
    )
    p_tri.add_argument("--max-lines", type=int, default=30)
    p_tri.set_defaults(func=cmd_trials)

    args = parser.parse_args()
    _COLOR = args.color or sys.stdout.isatty()
    WIDTH = args.width
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
