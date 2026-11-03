"""Generate Reasoning Gym rehearsal tasks.

Each rehearsal task gives the exact answer and only requires the agent to write
it to `/workspace/answer.txt` through the harness. The original verifier is
unchanged.

Samples 100 easy training tasks with seed 0. Tasks without an answer or with
answers longer than 2000 characters are skipped.

Usage:
    python gen_rehearsal.py   # -> runs/tasks/reasoning_gym/rehearsal-t2/
"""
import argparse
import json
import random
import shutil
from pathlib import Path

TASKS = Path(__file__).resolve().parents[3] / "runs" / "tasks" / "reasoning_gym"
N = 100

INSTRUCTION = """Transcription drill: your only task is to write an exact string to a file.

Write EXACTLY the following content (between the BEGIN/END markers, not including
the marker lines themselves) to the file "/workspace/answer.txt":

BEGIN
{answer}
END

Preserve every character, including signs, decimals, spaces and line breaks.
Use careful shell quoting (e.g. cat with a quoted heredoc) so nothing is
interpreted by the shell. Verify the file content matches before finishing.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-tasks", type=Path, default=TASKS / "train-easy",
                   help="easy training tasks (exp tasks -b reasoning_gym --split train-easy)")
    p.add_argument("--out", type=Path, default=TASKS / "rehearsal-t2")
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; remove it to regenerate")

    candidates = sorted(d.name for d in args.train_tasks.iterdir() if (d / "tests" / "entry_data.json").is_file())
    picked = random.Random(0).sample(candidates, min(N, len(candidates)))
    args.out.mkdir(parents=True)
    n = 0
    for name in picked:
        try:
            answer = json.load(open(args.train_tasks / name / "tests" / "entry_data.json"))["answer"]
        except Exception:
            continue
        if answer is None or (isinstance(answer, str) and len(answer) > 2000):
            continue
        shutil.copytree(args.train_tasks / name, args.out / name)
        text = answer if isinstance(answer, str) else json.dumps(answer)
        (args.out / name / "instruction.md").write_text(INSTRUCTION.format(answer=text))
        n += 1
    print(f"wrote {n} rehearsal tasks to {args.out}")


if __name__ == "__main__":
    main()
