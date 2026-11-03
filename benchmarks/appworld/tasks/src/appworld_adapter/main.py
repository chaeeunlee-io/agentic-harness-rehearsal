"""CLI entry point for generating AppWorld tasks in Harbor format."""

import argparse
from pathlib import Path

try:
    from .adapter import AppWorldAdapter
except ImportError:
    from adapter import AppWorldAdapter


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "datasets" / "appworld"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--splits", default=None, help="comma-separated: train,dev,test_normal,test_challenge")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--docker-image", required=True,
                        help="the task image, built with benchmarks/appworld/tasks/build_appworld_image.sh")
    args = parser.parse_args()

    adapter = AppWorldAdapter(
        args.output_dir,
        overwrite=args.overwrite,
        splits=args.splits,
        limit=args.limit,
        task_ids=args.task_ids,
        docker_image=args.docker_image,
    )
    total = adapter.run()
    print(f"Generated {total} AppWorld tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
