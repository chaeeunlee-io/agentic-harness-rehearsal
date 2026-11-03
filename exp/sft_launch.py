"""Run `training/sft_train.py`, optionally with cuDNN SDPA disabled.

With `--disable-cudnn-sdp`, disables cuDNN SDPA before launching the trainer.
Cluster 1 (BFCL, AppWorld) used this setting; cluster 2 (MAB, RG) ran the
trainer directly.
"""
import runpy
import sys
from pathlib import Path

TRAINER = Path(__file__).resolve().parent.parent / "training" / "sft_train.py"


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--disable-cudnn-sdp":
        import torch

        torch.backends.cuda.enable_cudnn_sdp(False)
        args = args[1:]
    if args and args[0] == "--":
        args = args[1:]
    sys.argv = [str(TRAINER), *args]
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
