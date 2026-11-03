"""Train with TRL SFT on rejection-sampled Harbor traces.

Input is a chat-format JSONL with `messages` and `meta` fields: a shipped pool,
or one built with `exp build-pool`. `SFTTrainer` reads `messages` directly and
applies the chat template (by default `training/templates/qwen3_acc_thinking.jinja2`;
`--chat-template ''` uses the model's own).

Training uses full-sequence loss by default. `--assistant-only` requires chat
templates with generation markers, which the shipped Qwen3 template lacks.

Normally run through `uv run exp sft`, which passes every argument. Directly:

Smoke:
    python sft_train.py --data smoke_train.jsonl --out runs/sft-smoke --max-steps 10

Train:
    python sft_train.py --data train.jsonl --epochs 2 --out .../rsft_qwen3_4b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True, help="output folder (exp sft: runs/checkpoints/<benchmark>/<model>/<pool>)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--assistant-only", action="store_true")
    ap.add_argument("--optim", default="adamw_torch",
                    help="e.g. paged_adamw_8bit (bitsandbytes) — halves optimizer memory; "
                    "needed alongside --liger for >16k max-len full finetunes on 80GB")
    ap.add_argument("--liger", action="store_true",
                    help="fused linear+CE (never materializes the seq_len x vocab logits tensor); "
                    "needed for 16k-token full finetunes on one 80GB card")
    ap.add_argument("--wandb-run", default=None, help="wandb run name (project harbor); omit = no wandb")
    ap.add_argument(
        "--chat-template",
        default=str(Path(__file__).parent / "templates" / "qwen3_acc_thinking.jinja2"),
        help="Jinja chat template file. Default: the accumulate-thinking Qwen3 "
        "template — Qwen3's SHIPPED template strips <think> from non-final "
        "turns, which would silently drop distilled reasoning. '' = model default.",
    )
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("messages",)])
    print(f"dataset: {len(ds)} examples")

    cfg = SFTConfig(
        output_dir=args.out,
        max_length=args.max_len,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to=(["wandb"] if args.wandb_run else []),
        run_name=args.wandb_run,
        assistant_only_loss=args.assistant_only,
        use_liger_kernel=args.liger,
        optim=args.optim,
    )
    processing_class = None
    if args.chat_template:
        from transformers import AutoTokenizer

        processing_class = AutoTokenizer.from_pretrained(args.model)
        processing_class.chat_template = Path(args.chat_template).read_text()
        print(f"chat template overridden from {args.chat_template}")

    trainer = SFTTrainer(
        model=args.model, args=cfg, train_dataset=ds, processing_class=processing_class
    )
    trainer.train()
    trainer.save_model(args.out + "/final")
    copy_processor_configs(args.model, args.out + "/final")
    print(f"SFT-DONE -> {args.out}/final")
    return 0


PROCESSOR_CONFIGS = ("preprocessor_config.json", "processor_config.json")


def copy_processor_configs(model: str, out: str) -> None:
    """Copy the base model's processor configs into the checkpoint.

    With a chat-template override the trainer saves a bare tokenizer, so a multimodal
    model (Gemma 3) comes out without its image-processor configs, and vLLM will not
    serve it without them, even for text-only use. The as-run Gemma trainings copied
    the same two files after every run. Models without these files (Qwen3) are untouched.
    """
    import shutil

    for name in PROCESSOR_CONFIGS:
        dst = Path(out) / name
        if dst.exists():
            continue
        src = Path(model) / name
        if not Path(model).is_dir():  # a hub id rather than a local copy
            try:
                from huggingface_hub import hf_hub_download

                src = Path(hf_hub_download(model, name))
            except Exception:
                continue
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"copied {name} from {model}")


if __name__ == "__main__":
    raise SystemExit(main())
