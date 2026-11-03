"""Merge SkyRL FSDP2 checkpoint shards into a Hugging Face safetensors export.

Usage:
    torchrun --nproc_per_node=2 merge_fsdp2_to_hf.py \
        --ckpt-policy-dir .../ckpts/global_step_12/policy \
        --out .../exports/global_step_12/policy

SkyRL's final Hugging Face export failed with OOM, while the native per-rank
checkpoints contain the same weights as DTensors. This script gathers them on
CPU with gloo (no GPU needed) and writes a standard Hugging Face export for vLLM
evaluation.
"""

import argparse
import os
import shutil

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-policy-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    shard_path = os.path.join(
        args.ckpt_policy_dir, f"model_world_size_{world}_rank_{rank}.pt"
    )
    state = torch.load(shard_path, map_location="cpu", weights_only=False)
    print(f"[rank-{rank}] loaded {shard_path}: {len(state)} keys", flush=True)

    full = {}
    for key in sorted(state.keys()):
        val = state[key]
        if hasattr(val, "full_tensor"):
            tensor = val.full_tensor()
        else:
            tensor = val
        if rank == 0:
            full[key] = tensor.detach().cpu().contiguous()
        del val
    state.clear()
    dist.barrier()

    if rank == 0:
        hf_dir = os.path.join(args.ckpt_policy_dir, "huggingface")
        tmp = args.out + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(hf_dir)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        model.save_pretrained(tmp, state_dict=full, safe_serialization=True)
        config.save_pretrained(tmp)
        AutoTokenizer.from_pretrained(hf_dir).save_pretrained(tmp)
        for extra in ("chat_template.jinja", "generation_config.json"):
            src = os.path.join(hf_dir, extra)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, extra))
        if os.path.isdir(args.out):
            shutil.rmtree(args.out)  # empty/stale dir from the crashed export
        os.rename(tmp, args.out)
        n_params = sum(t.numel() for t in full.values())
        print(f"[rank-0] wrote {args.out}: {len(full)} tensors, {n_params/1e9:.2f}B params", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
