# configs/

Configuration used by the `exp` commands. Values are taken from the original job YAMLs, launch scripts, and training logs.

- `models.yaml` — base models and teacher settings, including Hugging Face IDs and chat templates.
- `benchmarks/<benchmark>.yaml` — benchmark-specific splits, harnesses, serving settings, evaluation protocol, agent definitions, SFT settings, training datasets, and RL settings where applicable.
- `experiments.yaml` — definitions for each reported experiment.

`{harness}`, `{api_base}`, `{sandbox_api_base}`, `{sandbox_host}`, and `{served_model}` are filled by `exp` when rendering configs.

Cluster-specific settings are preserved as run rather than normalised. Cluster 1 (BFCL, AppWorld) served without a reasoning parser and trained with cuDNN attention disabled (`cudnn_sdp: false`). Cluster 2 (MAB, RG) used `--reasoning-parser deepseek_r1`. Reasoning Gym sampling also varies by harness.
