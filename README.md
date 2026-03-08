# SAE4DLM

### Sparse Autoencoders for Diffusion Language Models

**SAE4DLM** is the **first Sparse Autoencoder (SAE) interpretability framework for Diffusion Language Models (DLMs)**. It provides a unified workflow for **training**, **evaluation**, **automatic interpretation**, **steering**, and **order-strategy analysis** on DLMs.

📄 **Paper:** https://arxiv.org/pdf/2602.05859

 🤗 **SAE Collection:** https://huggingface.co/collections/AwesomeInterpretability/dlm-scope

------

## Overview

Diffusion Language Models generate text through iterative denoising, which makes their internal representations and control mechanisms fundamentally different from autoregressive LLMs. **SAE4DLM** is built to study those representations directly, using SAEs as a shared interface for measurement, interpretation, and intervention.

------

## Repository Structure

```
.
├── dictionary_learning_demo/   # Train and evaluate SAEs on DLM activations
├── autointerp_hf/             # Auto-interpret DLM SAE features
├── steering/                  # Steer DLMs with SAE features
└── dlm_order/                 # Analyze DLM order/update strategies with SAE features
```

------

## What this repository supports

### 1. 🏗️ Train DLM SAEs

Train sparse autoencoders on hidden activations from DLM layers, typically `resid_post_layer_L`. This gives a sparse, interpretable feature space for studying Dream-style diffusion models.

**Example**

```
cd dictionary_learning_demo
python -u parallel_training_layers.py
```
<img width="1419" height="658" alt="a1a404a31ce30048b7a56085bdff876e" src="https://github.com/user-attachments/assets/a91888c8-f0fa-4e39-8c79-15091591b7d8" />

------

### 2. 📊 Evaluate DLM SAEs

Evaluate SAEs with both **explained variance** and **delta LM loss**, so reconstruction quality and functional faithfulness can be measured together. For DLMs, this evaluation is diffusion-aware rather than standard next-token-only evaluation. (skip examples and use held-out dataset)

**Example**

```
python eval_delta_dlm_loss.py \
  --model_name Dream-org/Dream-v0-Base-7B \
  --ae_root <DREAM_SAE_ROOT> \
  --token_budget 10000000 \
  --batch_size_text 8 \
  --max_len 2048 \
  --device cuda:0 \
  --dtype bfloat16 \
  --heldout_dataset common-pile/comma_v0.1_training_dataset \
  --skip_first_n_examples 500000 \
  --t_min 0.05 --t_max 0.50 \
  --io out
```

------

### 3. 🔍 Auto-Interpret DLM SAEs

Automatically explain SAE features by collecting activation examples and asking a judge model to summarize what each latent represents. The output is a scored set of human-readable feature explanations that can be used for analysis or steering.

**Example**

```
CUDA_VISIBLE_DEVICES=0 python -m autointerp_hf.run_dlm_eval \
  --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
  --hook_module_path auto \
  --sae_path "<DREAM_SAE_ROOT>" \
  --device "cuda" \
  --torch_dtype "bfloat16" \
  --dataset_name "common-pile/comma_v0.1_training_dataset" \
  --total_tokens 2000000 \
  --context_length 128 \
  --batch_size 64 \
  --n_latents 500 \
  --latent_batch_size 100 \
  --dead_latent_threshold 15 \
  --seed 1234 \
  --judge_model "gpt-4o-mini" \
  --judge_base_url "<OPENAI_COMPATIBLE_ENDPOINT>" \
  --judge_timeout 60 \
  --judge_max_retries 3
```

------

### 4. 🎛️ Steer DLMs with SAE features

Use selected SAE features as interpretable control directions during diffusion generation. This enables concept-level intervention in Dream and other DLMs without relying on opaque latent directions.

**Build feature file**

```
python build_features_file.py \
  --autointerp_dir <AUTOINTERP_RESULTS_DIR> \
  --out_dir <FEATURE_FILE_DIR>
```

**Run DLM steering**

```
python run_steer/dlm_steer.py \
  --model_name Dream-org/Dream-v0-Base-7B \
  --features_file <FEATURE_FILE_DIR>/features_dream_layer10_l0_80.json \
  --sae_root_dir <DREAM_SAE_ROOT> \
  --device cuda:0 \
  --amp_factor 2.0 \
  --n_prefix 5 \
  --dlm_steps 30 \
  --max_new_tokens 30 \
  --top_p 0.95 \
  --alg entropy \
  --alg_temp 0.0 \
  --token_scope all \
  --temperature 0.7 \
  --do_sample \
  --seed 42
```

**Evaluate steering**

```
python eval_steer/steering_eval.py <STEERING_RESULTS_DIR> --device cuda:0
```

------

### 5. 🧠 Study DLM order strategies with SAE features

Compare different DLM update orders, such as `origin` and `entropy`, by tracking how SAE features evolve across denoising steps. This turns decoding order into something measurable at the feature level.

**Example**

```
python dlm_order/dlm_order.py \
  --model_name Dream-org/Dream-v0-Base-7B \
  --sae_root_dir <DREAM_SAE_ROOT> \
  --sae_layers 5,14,23 \
  --sae_k 80 \
  --algs origin,entropy \
  --positions_mode update_plus_anchors \
  --steps 256 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0 \
  --sim_metric jaccard \
  --device cuda:0 \
  --dtype bf16 \
  --output_dir ./outputs/order_study
```

------

## Recommended workflow

```
Train SAEs
   ↓
Evaluate EV / ΔLM
   ↓
Run AutoInterp
   ↓
Build feature files
   ↓
Steer DLMs
   ↓
Analyze order strategies
```

------

## Notes

This repository was assembled under a tight timeline, so some parts of the codebase may still be rough or incomplete. Thank you for your understanding, and please feel free to reach out with questions, suggestions, or bug reports:

📬 **sunny615@connect.hku.hk**

------

## Acknowledgements 🧩

We thank the following repositories for their excellent work and codebases:

- **SAE Training (Dictionary Learning):** https://github.com/saprmarks/dictionary_learning
- **SAEBench:** https://github.com/adamkarvonen/SAEBench
- **Steering Score (AxBench):** https://github.com/stanfordnlp/axbench

------

## Citation

If you find this project useful, please cite our paper:

```
@inproceedings{
wang2026dlmscope,
title={{DLM}-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders},
author={Xu Wang and Bingqing Jiang and Yu Wan and Baosong Yang and Lingpeng Kong and Difan Zou},
booktitle={ICLR 2026 Workshop on Principled Design for Trustworthy AI - Interpretability, Robustness, and Safety across Modalities},
year={2026},
url={https://openreview.net/forum?id=yO5buOEUag}
}
```
