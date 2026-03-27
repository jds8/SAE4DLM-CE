#!/usr/bin/env python3
"""
Experiment 5 (Issue #18): Train diffusion-stage specific SAEs.

Trains 6 SAEs total:
  - Layers: L1, L5
  - Stages: late (t=0.05-0.2), middle (t=0.2-0.35), early (t=0.35-0.5)
  - Architecture: top_k, sparsity k=50

Each SAE is trained only on activations from its designated diffusion stage.
This is done by restricting t_min/t_max to the stage's time range.

NOTE: Before running, set TARGET_L0s = [50] in demo_config.py to train only
      the k=50 sparsity level required by Experiment 5.
      (Or run as-is to train all sparsity levels and filter k=50 during eval.)
"""

import subprocess
import time
import os
import shlex
import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

# ====================== User-configurable section ======================

MODEL_NAME = "Dream-org/Dream-v0-Base-7B"  # or "Dream-org/Dream-v0-Instruct-7B"

# Save directory for stage-specific SAEs
SAVE_DIR = "saes_exp5_stage_specific"

USE_WANDB = False
HF_PROJECT_REPO_ID = None  # Set to a HF repo ID to push results

# Diffusion stage definitions (from Issue #18)
STAGES = {
    "late":  {"t_min": 0.05, "t_max": 0.20},
    # "middle": {"t_min": 0.20, "t_max": 0.35},
    "early":   {"t_min": 0.35, "t_max": 0.50},
}

# Layers to train (Issue #18 specifies L1 and L5)
# LAYERS = [1, 5]
LAYERS = [10]

# Architecture and mask policy
ARCH = "top_k"
DLM_MASK_POLICY = "mask"

# GPU assignment: map (layer, stage) -> physical GPU index.
# Adjust to match your available hardware.
# With 2 GPUs: L1 stages on GPU 0, L5 stages on GPU 1.
GPU_ASSIGNMENTS = {
    (1, "early"):  "cuda:4",
    (1, "middle"): "cuda:2",
    (1, "late"):   "cuda:0",
    (5, "early"):  "cuda:5",
    (5, "middle"): "cuda:3",
    (5, "late"):   "cuda:1",
    (10, "early"): "cuda:6",
    (10, "late"): "cuda:7",
}

# Launch gap between processes (seconds)
LAUNCH_GAP = 5

# ====================== Launch logic ======================


def main():
    os.makedirs("logs", exist_ok=True)

    configurations = []
    for layer in LAYERS:
        for stage_name, stage_range in STAGES.items():
            device = GPU_ASSIGNMENTS.get((layer, stage_name), "cuda:0")
            configurations.append({
                "layer": layer,
                "stage": stage_name,
                "t_min": stage_range["t_min"],
                "t_max": stage_range["t_max"],
                "device": device,
            })

    print(f"[Info] Will launch {len(configurations)} training process(es) "
          f"(layers={LAYERS}, stages={list(STAGES.keys())}).")

    procs = []
    for i, cfg in enumerate(configurations, start=1):
        layer = cfg["layer"]
        stage = cfg["stage"]
        t_min = cfg["t_min"]
        t_max = cfg["t_max"]
        device = cfg["device"]

        physical_gpu = device.split(":")[1] if ":" in device else "0"
        #local_device = "cuda:0"
        local_device = device
            
        # Save directory encodes both stage and layer for easy identification
        stage_save_dir = f"{SAVE_DIR}_{stage}"

        print('parallel: ', device)
        cmd = [
            "python", "-u", "demo.py",
            "--save_dir", stage_save_dir,
            "--model_name", MODEL_NAME,
            "--architectures", ARCH,
            "--layers", str(layer),
            "--device", local_device,
            "--dlm_mask_policy", DLM_MASK_POLICY,
            "--dlm_t_min", str(t_min),
            "--dlm_t_max", str(t_max),
            "--dlm_debug_print",
            "--dlm_debug_every", "50",
            "--save_checkpoints",
            "--device", device,
        ]

        if USE_WANDB:
            cmd.append("--use_wandb")
        if HF_PROJECT_REPO_ID:
            cmd += ["--hf_repo_id", HF_PROJECT_REPO_ID]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

        log_file = f"logs/exp5_{ARCH}_layer{layer}_{stage}.out"

        print(f"[Launch {i}/{len(configurations)}] "
              f"layer={layer} stage={stage} t=[{t_min},{t_max}] "
              f"CUDA_VISIBLE_DEVICES={physical_gpu}")
        print(f"  cmd: {' '.join(shlex.quote(x) for x in cmd)}")
        print(f"  log: {log_file}")

        f = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append((p, f, layer, stage))

        time.sleep(LAUNCH_GAP)

    print("[Info] All jobs submitted!")

    for p, f, layer, stage in procs:
        p.wait()
        f.close()
        rc = p.returncode
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[Done] layer={layer} stage={stage}: {status}")

    print("[Info] All jobs finished.")
    print(f"[Info] Stage-specific SAEs saved under: {SAVE_DIR}_<stage>/")


if __name__ == "__main__":
    main()
