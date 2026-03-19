#!/usr/bin/env python3
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

# Dream-7B model (Base or Instruct)
MODEL_NAME = "Dream-org/Dream-v0-Base-7B"   # or "Dream-org/Dream-v0-Instruct-7B"

# Save directory
SAVE_DIR = "saes_mask"

# Whether to use Weights & Biases
USE_WANDB = True
HF_PROJECT_REPO_ID = "AwesomeInterpretability/dlm-mask-topk-sae"  # Set to None to skip pushing

# Parallel configuration: each dict represents one demo.py process to launch
# - arch: use "top_k" (or multiple architectures separated by spaces)
# - layers: list of layers for this process to train
# - device: physical GPU index (e.g., "cuda:0", "cuda:1" ...) — used only to set CUDA_VISIBLE_DEVICES
# - save_checkpoints: whether to save intermediate checkpoints during training
configurations = [
    {
        "arch": "top_k",
        "layers": [1, 5, 10],
        "device": "cuda:0",
        "save_checkpoints": False,
    },
    {
        "arch": "top_k",
        "layers": [14, 23, 27],
        "device": "cuda:1",
        "save_checkpoints": False,
    },
]

# Optional: override DLM settings from here by passing CLI flags to demo.py.
# If you leave OVERRIDE_DLM=False, demo.py will pick smart defaults per model type.
OVERRIDE_DLM = True
DLM_MASK_POLICY = "mask"   # 'unmask' | 'mask' | 'all' | 'clean'
DLM_T_MIN = 0.05
DLM_T_MAX = 0.50

# Launch gap between processes (seconds)
LAUNCH_GAP = 2

# ====================== Launch logic ======================

def main():
    os.makedirs("logs", exist_ok=True)
    print(f"[Info] Will launch {len(configurations)} training process(es).")

    procs = []
    for i, cfg in enumerate(configurations, start=1):
        # Parse architectures and layers
        archs = cfg["arch"].split()
        layers = cfg["layers"] if isinstance(cfg["layers"], (list, tuple)) else [cfg["layers"]]

        # CUDA mapping: expose only one physical GPU to the child process
        physical_gpu = cfg["device"].split(":")[1] if ":" in cfg["device"] else "0"
        local_device = "cuda:0"

        # Build the command
        cmd = [
            "python", "-u", "demo.py",
            "--save_dir", SAVE_DIR,
            "--model_name", MODEL_NAME,
            "--architectures", *archs,
            "--layers", *[str(l) for l in layers],
            "--device", local_device,
            "--dlm_debug_print",
            "--dlm_debug_tokens",
            "--dlm_debug_every", "50",

        ]
        if USE_WANDB:
            cmd.append("--use_wandb")
        if cfg.get("save_checkpoints", False):
            cmd.append("--save_checkpoints")
        if HF_PROJECT_REPO_ID:
            cmd += ["--hf_repo_id", HF_PROJECT_REPO_ID]

        # Optionally pass DLM args; otherwise let demo.py choose defaults
        if OVERRIDE_DLM:
            cmd += [
                "--dlm_mask_policy", str(DLM_MASK_POLICY),
                "--dlm_t_min", str(DLM_T_MIN),
                "--dlm_t_max", str(DLM_T_MAX),
            ]

        # Environment variables: restrict to the selected physical GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        # Dream uses remote code; avoid interactive confirmations
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

        # Log file name
        layers_slug = "-".join(str(l) for l in layers)
        arch_slug = "_".join(archs)
        device_slug = cfg["device"].replace(":", "_")
        log_file = f"logs/{arch_slug}_layers_{layers_slug}_{device_slug}.out"

        print(f"[Launch {i}/{len(configurations)}] "
              f"CUDA_VISIBLE_DEVICES={physical_gpu} "
              f"{' '.join(shlex.quote(x) for x in cmd)}")
        print(f"[Log] {log_file}")

        # Start the subprocess and write logs
        f = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append((p, f))

        time.sleep(LAUNCH_GAP)

    print("[Info] All jobs submitted!")

    # Optional: wait for all subprocesses to finish
    for p, f in procs:
        p.wait()
        f.close()
    print("[Info] All jobs finished.")


if __name__ == "__main__":
    main()
