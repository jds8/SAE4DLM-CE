#!/usr/bin/env python3
import subprocess
import time
import os
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

# Configuration for A800*2
# For our current implementation, relative training speed is:
# standard / p_anneal > top_k > batch_top_k > jump_relu > gated
# So, we have jump_relu and gated on their own GPUs

# Set the Dream-7B model name (Dream v0 Base 7B or Instruct)
MODEL_NAME = "Dream-org/Dream-v0-Base-7B"  # or "Dream-org/Dream-v0-Instruct-7B"

# Based on model name, adjust the number of layers to use
if "Dream" in MODEL_NAME:
    layer = 10  # Dream-7B (Base and Instruct) uses layer 10, update as needed
else:
    raise ValueError("Unknown model name")

# Configuration settings for each GPU
configurations = [
    {
        "arch": "batch_top_k jump_relu",
        "layers": layer,
        "device": "cuda:0",
        "save_checkpoints": False
    },
    {
        "arch": "gated top_k",
        "layers": layer,
        "device": "cuda:1",
        "save_checkpoints": False
    },
]

SAVE_DIR = "saes"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Launch jobs
for i, config in enumerate(configurations):
    log_file = f"logs/{(config['arch'].replace(' ', '_'))}_l{config['layers']}_{config['device'].replace(':', '_')}.out"
    
    if config["save_checkpoints"]:
        save_command = "--save_checkpoints"
    else:
        save_command = ""

    gpu_index = config["device"].split(":")[1]
    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu_index} "
    cmd = [
        "python", "demo.py",
        "--save_dir", SAVE_DIR,
        "--use_wandb", 
        "--hf_repo_id", "AwesomeInterpretability/dlm-topk-sae",  # Update HuggingFace repo ID accordingly
        "--model_name", MODEL_NAME,
        "--architectures", config["arch"],
        "--layers", str(config["layers"]),
        "--device", config["device"]
    ]

    print(" ".join(cmd))
    
    # Launch with nohup (run in background)
    with open(log_file, "w") as f:
        subprocess.Popen(
            f"{env_prefix}nohup {' '.join(cmd)} > {log_file} 2>&1",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    print(f"Started job {i + 1}/{len(configurations)}: {config['arch']} with layers: {config['layers']}")
    time.sleep(2)

print("All jobs submitted!")
