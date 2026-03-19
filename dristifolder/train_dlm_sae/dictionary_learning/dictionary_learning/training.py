"""
Training dictionaries (SAEs) with generic trainers.

This file is agnostic to AR vs. DLM (e.g., Dream-7B). It consumes activations
produced by ActivationBuffer (which now optionally applies DLM forward noising)
and updates trainers accordingly.
"""

import json
import torch.multiprocessing as mp
import os
from queue import Empty
from typing import Optional
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch as t
from tqdm import tqdm
import wandb

from .dictionary import AutoEncoder  # noqa: F401
from .trainers.standard import StandardTrainer  # keep import to ensure availability in configs


# ----------------------------- Utilities ----------------------------- #

def _to_jsonable(obj):
    """Recursively convert objects (torch.device, torch.dtype, Path, Tensor, numpy arrays...) to JSONable types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, t.device):
        return str(obj)
    if isinstance(obj, t.dtype):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, t.Tensor):
        try:
            return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
        except Exception:
            return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        return str(obj)
    except Exception:
        return "<non-serializable>"


def _canonical_device_str(dev: str | t.device, fallback: str = "cuda") -> str:
    """
    Make sure a device string is valid on this machine. If not, fallback gracefully.
    """
    if isinstance(dev, t.device):
        dev = str(dev)

    try:
        if isinstance(dev, str) and dev.startswith("cuda"):
            if not t.cuda.is_available():
                return "cpu"
            if ":" in dev:
                try:
                    idx = int(dev.split(":")[1])
                    if idx < 0 or idx >= t.cuda.device_count():
                        return "cuda"
                except Exception:
                    return "cuda"
            return dev
        return dev
    except Exception:
        return "cuda" if t.cuda.is_available() else "cpu"


# --------------------------- WandB process --------------------------- #

def new_wandb_process(config, log_queue, entity, project):
    """Isolated WandB process to avoid CUDA/WandB multiprocessing issues."""
    wandb.init(entity=entity or None, project=project or None, config=config, name=config.get("wandb_name", None))
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
        except Empty:
            continue
    wandb.finish()


# --------------------------- Logging helpers ------------------------- #

def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list = [],
    verbose: bool = False,
):
    """
    Compute and (optionally) log per-trainer scalar metrics on the current activation batch.
    """
    with t.no_grad():
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act_i = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act_i = act_i[..., i, :]

            if not transcoder:
                _x_in, act_hat, f, losslog = trainer.loss(act_i, step=step, logging=True)

                l0 = (f != 0).float().sum(dim=-1).mean().item()
                total_variance = t.var(act_i, dim=0).sum()
                residual_variance = t.var(act_i - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log["frac_variance_explained"] = float(frac_variance_explained)
            else:
                x, x_hat, f, losslog = trainer.loss(act_i, step=step, logging=True)
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            if verbose:
                if not transcoder:
                    print(f"Step {step}: L0 = {l0:.3f}, frac_variance_explained = {float(frac_variance_explained):.6f}")
                else:
                    print(f"Step {step}: L0 = {l0:.3f}")

            for k, v in losslog.items():
                log[k] = v.cpu().item() if isinstance(v, t.Tensor) else v
            log["l0"] = l0

            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.detach().cpu().item()
                log[name] = value

            if log_queues:
                log_queues[i].put(log)


# --------------------------- Norm estimation ------------------------- #

def get_norm_factor(data, steps: int) -> float:
    """
    Estimate a scalar so activation vectors have unit mean squared norm.
    """
    total_mean_squared_norm = 0.0
    count = 0

    for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step > steps:
            break
        count += 1
        mean_squared_norm = t.mean(t.sum(act_BD ** 2, dim=1))
        total_mean_squared_norm += float(mean_squared_norm)

    average_mean_squared_norm = total_mean_squared_norm / max(count, 1)
    norm_factor = float(t.sqrt(t.tensor(average_mean_squared_norm)).item())

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")

    return norm_factor


# ----------------------------- Main train ---------------------------- #

def trainSAE(
    data,
    trainer_configs: list[dict],
    steps: int,
    use_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = "",
    save_steps: Optional[list[int]] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    activations_split_by_head: bool = False,
    transcoder: bool = False,
    run_cfg: dict = {},
    normalize_activations: bool = False,
    verbose: bool = False,
    device: str = "cuda",
    autocast_dtype: t.dtype = t.float32,
    backup_steps: Optional[int] = None,
):
    """
    Train SAEs using the given trainers.
    Activation source (clean vs. DLM-forward-noised) is decided by ActivationBuffer.
    """

    # Normalize provided device
    device = _canonical_device_str(device)

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)

    # Build trainers from configs
    trainers = []
    for i, config in enumerate(trainer_configs):
        if "device" in config:
            config["device"] = _canonical_device_str(config["device"])
        else:
            config["device"] = device

        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"

        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    # WandB setup
    wandb_processes = []
    log_queues = []

    if use_wandb:
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            safe_cfg = _to_jsonable(wandb_config)

            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(safe_cfg, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # Prepare save directories and export trainer/buffer config
    if save_dir is not None:
        save_dirs = [os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))]
        for trainer, dir_path in zip(trainers, save_dirs):
            os.makedirs(dir_path, exist_ok=True)
            cfg = {"trainer": _to_jsonable(trainer.config)}
            try:
                # ActivationBuffer now exposes dlm_mask_policy/t_min/t_max in .config
                cfg["buffer"] = _to_jsonable(data.config)
            except Exception:
                pass
            with open(os.path.join(dir_path, "config.json"), "w") as f:
                json.dump(cfg, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    # Norm (optional)
    norm_factor = 1.0
    if normalize_activations:
        norm_factor = get_norm_factor(data, steps=100)
        for trainer in trainers:
            trainer.config["norm_factor"] = norm_factor
            trainer.ae.scale_biases(1.0)

    # Main loop
    for step, act in enumerate(tqdm(data, total=steps)):
        act = act.to(device=device, dtype=autocast_dtype).contiguous()

        if normalize_activations:
            act = act / norm_factor

        if step >= steps:
            break

        if (use_wandb or verbose) and (log_steps is not None) and (step % log_steps == 0):
            log_stats(
                trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues, verbose=verbose
            )

        if save_steps is not None and step in save_steps:
            for dir_path, trainer in zip(save_dirs, trainers):
                if dir_path is None:
                    continue

                if normalize_activations:
                    trainer.ae.scale_biases(norm_factor)

                ckpt_dir = os.path.join(dir_path, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)

                checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                t.save(checkpoint, os.path.join(ckpt_dir, f"ae_{step}.pt"))

                if normalize_activations:
                    trainer.ae.scale_biases(1.0 / norm_factor)

        if backup_steps is not None and (step % backup_steps == 0):
            for dir_path, trainer in zip(save_dirs, trainers):
                if dir_path is None:
                    continue
                t.save(
                    {
                        "step": step,
                        "ae": trainer.ae.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "config": _to_jsonable(trainer.config),
                        "norm_factor": norm_factor,
                    },
                    os.path.join(dir_path, "ae.pt"),
                )

        for trainer in trainers:
            with autocast_context:
                trainer.update(step, act)

    # Save finals
    for dir_path, trainer in zip(save_dirs, trainers):
        if dir_path is not None:
            if normalize_activations:
                trainer.ae.scale_biases(norm_factor)
            final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
            t.save(final, os.path.join(dir_path, "ae.pt"))

    if use_wandb:
        for q in log_queues:
            q.put("DONE")
        for p in wandb_processes:
            p.join()
