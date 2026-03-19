import os
import sys

# ------------------ put warnings filter BEFORE importing anything heavy ------------------
import warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")
# ----------------------------------------------------------------------------------------

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

# Use segmented allocator to mitigate sporadic CUDA OOMs on large models
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import argparse
import random
import json
import time
import torch as t
import torch.multiprocessing as mp
import huggingface_hub
from datasets import config
from transformers import AutoModel, AutoTokenizer

import demo_config

# NOTE:
# The dictionary_learning package is imported as a submodule.
# The following utilities are used extensively across this demo:
from dictionary_learning.dictionary_learning.utils import (
    hf_dataset_to_generator,
    hf_mixed_dataset_to_generator,
    hf_sequence_packing_dataset_to_generator,
)
from dictionary_learning.dictionary_learning.pytorch_buffer import ActivationBuffer
from dictionary_learning.dictionary_learning.evaluation import evaluate
from dictionary_learning.dictionary_learning.training import trainSAE
import dictionary_learning.dictionary_learning.utils as utils


def get_args():
    """
    Parse CLI arguments for SAE training/evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Where to store the sweep results."
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--dry_run", action="store_true", help="Construct the sweep but do not train.")
    parser.add_argument(
        "--save_checkpoints", action="store_true", help="Save intermediate checkpoints during training."
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="Layer indices to train SAEs on."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model name (e.g., Dream-org/Dream-v0-Base-7B).",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=[e.value for e in demo_config.TrainerType],
        required=True,
        help="Which SAE architectures to train (e.g., batch_top_k, jump_relu).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on (e.g., cuda:0)."
    )
    parser.add_argument(
        "--hf_repo_id", type=str, help="Optional Hugging Face repo ID to push results to."
    )
    parser.add_argument(
        "--mixed_dataset", action="store_true",
        help="Use a mixed dataset pipeline (only if you know what you are doing)."
    )

    # ---------------- NEW: DLM forward-noising arguments ----------------
    parser.add_argument(
        "--dlm_mask_policy",
        type=str,
        choices=["unmask", "mask", "all", "clean"],
        default=None,
        help="DLM sampling policy for which positions to collect activations: "
             "'unmask' (default for Dream), 'mask', 'all', or 'clean' (default for AR LMs).",
    )
    parser.add_argument(
        "--dlm_t_min",
        type=float,
        default=None,
        help="Lower bound for per-batch masking probability p ~ U[t_min, t_max].",
    )
    parser.add_argument(
        "--dlm_t_max",
        type=float,
        default=None,
        help="Upper bound for per-batch masking probability p ~ U[t_min, t_max].",
    )
    # -------------------------------------------------------------------

    # ---------------- NEW: DLM debug printing arguments ----------------
    parser.add_argument(
        "--dlm_debug_print",
        action="store_true",
        help="If set, print per-batch mask/unmask selection stats during ActivationBuffer.refresh().",
    )
    parser.add_argument(
        "--dlm_debug_tokens",
        action="store_true",
        help="If set, also print a compact token visualization (first sample, truncated).",
    )
    parser.add_argument(
        "--dlm_debug_every",
        type=int,
        default=1,
        help="Print every N batches (default: 1). Only used if --dlm_debug_print is set.",
    )
    # -------------------------------------------------------------------

    args = parser.parse_args()
    return args


def _canonical_device(device_str: str) -> str:
    """
    Make sure the device string is valid on this machine. If not, fallback gracefully.
    """
    try:
        if device_str.startswith("cuda"):
            if not t.cuda.is_available():
                return "cpu"
            if ":" in device_str:
                idx = int(device_str.split(":")[1])
                if idx < 0 or idx >= t.cuda.device_count():
                    # Fallback to the default current CUDA device (let torch decide)
                    return "cuda"
            return device_str
        return device_str
    except Exception:
        # Extremely defensive fallback
        return "cuda" if t.cuda.is_available() else "cpu"


def _is_diffusion_lm(model) -> bool:
    """
    Heuristic to detect a Diffusion LM (like Dream).
    We avoid AR-only evaluations (e.g., CE loss recovered) for such models.
    """
    name = model.__class__.__name__.lower()
    if "dream" in name or "diffusion" in name:
        return True
    return hasattr(model, "diffusion_generate")


def _load_model_and_tokenizer(model_name: str, dtype):
    """
    Load a HF model and tokenizer in a way that is compatible with both AR LMs and Dream (DLM).
    We use AutoModel + trust_remote_code=True to ensure Dream's custom modeling files are respected.
    """
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    if not hasattr(tok, "pad_token") or tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,  # required for Dream models
        device_map="auto",
        dtype=dtype              # avoid deprecated torch_dtype
    )
    model.eval()
    return model, tok


def _apply_dlm_cli_defaults(args, is_dlm: bool):
    """
    Decide the effective DLM forward-noising settings based on CLI/env/defaults,
    and set them into environment variables so ActivationBuffer can read them.

    Priority:
      CLI args > existing env vars > model-type defaults (Dream->unmask, AR->clean).
    """
    # 1) policy
    default_policy = "unmask" if is_dlm else "clean"
    policy = args.dlm_mask_policy or os.getenv("DLM_MASK_POLICY") or default_policy
    policy = str(policy).strip().lower()
    if policy not in ("unmask", "mask", "all", "clean"):
        policy = default_policy

    # 2) t_min / t_max
    def _float_env(name: str, default_str: str) -> float:
        try:
            return float(os.getenv(name, default_str))
        except Exception:
            return float(default_str)

    t_min = args.dlm_t_min if args.dlm_t_min is not None else _float_env("DLM_T_MIN", "0.05")
    t_max = args.dlm_t_max if args.dlm_t_max is not None else _float_env("DLM_T_MAX", "0.50")
    try:
        if not (0.0 <= t_min <= t_max):
            raise ValueError
    except Exception:
        t_min, t_max = 0.05, 0.50

    # 3) export to env for ActivationBuffer
    os.environ["DLM_MASK_POLICY"] = policy
    os.environ["DLM_T_MIN"] = str(t_min)
    os.environ["DLM_T_MAX"] = str(t_max)

    print(f"[DLM] mask_policy={policy}, t_min={t_min}, t_max={t_max} (default_by_model={default_policy})")


def _apply_dlm_debug_cli(args):
    """
    Write DLM debug printing switches to environment so ActivationBuffer can pick them up.
    """
    # If user passed the flags, override; else honor existing env; else defaults.
    if args.dlm_debug_print:
        os.environ["DLM_DEBUG_PRINT"] = "1"
    else:
        os.environ.setdefault("DLM_DEBUG_PRINT", "0")

    if args.dlm_debug_tokens:
        os.environ["DLM_DEBUG_TOKENS"] = "1"
    else:
        os.environ.setdefault("DLM_DEBUG_TOKENS", "0")

    # dlm_debug_every applies only if printing enabled, but we still set it for consistency
    try:
        every = int(args.dlm_debug_every)
        if every <= 0:
            every = 1
    except Exception:
        every = 1
    os.environ["DLM_DEBUG_EVERY"] = str(every)

    # One-line summary to stdout for sanity
    print(
        f"[DLM-DEBUG] print={os.environ['DLM_DEBUG_PRINT']} "
        f"tokens={os.environ['DLM_DEBUG_TOKENS']} "
        f"every={os.environ['DLM_DEBUG_EVERY']}"
    )


def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
    mixed_dataset: bool = False,
    # NEW: pass-through of args for dlm defaults
    dlm_mask_policy: str | None = None,
    dlm_t_min: float | None = None,
    dlm_t_max: float | None = None,
    # NEW: pass-through of args for debug printing
    dlm_debug_print: bool = False,
    dlm_debug_tokens: bool = False,
    dlm_debug_every: int = 1,
):
    """
    Main entry to train one or more SAE trainers at a given model layer.
    """
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    # Canonicalize device once and pass it everywhere
    device = _canonical_device(device)

    # Model/data hyperparams (from demo_config for Dream-7B)
    context_length = demo_config.DLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.DLM_CONFIG[model_name].llm_batch_size
    sae_batch_size = demo_config.DLM_CONFIG[model_name].sae_batch_size
    dtype = demo_config.DLM_CONFIG[model_name].dtype

    num_buffer_inputs = buffer_tokens // context_length
    print(f"[Info] buffer_size={num_buffer_inputs}, buffer_tokens={buffer_tokens}")

    # Log cadence & total steps
    log_steps = 100
    steps = int(num_tokens / sae_batch_size)
    print(f"[Info] total_batches={steps}, num_tokens={num_tokens}, sae_batch_size={sae_batch_size}")

    # Optional checkpoint schedule
    if save_checkpoints:
        desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()
        print(f"[Info] desired_checkpoints={desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"[Info] save_steps={save_steps}")
    else:
        save_steps = None

    # Load model/tokenizer
    model, tokenizer = _load_model_and_tokenizer(model_name, dtype)

    # Decide DLM defaults now that we have the model object
    is_dlm = _is_diffusion_lm(model)
    _apply_dlm_cli_defaults(
        argparse.Namespace(
            dlm_mask_policy=dlm_mask_policy,
            dlm_t_min=dlm_t_min,
            dlm_t_max=dlm_t_max,
        ),
        is_dlm=is_dlm,
    )

    # Apply debug switches (print controls)
    _apply_dlm_debug_cli(
        argparse.Namespace(
            dlm_debug_print=dlm_debug_print,
            dlm_debug_tokens=dlm_debug_tokens,
            dlm_debug_every=dlm_debug_every,
        )
    )

    # Truncate for memory/computation
    model = utils.truncate_model(model, layer)

    # Submodule to hook
    submodule = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"

    io = "out"
    activation_dim = model.config.hidden_size

    # Data generator
    if mixed_dataset:
        qwen_system_prompt_to_remove = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        generator = hf_mixed_dataset_to_generator(
            tokenizer,
            system_prompt_to_remove=qwen_system_prompt_to_remove,
            min_chars=context_length * 4,
        )
    else:
        generator = hf_sequence_packing_dataset_to_generator(
            tokenizer,
            min_chars=context_length * 4,
        )
        print("[Info] Using hf_sequence_packing_dataset_to_generator")

    # Activation buffer
    activation_buffer = ActivationBuffer(
        generator=generator,              # alias accepted by patched ActivationBuffer
        model=model,
        submodule=submodule,
        n_ctxs=num_buffer_inputs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
        add_special_tokens=False,
    )

    # SAE trainer configs
    trainer_configs = demo_config.get_trainer_configs(
        architectures=architectures,
        learning_rates=learning_rates,
        seeds=random_seeds,
        activation_dim=activation_dim,
        dict_sizes=dictionary_widths,
        model_name=model_name,
        device=device,  # <-- pass canonical device
        layer=layer,
        submodule_name=submodule_name,
        steps=steps,
    )

    print(f"[Info] number_of_trainer_configs={len(trainer_configs)}")
    assert len(trainer_configs) > 0, "No trainer config generated."

    # Prepare output directories
    root = Path(save_dir) / submodule_name
    root.mkdir(parents=True, exist_ok=True)
    for i in range(len(trainer_configs)):
        (root / f"trainer_{i}").mkdir(parents=True, exist_ok=True)
    save_dir = str(root)

    # Train
    if not dry_run:
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            wandb_project=demo_config.wandb_project,
            normalize_activations=True,
            verbose=False,
            autocast_dtype=t.bfloat16,
            backup_steps=1000,
            device=device,               # <-- forward to training loop for autocast & logging
        )


@t.no_grad()
def eval_saes(
    model_name: str,
    ae_paths: list[str],
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
) -> dict:
    """
    Evaluate trained SAEs. For Diffusion LMs (e.g., Dream), we skip AR-only CE/loss-recovered evaluations.
    """
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    device = _canonical_device(device)

    io = "in_and_out" if transcoder else "out"

    context_length = demo_config.DLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.DLM_CONFIG[model_name].llm_batch_size
    dtype = demo_config.DLM_CONFIG[model_name].dtype

    loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    sae_batch_size = loss_recovered_batch_size * context_length

    # Find max layer to truncate
    max_layer = 0
    for ae_path in ae_paths:
        config_path = f"{ae_path}/config.json"
        if not os.path.exists(config_path):
            continue
        with open(config_path, "r") as f:
            cfg = json.load(f)
        layer = cfg["trainer"]["layer"]
        max_layer = max(max_layer, layer)

    # Load model
    model, _tok = _load_model_and_tokenizer(model_name, dtype)
    model = utils.truncate_model(model, max_layer)

    # Build a small input pool
    buffer_size = n_inputs
    n_batches = max(1, n_inputs // loss_recovered_batch_size)

    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i > n_inputs * 5:
            break

    eval_results = {}

    # Skip AR-only metrics for DLM
    diffusion_mode = _is_diffusion_lm(model)
    if diffusion_mode:
        print("[Eval] Detected a Diffusion LM. Skipping AR-only CE/loss-recovered metrics.")
        for ae_path in ae_paths:
            output_filename = f"{ae_path}/eval_results.json"
            if (not overwrite_prev_results) and os.path.exists(output_filename):
                print(f"[Eval] Skipping {ae_path} as eval results already exist")
                continue
            results = {
                "info": "Diffusion LM detected. AR-only CE/loss-recovered metrics are skipped.",
                "hyperparameters": {
                    "n_inputs": n_inputs,
                    "context_length": context_length,
                }
            }
            with open(output_filename, "w") as f:
                json.dump(results, f)
            eval_results[ae_path] = results
        return eval_results

    # ===== AR-only evaluation path (kept for completeness) =====
    for ae_path in ae_paths:
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results and os.path.exists(output_filename):
            print(f"[Eval] Skipping {ae_path} as eval results already exist")
            continue

        dictionary, cfg = utils.load_dictionary(ae_path, device)
        dictionary = dictionary.to(dtype=model.dtype)

        layer = cfg["trainer"]["layer"]
        submodule = utils.get_submodule(model, layer)
        activation_dim = cfg["trainer"]["activation_dim"]

        activation_buffer = ActivationBuffer(
            iter(input_strings),
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
        )

        results = evaluate(
            dictionary,
            activation_buffer,
            context_length,
            loss_recovered_batch_size,
            io=io,
            device=device,
            n_batches=n_batches,
        )
        results["hyperparameters"] = {
            "n_inputs": n_inputs,
            "context_length": context_length,
        }
        print(results)

        with open(output_filename, "w") as f:
            json.dump(results, f)
        eval_results[ae_path] = results

    return eval_results


def push_to_huggingface(save_dir: str, repo_id: str):
    """
    Optionally push the entire run directory to a Hugging Face model repo.
    """
    api = huggingface_hub.HfApi()
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=save_dir,
    )


if __name__ == "__main__":
    """
    Example usage:
      python demo.py --save_dir ./run --model_name Dream-org/Dream-v0-Base-7B \
        --layers 18 --architectures batch_top_k jump_relu --use_wandb \
        --dlm_mask_policy unmask --dlm_t_min 0.05 --dlm_t_max 0.5 \
        --dlm_debug_print --dlm_debug_tokens --dlm_debug_every 1
    """
    args = get_args()

    hf_repo_id = args.hf_repo_id
    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    # Improve stability for multiprocessing and streaming reads
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    mp.set_start_method("spawn", force=True)
    config.STREAMING_READ_MAX_RETRIES = 100
    config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()

    # Compose a save directory slug that includes model and arch info
    save_dir = (
        f"{args.save_dir}_{args.model_name}_{'_'.join(args.architectures)}".replace(
            "/", "_"
        )
    )

    # Train SAEs for each requested layer
    for layer in args.layers:
        run_sae_training(
            model_name=args.model_name,
            layer=layer,
            save_dir=save_dir,
            device=args.device,
            architectures=args.architectures,
            num_tokens=demo_config.num_tokens,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=demo_config.dictionary_widths,
            learning_rates=demo_config.learning_rates,
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
            save_checkpoints=args.save_checkpoints,
            mixed_dataset=args.mixed_dataset,
            # NEW: forward CLI overrides to per-layer run (which then set env before buffer)
            dlm_mask_policy=args.dlm_mask_policy,
            dlm_t_min=args.dlm_t_min,
            dlm_t_max=args.dlm_t_max,
            # NEW: debug print switches
            dlm_debug_print=args.dlm_debug_print,
            dlm_debug_tokens=args.dlm_debug_tokens,
            dlm_debug_every=args.dlm_debug_every,
        )

    # Collect all trained AE folders and (optionally) evaluate
    ae_paths = utils.get_nested_folders(save_dir)
    eval_saes(
        model_name=args.model_name,
        ae_paths=ae_paths,
        n_inputs=demo_config.eval_num_inputs,
        device=args.device,
        overwrite_prev_results=True,
    )

    print(f"[Done] Total time: {time.time() - start_time:.2f}s")

    # Optionally push results to HF
    if hf_repo_id:
        push_to_huggingface(save_dir, hf_repo_id)
