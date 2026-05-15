from huggingface_hub import list_repo_files, hf_hub_download
from transformers import AutoModel, AutoTokenizer
import torch

files = list_repo_files("AwesomeInterpretability/dlm-mask-topk-sae")

for f in files:
    if f.endswith('ae.pt') or f.endswith('config.json'):
        path = hf_hub_download(
            repo_id="AwesomeInterpretability/dlm-mask-topk-sae",
            local_dir='dream_saes',
            filename=f
        )

files = list_repo_files("AwesomeInterpretability/llada-mask-topk-sae")

for f in files:
    if f.endswith('ae.pt') or f.endswith('config.json'):
        path = hf_hub_download(
            repo_id="AwesomeInterpretability/llada-mask-topk-sae",
            local_dir='llada_saes',
            filename=f
        )

model_paths = ["Dream-org/Dream-v0-Base-7B", "GSAI-ML/LLaDA-8B-Base"]
model_names = ["dream", "llada"]

for model_path, model_name in zip(model_paths, model_names):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model.save_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    tokenizer_name = model_name + '_tokenizer'
    tokenizer.save_pretrained(tokenizer_name)
