from huggingface_hub import list_repo_files, hf_hub_download

files = list_repo_files("AwesomeInterpretability/dlm-mask-topk-sae")

for f in files:
    if f.endswith('ae.pt') or f.endswith('config.json'):
        path = hf_hub_download(
            repo_id="AwesomeInterpretability/dlm-mask-topk-sae",
            local_dir='saes',
            filename=f
        )