from huggingface_hub import list_repo_files, hf_hub_download
from transformers import AutoModel, AutoTokenizer

files = list_repo_files("AwesomeInterpretability/dlm-mask-topk-sae")

for f in files:
    if f.endswith('ae.pt') or f.endswith('config.json'):
        path = hf_hub_download(
            repo_id="AwesomeInterpretability/dlm-mask-topk-sae",
            local_dir='saes',
            filename=f
        )

model_path = "Dream-org/Dream-v0-Base-7B"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.save_pretrained('dream')

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

tokenizer.save_pretrained('dream_tokenizer')