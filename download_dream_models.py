from huggingface_hub import list_repo_files, hf_hub_download
from transformers import AutoModel, AutoTokenizer

qwen_sae = 'qwen-topk-sae'
dream_sae = 'dlm-mask-topk-sae'
sae_type = dream_sae
files = list_repo_files("AwesomeInterpretability/{}".format(sae_type))

for f in files:
    if f.endswith('ae.pt') or f.endswith('config.json'):
        path = hf_hub_download(
            repo_id="AwesomeInterpretability/{}".format(sae_type),
            local_dir='{}s'.format(sae_type),
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
