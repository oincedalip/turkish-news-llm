import json
from pathlib import Path
from huggingface_hub import login as hf_login


def huggingface_login():
    token_json_path = Path(__file__).resolve().parent / 'huggingface_token.json'
    with open(token_json_path, 'r') as f:
        token_json = json.load(f)
        token = token_json.get('token')
    hf_login(token=token)