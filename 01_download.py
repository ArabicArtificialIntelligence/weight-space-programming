"""Step 1 - download the original SmolLM2-360M-Instruct from Hugging Face.

    python 01_download.py

Saves to ./models/original  (about 724 MB). Nothing is changed in it.
"""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEST = Path(__file__).parent / "models" / "original"

path = snapshot_download(
    repo_id=REPO,
    local_dir=DEST,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
print(f"Downloaded {REPO} -> {path}  ({size / 1e6:.0f} MB)")
for f in sorted(Path(path).iterdir()):
    if f.is_file():
        print(f"  {f.name:32s} {f.stat().st_size / 1e6:8.1f} MB")
