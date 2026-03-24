#!/bin/bash
#SBATCH --job-name=bench_streetclip
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --output=/usr/users/geoguessr_ia/verbec_dia/logs/%x_%j.out
#SBATCH --error=/usr/users/geoguessr_ia/verbec_dia/logs/%x_%j.err

set -euxo pipefail

mkdir -p /usr/users/geoguessr_ia/verbec_dia/logs
mkdir -p /usr/users/geoguessr_ia/verbec_dia/outputs/streetclip_eval
cd /usr/users/geoguessr_ia/verbec_dia

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "Python=$(which python || true)"
echo "VENV exists? $(ls -la .venv/bin/python || true)"

source .venv/bin/activate

python -V
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python - <<'PY'
from pathlib import Path

p = Path("/usr/users/geoguessr_ia/verbec_dia/StreetCLIP/pytorch_model.bin")
is_lfs_pointer = False
if p.exists() and p.stat().st_size <= 1024:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    is_lfs_pointer = "git-lfs.github.com/spec/v1" in txt

if is_lfs_pointer:
    print("[INFO] LFS pointer detected, downloading real StreetCLIP weights from Hugging Face...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="geolocal/StreetCLIP",
        local_dir="/usr/users/geoguessr_ia/verbec_dia/StreetCLIP",
        local_dir_use_symlinks=False,
    )
    print("[INFO] Download complete.")
else:
    print("[INFO] Local StreetCLIP weights look valid, no download needed.")
PY

python -u bench_streetclip.py \
  --dataset-dir dataset/osv5m_test \
  --csv-path dataset/osv5m_test/test_filtered.csv \
  --model-dir StreetCLIP \
  --output-dir outputs/streetclip_eval \
  --batch-size 16 \
  --num-workers 4 \
  --text-batch-size 256
