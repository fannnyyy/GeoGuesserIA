#!/bin/bash
#SBATCH --job-name=knn_dinov2_large
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --output=/usr/users/geoguessr_ia/badoul_fan/logs/%x_%j.out
#SBATCH --error=/usr/users/geoguessr_ia/badoul_fan/logs/%x_%j.err

set -euxo pipefail

mkdir -p /usr/users/geoguessr_ia/badoul_fan/logs
mkdir -p /usr/users/geoguessr_ia/badoul_fan/.cache/huggingface
cd /usr/users/geoguessr_ia/badoul_fan

export HF_HOME=/usr/users/geoguessr_ia/badoul_fan/.cache/huggingface
export TRANSFORMERS_CACHE=/usr/users/geoguessr_ia/badoul_fan/.cache/huggingface

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "Python=$(which python || true)"
echo "VENV exists? $(ls -la .venv/bin/python || true)"
echo "HF_HOME=$HF_HOME"

source .venv/bin/activate

python -V
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

python -u dinov2_knn.py \
  --dinov2-model-name facebook/dinov2-large \
  --checkpoint-path checkpoints/dinov2_knn_geo.pt \
  --output-path checkpoints/dinov2_knn_geo_index.pt \
  --summary-path checkpoints/dinov2_knn_geo_summary.json \
  --batch-size 16 \
  --num-workers 4 \
  --num-augmented-views 0 \
  --rotation-deg 0 \
  --color-jitter 0 \
  --country-penalty-multiplier 10 \
  --temperature 0.05 \
  --k-values 1,2,3
