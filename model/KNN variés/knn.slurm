#!/bin/bash
#SBATCH --job-name=knn_geo_light_country
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --output=/usr/users/geoguessr_ia/verbec_dia/logs/%x_%j.out
#SBATCH --error=/usr/users/geoguessr_ia/verbec_dia/logs/%x_%j.err

set -euxo pipefail

mkdir -p /usr/users/geoguessr_ia/verbec_dia/logs
cd /usr/users/geoguessr_ia/verbec_dia

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "Python=$(which python || true)"
echo "VENV exists? $(ls -la .venv/bin/python || true)"

source .venv/bin/activate

python -V
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available())"

python -u dataaugmentation_knn.py \
  --feature-backbone geo_vit \
  --geo-vit-checkpoint VIT/checkpoints/vit_geo_best.pt \
  --checkpoint-path checkpoints/knn_geo_light_country.pt \
  --output-path checkpoints/knn_geo_light_country_index.pt \
  --summary-path checkpoints/knn_geo_light_country_summary.json \
  --batch-size 64 \
  --num-workers 4 \
  --num-augmented-views 1 \
  --rotation-deg 4 \
  --color-jitter 0.05 \
  --country-penalty-multiplier 10 \
  --k-values 1,2,3,5
