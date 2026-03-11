#!/bin/bash
#SBATCH --job-name=vit_geo
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

python -u train_vit_geo.py