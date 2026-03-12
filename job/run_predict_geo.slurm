#!/bin/bash
#SBATCH --job-name=predict_geo
#SBATCH --partition=gpu_inter
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --output=/usr/users/geoguessr_ia/verbec_dia/logs/%x_%j.out
#SBATCH --error=/usr/users/geoguessr_ia/verbec_dia/logs/%x_%j.err

set -euo pipefail

mkdir -p /usr/users/geoguessr_ia/verbec_dia/logs
cd /usr/users/geoguessr_ia/verbec_dia

source .venv/bin/activate

python3 predict_geo.py --image timesquare.jpg --checkpoint checkpoints/vit_geo_best.pt