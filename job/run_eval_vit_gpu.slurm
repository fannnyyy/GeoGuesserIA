#!/bin/bash
#SBATCH --job-name=eval_geo
#SBATCH --partition=gpu_inter
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28000M
#SBATCH --output=/usr/users/geoguessr_ia/verbec_dia/logs/%x_%j.out
#SBATCH --error=/usr/users/geoguessr_ia/verbec_dia/logs/%x_%j.err

set -euo pipefail
mkdir -p /usr/users/geoguessr_ia/verbec_dia/logs
cd /usr/users/geoguessr_ia/verbec_dia

source .venv/bin/activate
python -u eval_geo.py