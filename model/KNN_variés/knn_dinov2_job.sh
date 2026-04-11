#!/bin/bash
#SBATCH --job-name=knn_dinov2_large
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000M
#SBATCH --output=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.out
#SBATCH --error=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.err

module load anaconda3/2022.10/gcc-13.1.0
eval "$(conda shell.bash hook)"
conda activate geoguesseriassh

cd $HOME/GeoGuesserIA/model/KNN_variés

export HF_HOME=/usr/users/geoguessr_ia/badoul_fan/.cache/huggingface
export TRANSFORMERS_CACHE=/usr/users/geoguessr_ia/badoul_fan/.cache/huggingface

python3 -u dinov2_knn.py \
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