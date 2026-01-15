#!/bin/bash
#SBATCH --job-name=download_osv5m
#SBATCH --partition=gpu_prod_night
#SBATCH --time=12:00:00
#SBATCH --output=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.out
#SBATCH --error=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.err

source ~/.bashrc
conda activate geoguesseriassh

cd ~/GeoGuesserIA/dataset_OSV5M

python3 download_dataset.py

echo "Téléchargement terminé"