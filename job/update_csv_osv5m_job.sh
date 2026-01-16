#!/bin/bash
#SBATCH --job-name=update_csv_osv5m
#SBATCH --partition=gpu_prod_night
#SBATCH --time=12:00:00
#SBATCH --output=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.out
#SBATCH --error=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.err


module load anaconda3/2022.10/gcc-13.1.0
eval "$(conda shell.bash hook)"
conda activate geoguesseriassh

cd ~/GeoGuesserIA/dataset_OSV5M

python3 Update_csv.py