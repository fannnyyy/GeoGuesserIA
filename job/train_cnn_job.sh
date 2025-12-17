#!/bin/bash 
#SBATCH --job-name=geoguessr_cnn
#SBATCH --partition=gpu_prod_night
#SBATCH --time=10:00:00
#SBATCH --output=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.out
#SBATCH --error=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.err

module load anaconda3/2022.10/gcc-13.1.0
eval "$(conda shell.bash hook)"
conda activate geoguesseriassh

cd $HOME/GeoGuesserIA/model/cnn
python3 cnn.py