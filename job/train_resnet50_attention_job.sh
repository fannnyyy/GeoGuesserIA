#!/bin/bash 
#SBATCH --job-name=geoguessr
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.out
#SBATCH --error=/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/logslurms/slurm-%j.err

module load anaconda3/2022.10/gcc-13.1.0
eval "$(conda shell.bash hook)"
conda activate geoguesseriassh

cd $HOME/GeoGuesserIA/model/cnn/cnn_with_attention_module
python3 resnet_cbam_undersamplingUS.py