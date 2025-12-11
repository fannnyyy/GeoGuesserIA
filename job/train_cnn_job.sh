#!/bin/bash 
#SBATCH --job-name=geoguessr
#SBATCH --partition=gpu_tp
#SBATCH --time=2:00:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err

module load anaconda3/2022.10/gcc-13.1.0

eval "$(conda shell.bash hook)"

conda activate geoguesseriassh
pip install scikit-learn torch torchvision pillow numpy pandas matplotlib



cd $HOME/GeoGuesserIA/model/cnn
python3 cnn.py