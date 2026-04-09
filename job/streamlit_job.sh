#!/bin/bash
#SBATCH --job-name=streamlit_geo
#SBATCH --output=logs/streamlit_%j.log
#SBATCH --error=logs/streamlit_%j.err
#SBATCH --partition=gpu_tp
#SBATCH --time=02:00:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4


module load anaconda3/2022.10/gcc-13.1.0
eval "$(conda shell.bash hook)"
conda activate geoguesseriassh

cd $HOME/GeoGuesserIA/visualisation
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true