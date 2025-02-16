#!/bin/sh
#SBATCH --job-name=learning-NN
#SBATCH --partition seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=20G
#SBATCH -t 0-6:00 
#SBATCH -c 1
#SBATCH --chdir=/n/home04/keremdayi/learning-NN
#SBATCH --output=/n/home04/keremdayi/logs/learning-NN_%A__%a.out
#SBATCH --mail-user=keremdayi@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.12-fasrc01
module load intelpython/3.9.16-fasrc01

mamba activate learning-nn

export GLOO_SOCKET_IFNAME=eth0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

python /n/home04/keremdayi/learning-NN/train.py --device=cuda --seed=62
