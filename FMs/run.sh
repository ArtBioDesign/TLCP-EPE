#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=qgpu_a40
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH --output=%j.out
#SBATCH --error=%j.err

/hpcfs/fhome/yangchh/software/anaconda/envs/bioseq/bin/python do_embedding.py