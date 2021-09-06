#!/bin/bash -l
#SBATCH -J mu
#SBATCH -o mu_%j.out
#SBATCH -e mu_%j.err
#SBATCH -n 1

export CUDA_VISIBLE_DEVICES='-1'
export LD_LIBRARY_PATH=/opt/cuda/9.0/lib64:/opt/cudnn/7.2
export LC_ALL=

SPLIT='val'

time python -m utils.misunderstandings \
    -split $SPLIT \
    -out_fname mu-$SPLIT-Tkn_history_rerun.json \
    -img_feat rss \
    -bin_path bin/Oracle/oracleTkn_history 
