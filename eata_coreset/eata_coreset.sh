#!/usr/bin/bash


#SBATCH -J eata
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

# =====================
# 인자 값 정의
BATCH_SIZE=64
CORESET_SIZE=32
# =====================

pwd
which python
hostname
python3 main_coreset.py \
 --data /local_datasets/imagenet/ \
 --data_corruption /local_datasets/imagenet-c \
 --exp_type 'each_shift_reset' \
 --algorithm 'eata_coreset_center_based_distance' \
 --batch_size ${BATCH_SIZE} \
 --coreset_size ${CORESET_SIZE} \
 --output /data/okys515/repos/tta/eata_coreset/output/dir

exit 0