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
FILTERING_SIZE=16
RANDOM_SEED=1013 # [1013, 2029, 3617, 5821, 7919], = 에 공백이 있으면 안됨!!
# =====================

pwd
which python
hostname
python3 main_ptflops_random_filtering.py \
 --data /local_datasets/imagenet/ \
 --data_corruption /local_datasets/imagenet-c \
 --exp_type 'each_shift_reset' \
 --algorithm 'eata_ptflops_random_filtering_last' \
 --batch_size ${BATCH_SIZE} \
 --filtering_size ${FILTERING_SIZE} \
 --seed ${RANDOM_SEED} \
 --output /data/okys515/repos/fake_audio_detection/project/final_tta_coreset/EATA/output/dir
 

exit 0