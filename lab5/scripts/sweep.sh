#!/bin/bash

BASE_ENV="gym_xarm/XarmPickPlace"
REWARDS=("Dense" "Semi" "Sparse")
SUFFIX="-v0"


# for rwd in "${REWARDS[@]}"; do
#   echo "Running PPO | reward=$rwd"
#   ENV="${BASE_ENV}${rwd}${SUFFIX}"
#   python train.py \
#     --env $ENV \
#     --algo ppo \
#     --log_dir logs/long_ppo_${rwd}
# done

for rwd in "${REWARDS[@]}"; do
  echo "Running SAC | reward=$rwd"
  ENV="${BASE_ENV}${rwd}${SUFFIX}"
  python train.py \
    --env $ENV \
    --algo sac \
    --log_dir logs/long_sac_${rwd}
done
