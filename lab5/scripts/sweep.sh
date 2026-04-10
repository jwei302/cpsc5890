#!/bin/bash

ENV="gym_xarm/XarmReach-v0"
TIMESTEPS=3000000

# Common values
LR_VALUES=(1e-5 3e-5 1e-4)
GAMMA_VALUES=(0.95 0.99 0.999)

CLIP_VALUES=(0.1 0.2 0.3)

for lr in "${LR_VALUES[@]}"; do
  for gamma in "${GAMMA_VALUES[@]}"; do
    for clip in "${CLIP_VALUES[@]}"; do

      echo "Running PPO | lr=$lr gamma=$gamma clip=$clip"

      python train.py \
        --env $ENV \
        --algo ppo \
        --timesteps $TIMESTEPS \
        --lr $lr \
        --gamma $gamma \
        --clip_range $clip \
        --log_dir logs/long_ppo_lr${lr}_g${gamma}_clip${clip}

    done
  done
done

# Common values
LR_VALUES=(1e-3 5e-3 1e-2)
GAMMA_VALUES=(0.9 0.99 0.95)

ENT_VALUES=("auto" "0.1" "0.5")

for lr in "${LR_VALUES[@]}"; do
  for gamma in "${GAMMA_VALUES[@]}"; do
    for ent in "${ENT_VALUES[@]}"; do

      echo "Running SAC | lr=$lr gamma=$gamma ent=$ent"

      python train.py \
        --env $ENV \
        --algo sac \
        --timesteps $TIMESTEPS \
        --lr $lr \
        --gamma $gamma \
        --ent_coef $ent \
        --log_dir logs/long_sac_lr${lr}_g${gamma}_ent${ent}

    done
  done
done