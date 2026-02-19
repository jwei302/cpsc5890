#!/bin/bash

# Arrays of values
z_dims=(16 32)
ae_hiddens=(128 256 512)

# Loop over all combinations
for z in "${z_dims[@]}"; do
  for h in "${ae_hiddens[@]}"; do
    echo "Running with z_dim=$z and ae_hidden=$h"
    python3 action_vae.py --mode bc --z_dim "$z" --ae_hidden "$h" --ae_epochs 50 --bc_epochs 30 --ckpt_path asset/checkpoints/bcconv_latent_final_{$z}_{$h}.pt > runs/bc/run_${z}_${h}.txt
  done
done

# for z in "${z_dims[@]}"; do
#   for h in "${ae_hiddens[@]}"; do
#     echo "Running with z_dim=$z and ae_hidden=$h"
#     python3 action_vae.py --mode vae --z_dim "$z" --ae_hidden "$h" --ae_epochs 50 --bc_epochs 50 > runs/vae/run_${z}_${h}.txt
#     python3 action_vae.py --mode bc --z_dim "$z" --ae_hidden "$h" --ae_epochs 50 --ckpt_path asset/checkpoints/bcconv_latent_final_{$z}_{$h}.pt > runs/bc/run_${z}_${h}.txt
#   done
# done


#--ckpt_path asset/checkpoints/bcconv_latent_final_16_256.pt 