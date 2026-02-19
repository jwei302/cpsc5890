# Lab 3 — Visuomotor Policies, Action Embeddings, Autoencoders, and VAEs

Completed by Jeffrey Wei, Austin Feng. NOTE: Jeffrey missed Lab 3 on February 6th, so we were granted one extra day by Professor Fitzgerald to complete the lab and lab report. 

## Objectives

By the end of this lab, you will:

- Incorporate **observations (images)** into robot policies  
- Turn a **state-based BC policy** into a **visuomotor policy**
- Understand how **visual + state features** are fused inside policies
- Implement and train **Autoencoders (AE)** and **Variational Autoencoders (VAE)** for **action embeddings**
- Study how **model complexity** and **latent dimensionality** affect performance
- Use learned embeddings inside **Behavior Cloning (BC)**
- Evaluate performance on:
  - in-distribution (ID) start states  
  - out-of-distribution (OOD) start states  
- Deploy best models on the **real robot**

---

## Setup

Install GELLO software: https://github.com/wuphilipp/gello_software

Download training dataset of lifting task: https://drive.google.com/file/d/1wXDLTP4kNBk1rIlRg8erWKcrAW5BFSfq/view?usp=drive_link

Connect to Yale Secure: https://yale.service-now.com/it?id=kb_article&sysparm_article=KB0025500

## Part 1 — Visuomotor Behavior Cloning

We extend BC from:
state → action
to
(image, state) → action

### Policy Architecture

| Component | Role |
|----------|------|
| CNN encoder | Extract visual features |
| MLP encoder | Encode robot state |
| Fusion | Concatenate vision + state features |
| Policy head | Predict action |

### Tasks

1. Inspect the starter architectures
2. Match code to architecture diagrams
3. Train visuomotor BC
4. Compare:
   - training loss  
   - validation loss  
5. Evaluate policy on:
   - ID start states  
   - OOD start states  

### Reflection Questions

When do images actually help Behavior Cloning?

A: Images help when not all the information can be found in the state / joints data, which include information about the object, occlusions, and setting. It also helps with exact scene geometry and more precise and dexterous navigations. 

Why is the image encoder applied independently at each timestep?

A: The image encoder is applied independently such that we can use the same set of parameters to encode useful features, and separate the temporal reasoning component into a 1D CNN. The separation of spatial and temporal components can be useful for better training in comparison to one large spatiotemporal matrix.  

Why can BatchNorm be problematic in image-based BC?

A: Image-based BC uses smaller batch sizes for behavior cloning because of GPU memory storage limitations during training. BatchNorm however, works best with larger batch sizes because there are more stable mean and variance across larger batch sizes. Smaller batch sizes yield mean and variances with high noise. This is why we switch to GroupNorm. 

What are the risks of training an image encoder from scratch in BC?

A: Training an image encoder from scratch increases the likelihood for the encoder to overfit to the limited demos in the BC dataset. This causes the policy to rely less on the visual information due to overfitting and mostly rely on the states for action generation which yields poor OOD generalization. 

How can you tell whether the policy is actually using visual information?

A: We can test this at inference time by adding perturbations or augmentations to `obs_image` or `obs_wrist_image` and keeping the state fixed and see if the policy operate very differently. If they do, then the policy is meaningfully interpreting the images. 
---

## Part 2 — Action Autoencoders

We compress actions using:
action → encoder → latent z → decoder → reconstructed action

VAEs learn a **distribution** over latent actions.

Loss:

\[
L = \text{reconstruction} + \beta \cdot KL(q(z|a) || N(0,1))
\]

| AE | VAE |
|----|----|
| Deterministic | Probabilistic |
| Can overfit | Regularized |
| Weak OOD behavior | Better structured latent space |

### VAE → AE

If you:
- remove KL term
- make encoder deterministic  

then a VAE becomes a regular AE.

### Two Axes to Explore

| Axis | Meaning |
|------|---------|
| Model complexity | depth / width of encoder and decoder |
| Latent dimension | size of bottleneck |

### Deliverable 1 — VAE Performance Grid

Create a **3×3 table**:

VAE Training Reconstruction Loss

| Latent Dim (z_dim) / Hidden Size | 128 | 256 | 512 |
|--------------------------------------|-----|-----|-----|
| 8                                    | 0.001265 | 0.000822 | 0.000741 |
| 16                                   | 0.001055 | 0.000740 | 0.000735 |
| 32                                   | 0.000855 | 0.000648 | 0.000687 |

VAE Validation Reconstruction Loss

| Latent Dim (z_dim) / Hidden Size | 128 | 256 | 512 |
|--------------------------------------|-----|-----|-----|
| 4                                    | 0.002282 | 0.001864 | 0.001458 |
| 16                                   | 0.002156 | 0.001566 | 0.001340 |
| 32                                   | 0.001525 | 0.001301 | 0.001105 |

Report:
- Training reconstruction loss
- Validation reconstruction loss

### Questions

- When does increasing latent size stop helping?

A: In the latent dimensions that we tested, increasing the latent size has continually helped the validation loss. We suspect that if the latent dimension is too large however that there will be diminishing returns as there is little compression being done for the model to learn from reconstruction.  

- Which model overfits?

A: All the models overfit since all the training losses are significantly lower than their corresponding test loss. The clearest overfitting model is `hidden=128, z_dim=16`. 

- What is the best latent dimension for reconstruction?

A: The best latent dimension for reconstruction is `z_dim=32` because it has the lowest validation loss across different hidden dimensions. 

Now BC predicts **latent actions** instead of raw actions:

BC: (obs) → z
Decoder(z) → action

### Deliverable 2 — BC Performance Grid

Same 3×3 table, but report:

- BC training loss  
- BC validation loss  
- (Optional) task success rate  

BC Training Loss (final epoch, `[BC->z 030]`)

| Latent Dim (z_dim) / Hidden Size | 128 | 256 | 512 |
|--------------------------------------|-----|-----|-----|
| 8                                    | 0.003992 | 0.004702 | 0.004295 |
| 16                                   | 0.004333 | 0.004285 | 0.003775 |
| 32                                   | 0.004667 | 0.003989 | 0.003882 |

BC Testing Loss (final epoch, `[BC->z 030]`)

| Latent Dim (z_dim) / Hidden Size | 128 | 256 | 512 |
|--------------------------------------|-----|-----|-----|
| 8                                    | 0.077728 | 0.083724 | 0.084485 |
| 16                                   | 0.072760 | 0.070459 | 0.065360 |
| 32                                   | 0.070796 | 0.061343 | 0.060517 |

### Question

Does lower VAE reconstruction loss → better BC performance?  

A: Not necessarily because a VAE's econstruction quality does not necessarily imply control usefulness.

### Required Trials

Inference the best performing model on the real robot:

| Type | # Trials |
|------|----------|
| ID start | 2 |
| OOD start | 2 |

Submit videos.

## Part 6 — Repeat for State Embeddings

Repeat AE and VAE experiments, but encode:
