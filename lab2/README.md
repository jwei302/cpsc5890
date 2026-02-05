# Lab 2 — Behavior Cloning (BC) and DAgger

Partners: Jeffrey Wei, Austin Feng
## Objectives
- Understand how demonstrations are represented as data
- Collect demonstrations using GELLO or kinesthetic teaching
- Replay and analyze demonstrations
- Implement simple policies and Behavior Cloning (BC)
- Train, evaluate, and visualize BC models
- Implement, train, and evaluate DAgger (robot-led and human-led)
- Compare vanilla BC and DAgger approaches
- Understand evaluation metrics and intervention strategies


## Part 1: Understanding Demonstrations

### Step 1: Demonstrations

GELLO walk-through.

## Part 2: Behavior Cloning

### Goal

In this part, you will train and evaluate a Behavior Cloning (BC) policy that learns to imitate robot behavior from demonstration data. Your objective is to verify that the trained BC model can reproduce actions similar to the recorded demonstrations and to understand its limitations.

Behavior Cloning treats imitation learning as a supervised learning problem: given a robot state, the model predicts the action taken by the demonstrator in that state.

### Task

### Step 1: Train the BC model

Use a pre-recorded demonstration dataset ```asset/demo.npz``` to train a BC model:

```bash

python -m scripts.bc --mode train --ip <robot_ip> --epochs <epochs> --batch_size <batch_size> --lr <lr>
```
* Experiment with hyperparameters (epochs, batch_size, lr)

* Observe the training and test loss over epochs

### Step 2: Run inference on the robot
Use the trained BC model to generate actions on the robot:

```bash

python -m scripts.bc --ip <robot_ip> --mode inference
```
The robot will replay actions predicted by the BC model
Observe smoothness, accuracy, and timing relative to the original demonstration
Record the EEF state trajectory using the provided visualization code.

### Step 3: Training and Inference Using High-Frequency Data

Repeat Step 1 and 2 using ```asset/demo_high_freq.npz``` instead.

```bash
python -m scripts.bc --mode train --ip <robot_ip> \
    --data asset/demo_high_freq.npz \
    --epochs <epochs> --batch_size <batch_size> --lr <lr>
```

Tune hyperparameters as needed (epochs, batch size, learning rate).

Monitor training and test loss curves.

Compare convergence behavior against the low-frequency dataset.

Run inference using the BC model trained on high-frequency data. Be sure to set ```wait=False``` in ```set_servo_angle``` (line 355) and ```set_gripper_position``` (line 361) to enable high-frequency control. Leave ```wait=True``` at anywhere else.

```bash
python -m scripts.bc --mode inference --ip <robot_ip>
```
Observe the resulting robot motion.

Pay attention to smoothness, responsiveness, and stability.

Compare execution timing and trajectory fidelity against the original demonstrations.

Record the EEF state trajectory using the provided visualization code.

### What to Record and Report
* Are the movements smooth?
* Do actions closely follow the demonstration?
* Are there any large deviations, jerks, or unexpected behavior?
* Start the robot from slightly different initial poses. Observe whether the BC model still produces reasonable behavior
* Training hyperparameters: epochs, batch_size, lr
* Loss over time. Final training and test loss
* Visualization of visited EEF states
* Record a video of successes and failures

### Reflection Questions
* How closely does the BC model reproduce the original demonstrations?
* Where does the model fail or deviate most significantly? Why might that happen?
* What could go wrong if the robot starts from a pose outside the demonstration distribution?
* Why is normalization of states and actions important for BC performance? Is this the only way to pre-process data?

## Part 3: DAgger

### Goal

In this part, you will iteratively improve the Behavior Cloning (BC) policy by collecting on-policy states visited by the learned policy and labeling them with the expert (human or scripted). The goal is to reduce compounding errors that occur when the BC policy encounters states not present in the original demonstrations.

#### Key Concept:

Vanilla BC only learns from offline demonstrations. DAgger collects states visited by the learned policy and adds expert labels, reducing distributional shift.

### Step 1: Initialize DAgger

Use the high-frequency demonstration dataset ```asset/demo_high_freq.npz``` to train an initial BC policy:

```bash

python -m scripts.dagger \
    --mode train \
    --data asset/demo_high_freq.npz \
    --epochs <epochs> \
    --batch-size <batch-size> \
    --lr <lr> \
    --ip <robot_ip>
```
This is your starting BC model.

### Step 2: Run DAgger Iterations
Run DAgger to collect new on-policy states and aggregate them with the original dataset. Finish any TODO before running the following script.

```bash
python -m scripts.dagger \
    --mode dagger \
    --ip <robot_ip>
```
Add and adjust prarameter as needed.

Explanation of parameters:
```bash
--dagger-iters: Number of DAgger iterations (retrain with aggregated data).

--dagger-rollout-episodes: Number of episodes to collect per iteration.

--beta0 / --beta-decay: Probability of following the expert vs learned policy during rollouts.

```

During each iteration:

The robot executes a mixture policy (expert with probability β, learned BC policy otherwise).

States visited by the robot are labeled with the expert action.

The dataset is aggregated and used to retrain the BC model.

After each iteration, the following are saved automatically:
```bash
asset/bc_policy.pt       # retrained model
asset/bc_norm.npz        # normalization stats
asset/dagger_agg.npz     # aggregated dataset
```

### Step 3: Run Inference Using the DAgger-Trained Policy
Repeat Step 1 and 2 using ```asset/demo_high_freq.npz``` instead,

Train the BC model using the high-frequency dataset:

```bash
python -m lab2.scripts.dagger \
    --mode inference \
    --ip <robot_ip> \
    --episodes <episodes> \
    --out asset/inf_dagger.npz
```

This runs the DAgger-trained policy on the robot.

Observe trajectory smoothness, accuracy, and response compared to vanilla BC.

Record EEF state trajectories using plot_3d_positions.

### What to Record and Report
* Aggregated dataset size after each DAgger iteration.
* Training and test loss curves across iterations.
* Comparison of robot motion: BC vs DAgger.
* Are movements smoother?
* Does DAgger reduce deviations in states outside the original demonstration distribution?
* Videos of robot performing DAgger policy. Label successes and failures.
* EEF trajectories visualized in 3D.

### Reflection Questions
* How does DAgger improve performance compared to vanilla BC?
* Which states benefit most from expert relabeling?
* How does high-frequency data affect DAgger’s stability and learning?
* What are potential risks if β decays too quickly or too slowly?
* How could you extend this approach to handle dynamic tasks or obstacles?

## Final Questions

* How are demonstrations represented in the dataset? What do the observation and action arrays correspond to?

A: 

* How does the sampling frequency (low vs high frequency) affect the recorded data?

A: The higher the frequency, the more smooth the recorded data. The lower the frequency, the choppier the recorded data. 

* Did you notice any noise or irregularities in the demonstrations? How might these affect imitation learning?

A: Did not get to this point in the lab. 

* How closely did the BC model reproduce the original demonstrations? Provide examples.

A: We ran into some errors with the BC model. We suspect some normalization error or torch tensor configuration errors. 

* In which situations did the BC model fail or deviate from the demonstrations? Why might this happen?

A: In more difficult tasks, one wrong action pushes the model into an unknown distribution and causes a compounding error. 

* How does the model behave when the robot starts from a state outside the demonstration distribution?

A: The model hallucinates and is unsure because it is out of domain. It's learned state action pairs are only for states within the demonstration distribution. Thus, it will do poorly. 

* How do hyperparameters (epochs, batch size, learning rate) affect the training and test loss?

A: The more epochs the lower the test loss. A higher learning rate yields a faster decaying loss curve. Increasing batch size helps decrease the stochasticity of gradient descent, but we are unsure if it would improve or worsen the train / test loss. 

* How smooth and responsive were the robot’s actions during BC inference? Were there any jerks or unexpected movements?

A: The actions were relatively jerky. 

* Why is normalization of states and actions crucial for BC? Can you think of other preprocessing methods that might help?

A: It forces input features and output actions to be within a reasonable range, allowing the training signal to be better distributed for supervised learning regression tasks. Without it, the large gripper actions or the different degrees will dominate the gradient signal. Some other preprocessing could be Image Preprocessing / Augmentation such as perturbing the input image to add robustness to the state action pair that the model is learning. 

* How does DAgger address the compounding error problem seen in vanilla BC?

A: It does so by sampling with probability $\beta$ expert demonstrations to add to its training data after every dagger iteration. Then by training with expert demonstration data at the current state, it is able to keep the policy on track. 

* What effect did aggregating on-policy states have on model performance?

A: We did not get to this part. 

* How did the choice of beta (expert probability) affect the policy rollout? What happened when beta decayed too quickly or too slowly?

A: We did not get to this part. We suspect if $\beta$ decays to quickly, then the BC policy may not get enough expert demonstration support, causing errors to seep in. If it is decays too slowly, the the policy may roll super slowly because it is frequently requesting expert help. 

* If DAgger did not improve policy by much, why?

A: We did not get to this part. 