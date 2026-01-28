# Lab 2 — Behavior Cloning (BC) and DAgger

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
**Goals**


**Tasks**


## Part 2: Behavior Cloning

### Goal

In this part, you will train and evaluate a Behavior Cloning (BC) policy that learns to imitate robot behavior from demonstration data. Your objective is to verify that the trained BC model can reproduce actions similar to the recorded demonstrations and to understand its limitations.

Behavior Cloning treats imitation learning as a supervised learning problem: given a robot state, the model predicts the action taken by the demonstrator in that state.

### Task

### Step 1: Train the BC model

Use a pre-recorded demonstration dataset ```asset/demo.npz``` to train a BC model:

```bash

python scripts/bc.py --mode train --ip <robot_ip> --epochs <epochs> --batch_size <batch_size> --lr <lr>
```
-Experiment with hyperparameters (epochs, batch_size, lr)
-Observe the training and test loss over epochs

### Step 2: Run inference on the robot
Use the trained BC model to generate actions on the robot:

```bash

python scripts/bc.py --ip <robot_ip> --mode inference
```
- The robot will replay actions predicted by the BC model
- Observe smoothness, accuracy, and timing relative to the original demonstration
- Record the EEF state trajectory using provided visualization code

### Step 3: Training and Inference Using High-Frequency Data
Repeat Step 1 and 2 using ```asset/demo_high_freq.npz``` instead,

Train the BC model using the high-frequency dataset:

```bash
python scripts/bc.py --mode train --ip <robot_ip> \
    --data asset/demo_high_freq.npz \
    --epochs <epochs> --batch_size <batch_size> --lr <lr>
```

Tune hyperparameters as needed (epochs, batch size, learning rate).

Monitor training and test loss curves.

Compare convergence behavior against the low-frequency dataset.

Run inference using the BC model trained on high-frequency data. Be sure to turn off ```wait=False``` in ```set_servo_angle``` and ```set_gripper_position``` to allow high frequency control.

```bash
python scripts/bc.py --mode inference --ip <robot_ip>
```
Observe the resulting robot motion.

Pay attention to smoothness, responsiveness, and stability.

Compare execution timing and trajectory fidelity against the original demonstrations.

Record the EEF state trajectory using provided visualization code.

### What to Record and Report
* Are the movements smooth?
* Do actions closely follow the demonstration?
* Are there any large deviations, jerks, or unexpected behavior?
* Start the robot from slightly different initial poses. Observe whether the BC model still produces reasonable behavior
* Training hyperparameters: epochs, batch_size, lr
* Loss over time. Final training and test loss
* Visualization of visited EEF states
* Video of the robot performing the BC-inferred trajectory
* Record a video of success and failures, and caption it explaining when success and failures occur

### Reflection Questions
* How closely does the BC model reproduce the original demonstrations?
* Where does the model fail or deviate most significantly? Why might that happen?
* What could go wrong if the robot starts from a pose outside the demonstration distribution?
* Why is normalization of states and actions important for BC performance? Is this the only way to pre-process data?
