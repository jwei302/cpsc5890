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

***Goals***

In this part, you will verify that your Behavior Cloning (BC) model produces actions that match the recorded demonstrations.

**Tasks**

### Step 1: Train the BC model

Use a pre-recorded demonstration dataset to train a BC model:

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
