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
- Learn how robot demonstrations are represented (joint states, EE axes)
- Collect and replay a real demonstration
- Analyze the demonstration data

**Tasks**
1. Collect a single demonstration on the robot  
   ```bash
   # Example: using GELLO teleoperation
   python scripts/collect_demo.py --mode gello --duration 10
   ```

    Move smoothly and stay within safe workspace

    Examine the collected data

        Identify which values correspond to joint angles, EE positions, or EE orientation

        Note timestamps and sampling frequency

    Replay the demonstration on the robot

    python scripts/replay_demo.py --traj demo1.traj

    Understand how policies map states to actions

    Implement a simple hand-coded policy

        Example: Move EE in a straight line or follow a preset trajectory

        Observe resulting behavior

Deliverable

    Notes on demonstration data and simple policy behavior

## Part 2: Behavior Cloning

In this part, you will verify that your Behavior Cloning (BC) model produces actions that match the recorded demonstrations.

# Step 1: Train the BC model

Use a pre-recorded demonstration dataset to train a BC model:

```bash

python scripts/bc.py --mode train --ip <robot_ip> --epochs <epochs> --batch_size <batch_size> --lr <lr>
```
-Experiment with hyperparameters (epochs, batch_size, lr)
-Observe the training and test loss over epochs

# Step 2: Run inference on the robot
Use the trained BC model to generate actions on the robot:

```bash

python scripts/bc.py --ip <robot_ip> --mode inference
```
- The robot will replay actions predicted by the BC model
- Observe smoothness, accuracy, and timing relative to the original demonstration

# What to record:
1. Observe robot execution:
* Are the movements smooth?
* Do actions closely follow the demonstration?
* Are there any large deviations, jerks, or unexpected behavior?
2. Check model generalization
* Start the robot from slightly different initial poses
* Observe whether the BC model still produces reasonable behavior

# What to Record and Report

* Training hyperparameters: epochs, batch_size, lr
* Final training and test loss
* Visualization of visited EEF states
* Video of the robot performing the BC-inferred trajectory

# Reflection Questions for Students
* How closely does the BC model reproduce the original demonstrations?
* Where does the model fail or deviate most significantly? Why might that happen?
* How does the choice of obs_horizon affect prediction quality?
* How does training dataset size or diversity affect generalization?
* What could go wrong if the robot starts from a pose outside the demonstration distribution?
* Why is normalization of states and actions important for BC performance?
* How might you extend this BC model to handle multi-step planning or sequences longer than the observation horizon?
