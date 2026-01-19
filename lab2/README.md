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

---

## Agenda & Procedure

### Part 1: Understanding Demonstrations
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

### Part 2: Behavior Cloning

In this part, you will verify that your Behavior Cloning (BC) model produces actions that match the recorded demonstrations.

This step is critical for building intuition about:

how demonstrations are represented as data (joint states, EE positions, orientations)

mapping states to actions via a policy

how model predictions compare to actual robot trajectories

# Procedure

Use a pre-recorded demonstrations to train a BC model.

```bash

python scripts/bc.py --ip <robot_ip>
```
