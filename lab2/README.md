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

### Part 1: Understanding Demonstrations (≈45 mins)
**Goals**
- Learn how robot demonstrations are represented (joint states, EE axes)
- Collect and replay a real demonstration
- Analyze the demonstration data

**Tasks**
1. Collect a single demonstration on the robot  
   ```bash
   # Example: using GELLO teleoperation
   python scripts/collect_demo.py --mode gello --duration 10

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

Part 2: Behavior Cloning (≈1 hr)

Goals

    Train a policy from demonstration data

    Evaluate performance in simulation first for safety

Procedure

    Dataset

        Use pre-collected pick-and-place demonstrations (provided by TAs)

        Visualize trajectories and states

    Data Analysis

        Replay individual demonstrations in simulation

        Plot distribution of visited states

    Implementation

        Fill in missing code in the BC template

        Train the BC model on the dataset

        Visualize training loss over time

    Evaluation

        Test trained model from multiple start states in simulation

        Observe successes and failures

        Develop evaluation metrics (e.g., success rate, EE distance error)

        Record video clips of both success and failure cases with captions

Deliverable

    Video demonstrating BC performance with captions
