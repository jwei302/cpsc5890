#!/usr/bin/env python3
"""
Behavior Cloning (BC) for xArm
- Train a delta-joint-angle policy from demonstrations
- Run inference on the real robot using a sliding observation window
"""

import os
import argparse
from collections import deque
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xarm_lab.arm_utils import (
    connect_arm,
    disconnect_arm,
    ArmConfig,
    get_joint_angles,
    get_tcp_pose,
    get_gripper_position,
)
from xarm_lab.safety import enable_basic_safety, clear_faults
from xarm_lab.kinematics import ik_from_pose


# ============================================================
# Dataset utilities
# ============================================================

def load_data_by_episode(path, H, test_frac=0.2, seed=0):
    """
    Load episodic data and flatten into (X, Y) pairs using
    a sliding window of length H.

    X: stacked observations  (H * obs_dim)
    Y: single-step action    (act_dim)
    """
    data = np.load(path, allow_pickle=True)
    states = data["states"]     # (E,) object array
    actions = data["actions"]   # (E,) object array
    assert len(states) == len(actions)

    E = len(states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)

    n_test = int(test_frac * E)
    test_eps = perm[:n_test]
    train_eps = perm[n_test:]

    def flatten(episode_indices):
        X, Y = [], []
        for idx in episode_indices:
            s = states[idx]     # (T, obs_dim)
            a = actions[idx]    # (T, act_dim)
            T = len(s)

            if T < H:
                continue

            for t in range(H - 1, T):
                X.append(s[t - H + 1 : t + 1].reshape(-1))
                Y.append(a[t])

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
        )

    X_train, Y_train = flatten(train_eps)
    X_test,  Y_test  = flatten(test_eps)

    return X_train, Y_train, X_test, Y_test


# ============================================================
# TODO: Compute normalization stats
# ============================================================
def compute_norm_stats(X, eps=1e-8):
    """
    Student TODO:
    - Compute mean and std along each feature dimension
    - Make sure std is never smaller than eps to avoid division by zero
    """
    # -------------------------------
    # TODO: implement
    mean = None  # replace None
    std  = None  # replace None
    # -------------------------------
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std



# ============================================================
# TODO: Policy network skeleton
# ============================================================
class BCPolicy(nn.Module):
    """
    Student TODO:
    Implement a simple MLP policy mapping observations to actions
    Suggested:
        - Input layer: obs_dim
        - Hidden layers: 1-2 layers with ReLU
        - Output layer: act_dim
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # -------------------------------
        # TODO: define network layers
        self.net = None
        # -------------------------------

    def forward(self, x):
        return self.net(x)


# ============================================================
# Training / evaluation helpers
# ============================================================
def evaluate(model, loader, device):
    """
    Compute mean squared error (MSE) over a dataset.

    Student TODO:
    Fill in the loss calculation only. The forward pass is provided.
    """
    model.eval()
    mse, n = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass (provided)
            pred = model(x)

            # -------------------------------
            # TODO: Compute batch squared error
            # Replace the next line with the actual computation
            batch_mse = None
            # -------------------------------

            mse += batch_mse
            n += len(x)

    return mse / n


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--data", default="asset/demo.npz")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--out", default="asset/inf.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obs_horizon", type=int, default=1)
    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility & device
    # --------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load and normalize dataset
    # --------------------------------------------------------
    Xtr, Ytr, Xte, Yte = load_data_by_episode(
        args.data,
        H=args.obs_horizon,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    X_mean, X_std = compute_norm_stats(Xtr)
    Y_mean, Y_std = compute_norm_stats(Ytr)

    print("X mean/std:", X_mean, X_std)
    print("Y mean/std:", Y_mean, Y_std)

    Xtr = normalize(Xtr, X_mean, X_std)
    Xte = normalize(Xte, X_mean, X_std)
    Ytr = normalize(Ytr, Y_mean, Y_std)
    Yte = normalize(Yte, Y_mean, Y_std)

    # ========================================================
    # TRAINING
    # ========================================================
    if args.mode == "train":

        print(f"Train samples: {len(Xtr)} | Test samples: {len(Xte)}")

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Yte)),
            batch_size=args.batch_size,
        )

        model = BCPolicy(Xtr.shape[1], Ytr.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        # Track loss over epochs
        train_losses = []
        test_losses = []

        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate at the end of the epoch
            train_mse = evaluate(model, train_loader, device)
            test_mse = evaluate(model, test_loader, device)

            train_losses.append(train_mse)
            test_losses.append(test_mse)

            if ep % 5 == 0 or ep == 1:
                print(
                    f"Epoch {ep:03d} | "
                    f"Train MSE: {train_mse:.6f} | "
                    f"Test MSE: {test_mse:.6f}"
                )

        # Save artifacts
        torch.save(model.state_dict(), "asset/bc_policy.pt")
        np.savez(
            "asset/bc_norm.npz",
            X_mean=X_mean, X_std=X_std,
            Y_mean=Y_mean, Y_std=Y_std,
        )

        # ---------------- Plot training/test loss ----------------
        plt.figure(figsize=(8,5))
        plt.plot(range(1, args.epochs+1), train_losses, label="Train MSE")
        plt.plot(range(1, args.epochs+1), test_losses, label="Test MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Behavior Cloning Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("Model and normalization saved.")

    # ========================================================
    # INFERENCE
    # ========================================================
    else:
        model = BCPolicy(Xtr.shape[1], Ytr.shape[1]).to(device)
        model.load_state_dict(torch.load("asset/bc_policy.pt", map_location=device))
        model.eval()

        norm = np.load("asset/bc_norm.npz")
        X_mean, X_std = norm["X_mean"], norm["X_std"]
        Y_mean, Y_std = norm["Y_mean"], norm["Y_std"]

        arm = connect_arm(ArmConfig(ip=args.ip))

        ep_states_list = []
        ep_actions_list = []

        try:
            clear_faults(arm)
            enable_basic_safety(arm)

            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(5000)

            print("\n=== BC Inference on Robot ===")

            for ep in range(args.episodes):
                # Home robot
                _, init_joints = arm.get_initial_point()
                arm.set_servo_angle(angle=init_joints, speed=20.0, wait=True, is_radian=False)

                # Randomized initial pose
                pose = get_tcp_pose(arm)
                pose[:3] += np.random.uniform(-5, 5, size=3)
                joints = ik_from_pose(arm, pose)

                arm.set_servo_angle(angle=joints, speed=20.0, wait=True, is_radian=True)
                arm.set_gripper_position(600, wait=True, speed=0.1)

                print(f"Episode {ep + 1}: start pose {pose}")

                obs_buffer = deque(maxlen=args.obs_horizon)
                states, eef_states, actions = [], [], []

                for _ in range(7):  # safety-limited horizon
                    q = get_joint_angles(arm)
                    g = get_gripper_position(arm)
                    state = np.concatenate([q, [g]])

                    obs_buffer.append(state)
                    if len(obs_buffer) < args.obs_horizon:
                        continue

                    obs = np.concatenate(obs_buffer)
                    x = torch.tensor((obs - X_mean) / X_std, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        a_norm = model(x).cpu().numpy()

                    action = a_norm * Y_std + Y_mean

                    arm.set_servo_angle(
                        angle=(q + action[:7]).tolist(),
                        speed=0.5,
                        wait=True,
                        is_radian=True,
                    )
                    arm.set_gripper_position(action[-1], wait=True, speed=0.1)

                    states.append(state)
                    actions.append(action)
                    eef_states.append(get_tcp_pose(arm))

                # Convert to NumPy array (shape: N x 6)
                eef_states_np = np.array(eef_states)  # each element is [x, y, z, roll, pitch, yaw]

                # Extract first 3 axes: x, y, z
                eef_xyz = eef_states_np[:, :3]

                # Optional: 3D scatter plot
                fig = plt.figure(figsize=(7,7))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(eef_xyz[:, 0], eef_xyz[:, 1], eef_xyz[:, 2], c=np.arange(len(eef_xyz)), cmap='viridis')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title("EEF Position in 3D")
                plt.show()

                ep_states_list.append(np.asarray(states, np.float32))
                ep_actions_list.append(np.asarray(actions, np.float32))

            np.savez(
                args.out,
                states=np.array(ep_states_list, dtype=object),
                actions=np.array(ep_actions_list, dtype=object),
                action_type="delta_joint_angles",
                unit="radians",
            )

            print(f"\nDataset saved to {args.out}")

        finally:
            disconnect_arm(arm)


if __name__ == "__main__":
    main()
