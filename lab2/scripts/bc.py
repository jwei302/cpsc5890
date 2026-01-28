import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.utils.data import DataLoader, TensorDataset


from xarm_lab.arm_utils import connect_arm, disconnect_arm, ArmConfig, get_joint_angles, get_tcp_pose, get_gripper_position
from xarm_lab.safety import enable_basic_safety, clear_faults
from xarm_lab.kinematics import ik_from_pose
from utils.plot import plot_3d_positions

# -----------------------------
# Load + flatten dataset
# -----------------------------

def load_data_by_episode(path, H, test_frac=0.2, seed=0):
    data = np.load(path, allow_pickle=True)

    states = data["states"]    # (E,) object array
    actions = data["actions"]  # (E,) object array

    assert len(states) == len(actions)

    E = len(states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)

    n_test = int(test_frac * E)
    test_eps = perm[:n_test]
    train_eps = perm[n_test:]

    def flatten(episode_indices, H):
        X, Y = [], []
        for i in episode_indices:
            s = states[i]   # (T, obs_dim)
            a = actions[i]  # (T, act_dim)
            # s = states[i][..., :-1]          # (T, 7)  joint angles only
            # s = angle_to_continuous(q)       # (T, 14)
            # a = actions[i][..., :-1]  # (T, act_dim)

            T = len(s)
            if T < H:
                continue
            for t in range(H - 1, T):
                X.append(s[t-H+1:t+1].reshape(-1))  # (H*obs_dim,)
                Y.append(a[t])
        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
        )

    X_train, Y_train = flatten(train_eps, H)
    X_test, Y_test   = flatten(test_eps, H)

    return X_train, Y_train, X_test, Y_test

def compute_norm_stats(X, eps=1e-8):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.maximum(std, eps)
    return mean, std

def normalize(X, mean, std):
    return (X - mean) / std

# -----------------------------
# BC policy
# -----------------------------

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Train / eval
# -----------------------------

def evaluate(model, loader, device):
    model.eval()
    mse = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse += torch.sum((pred - y) ** 2).item()
            n += len(x)
    return mse / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--data", default="asset/demo.npz")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--out", default="asset/inf.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--inf_steps", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    Xtr, Ytr, Xte, Yte = load_data_by_episode(
        args.data,
        H=args.obs_horizon,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # Normalize using TRAIN statistics only
    X_mean, X_std = compute_norm_stats(Xtr)
    Y_mean, Y_std = compute_norm_stats(Ytr)

    print(X_mean, X_std)

    print(Y_mean, Y_std)

    Xtr = normalize(Xtr, X_mean, X_std)
    Xte = normalize(Xte, X_mean, X_std)

    Ytr = normalize(Ytr, Y_mean, Y_std)
    Yte = normalize(Yte, Y_mean, Y_std)

    if args.mode == "train":

        print(f"Train samples: {len(Xtr)} | Test samples:  {len(Xte)}")

        train_ds = TensorDataset(
            torch.from_numpy(Xtr), torch.from_numpy(Ytr)
        )
        test_ds = TensorDataset(
            torch.from_numpy(Xte), torch.from_numpy(Yte)
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        # Model
        model = BCPolicy(obs_dim=Xtr.shape[1], act_dim=Ytr.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        # Train
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ep % 5 == 0 or ep == 1:
                train_mse = evaluate(model, train_loader, device)
                test_mse = evaluate(model, test_loader, device)
                print(
                    f"Epoch {ep:03d} | "
                    f"Train MSE: {train_mse:.6f} | "
                    f"Test MSE: {test_mse:.6f}"
                )

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join("asset", "bc_policy.pt")
        )

        # Save normalization stats
        np.savez(
            os.path.join("asset", "bc_norm.npz"),
            X_mean=X_mean,
            X_std=X_std,
            Y_mean=Y_mean,
            Y_std=Y_std,
        )

        print("Model and normalization saved.")

    elif args.mode == "inference":

        # Load model
        model = BCPolicy(obs_dim=Xtr.shape[1], act_dim=Ytr.shape[1]).to(device)
        model.load_state_dict(torch.load(os.path.join("asset", "bc_policy.pt"), map_location=device))
        model.eval()

        # Load normalization
        norm = np.load(os.path.join("asset", "bc_norm.npz"))
        X_mean, X_std = norm["X_mean"], norm["X_std"]
        Y_mean, Y_std = norm["Y_mean"], norm["Y_std"]

        arm = connect_arm(ArmConfig(ip=args.ip))

        ep_states_list = []
        ep_actions_list = []

        try:
            clear_faults(arm)
            enable_basic_safety(arm)

            code = arm.set_gripper_mode(0)
            code = arm.set_gripper_enable(True)
            code = arm.set_gripper_speed(5000)

            print("\n=== Pick-and-Place Demonstration (IK → Δq) ===")

            for ep in range(args.episodes):

                code, initial_joints = arm.get_initial_point()
                arm.set_servo_angle(
                    angle=initial_joints,
                    speed=20.0,
                    wait=True,
                    is_radian=False
                )
                pose = get_tcp_pose(arm)
                pose[:3] += np.random.uniform(-5, 5, size=3)
                joint_angles = ik_from_pose(arm, pose)
                arm.set_servo_angle(
                    angle=joint_angles,
                    speed=20.0,
                    wait=True,
                    is_radian=True
                )
                arm.set_gripper_position(600, wait=True, speed=0.1)

                print(f"Episode {ep+1}: robot homed to {np.asarray(pose, dtype=float)}")

                states = []
                actions = []
                eefs = []

                obs_buffer = deque(maxlen=args.obs_horizon)

                for t in range(args.inf_steps):  # fixed horizon (safety)

                    # ---- Read state ----
                    q = get_joint_angles(arm)              # (7,)
                    # state = angle_to_continuous(q)         # (14,)
                    g = get_gripper_position(arm)  # scalar
                    state = np.concatenate([q, [g]])  # (8,)
                    eef_state = get_tcp_pose(arm)
                    # state = np.concatenate([q])  # (8,)

                    obs_buffer.append(state)

                    if len(obs_buffer) < args.obs_horizon:
                        continue   # wait until buffer is full

                    obs_stack = np.concatenate(list(obs_buffer), axis=0)  # (H*8,)

                    x = (obs_stack - X_mean) / X_std
                    x = torch.tensor(x, dtype=torch.float32).to(device)

                    # ---- Predict action ----
                    with torch.no_grad():
                        a_norm = model(x).cpu().numpy()

                    # ---- Unnormalize ----
                    action = a_norm * Y_std + Y_mean

                    dq = action[:7]
                    dg = int(action[7] >= 0.5)

                    # ---- Execute ----
                    arm.set_servo_angle(
                        angle=(q + dq).tolist(),
                        speed=0.5,
                        wait=True,
                        is_radian=True,
                    )

                    arm.set_gripper_position(action[-1], wait=True, speed=0.1)

                    states.append(state)
                    actions.append(action)
                    eefs.append(eef_state)

                ep_states_list.append(np.asarray(states, dtype=np.float32))
                ep_actions_list.append(np.asarray(actions, dtype=np.float32))

                plot_3d_positions(np.array(eefs)[:,:3])

            np.savez(
                args.out,
                states=np.array(ep_states_list, dtype=object),   # (E,) each item (Ti,7)
                actions=np.array(ep_actions_list, dtype=object), # (E,) each item (Ti,8)
                action_type="delta_joint_angles",
                unit="radians"
            )

            print(f"\nDataset saved to {args.out}")

        finally:
            disconnect_arm(arm)


if __name__ == "__main__":
    main()