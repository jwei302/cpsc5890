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
from utils.collect_demo_high_freq import policy

from scripts.bc import *

def train_bc_on_arrays(
    X_raw_train, Y_raw_train, X_raw_test, Y_raw_test,
    device,
    epochs=200,
    batch_size=256,
    lr=1e-3,
):
    # recompute normalization each (re)train on aggregated data
    X_mean, X_std = compute_norm_stats(X_raw_train)
    Y_mean, Y_std = compute_norm_stats(Y_raw_train)

    Xtr = normalize(X_raw_train, X_mean, X_std)
    Xte = normalize(X_raw_test,  X_mean, X_std)
    Ytr = normalize(Y_raw_train, Y_mean, Y_std)
    Yte = normalize(Y_raw_test,  Y_mean, Y_std)

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    test_ds  = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Yte))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    model = BCPolicy(obs_dim=Xtr.shape[1], act_dim=Ytr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
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
            print(f"Epoch {ep:03d} | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f}")

    return model, (X_mean, X_std, Y_mean, Y_std)

def rollout_dagger_collect(
    arm,
    model,
    norm_stats,
    device,
    obs_horizon,
    episodes,
    beta,
):
    """
    Runs rollouts on robot and collects (obs_stack, expert_action) pairs.
    Executes mixture policy: expert w.p. beta else learned.
    Returns: X_new_raw, Y_new_raw
    """
    X_mean, X_std, Y_mean, Y_std = norm_stats

    X_new, Y_new = [], []

    model.eval()

    for ep in range(episodes):
        # Home & randomize start (your existing logic)
        code, initial_joints = arm.get_initial_point()
        arm.set_servo_angle(angle=initial_joints, speed=20.0, wait=True, is_radian=False)

        pose = get_tcp_pose(arm)
        pose[:3] += np.random.uniform(-5, 5, size=3)
        goal_pose = pose.copy()

        goal_q = ik_from_pose(arm, goal_pose)

        arm.set_servo_angle(angle=goal_q, speed=20.0, wait=True, is_radian=True)
        arm.set_gripper_position(600, wait=True, speed=0.1)

        obs_buffer = deque(maxlen=obs_horizon)

        policy.stage = "PICK_HOVER"
        policy.seg_t0 = None
        policy.seg_p0 = None
        policy.seg_p1 = None
        policy.seg_T  = None

        for t in range(200):
            # --- read state ---
            q = get_joint_angles(arm)  # (7,)
            g = np.float32(get_gripper_position(arm))     # scalar

            state = np.concatenate([q, [g]], axis=0).astype(np.float32)  # (8,)
            obs_buffer.append(state)

            if len(obs_buffer) < obs_horizon:
                continue

            obs_stack = np.concatenate(list(obs_buffer), axis=0).astype(np.float32)  # (H*8,)

            # --- expert label ---
            # IMPORTANT: label with expert for visited states (DAgger)
            act_dim = (len(Y_mean) if Y_mean is not None else 8)
            # a_exp = expert_action(arm, q, g, goal_pose, goal_q, act_dim)
            a_exp, done = policy(arm)

            X_new.append(obs_stack)
            Y_new.append(a_exp)

            # --- mixture execution ---
            # TODO: implement the DAgger mixture policy
            # The goal: at each timestep, decide whether to execute the expert action
            # or the learned policy action, with probability beta for the expert.

            dq = action_exec[:7]
            arm.set_servo_angle(
                angle=(q + dq).tolist(),
                speed=0.5,
                wait=False,
                is_radian=True,
            )

            if len(action_exec) >= 8:
                arm.set_gripper_position(float(action_exec[7]), wait=False, speed=0.1)

            if done:
                break

        print(f"DAgger rollout episode {ep+1}/{episodes} done.")

    return np.asarray(X_new, dtype=np.float32), np.asarray(Y_new, dtype=np.float32)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference", "dagger"], default="train")
    parser.add_argument("--data", default="asset/demo_high_freq.npz")

    # BC training params
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # robot / inference params
    parser.add_argument("--ip", required=True)
    parser.add_argument("--out", default="asset/inf.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--inf_steps", type=int, default=200)

    # dagger params
    parser.add_argument("--dagger-iters", type=int, default=1)
    parser.add_argument("--dagger-rollout-episodes", type=int, default=5)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta-decay", type=float, default=0.8)  # beta_k = beta0 * decay^k

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load initial dataset (expert demos)
    Xtr0, Ytr0, Xte0, Yte0 = load_data_by_episode(
        args.data,
        H=args.obs_horizon,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    if args.mode == "train":
        print(f"Train samples: {len(Xtr0)} | Test samples: {len(Xte0)}")
        model, (X_mean, X_std, Y_mean, Y_std) = train_bc_on_arrays(
            Xtr0, Ytr0, Xte0, Yte0,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        torch.save(model.state_dict(), os.path.join("asset", "bc_policy.pt"))
        np.savez(os.path.join("asset", "bc_norm.npz"),
                 X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
        print("Model and normalization saved.")

    elif args.mode == "dagger":
        # Aggregated dataset starts with original demos
        Xtr_agg = Xtr0.copy()
        Ytr_agg = Ytr0.copy()
        Xte = Xte0
        Yte = Yte0

        # Initial train
        print("[DAgger] Initial BC training on demonstrations...")
        model, (X_mean, X_std, Y_mean, Y_std) = train_bc_on_arrays(
            Xtr_agg, Ytr_agg, Xte, Yte,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        arm = connect_arm(ArmConfig(ip=args.ip))
        try:
            clear_faults(arm)
            enable_basic_safety(arm)

            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(5000)

            for k in range(args.dagger_iters):
                beta = args.beta0 * (args.beta_decay ** k)
                print(f"\n[DAgger] Iter {k+1}/{args.dagger_iters} | beta={beta:.4f}")

                # collect new on-policy states, label by expert
                X_new, Y_new = rollout_dagger_collect(
                    arm=arm,
                    model=model,
                    norm_stats=(X_mean, X_std, Y_mean, Y_std),
                    device=device,
                    obs_horizon=args.obs_horizon,
                    episodes=args.dagger_rollout_episodes,
                    beta=beta,
                )

                print(f"[DAgger] Collected {len(X_new)} new labeled samples.")

                # aggregate
                if len(X_new) > 0:
                    Xtr_agg = np.concatenate([Xtr_agg, X_new], axis=0)
                    Ytr_agg = np.concatenate([Ytr_agg, Y_new], axis=0)

                # retrain on aggregated dataset
                print(f"[DAgger] Retraining on aggregated set: {len(Xtr_agg)} samples...")
                model, (X_mean, X_std, Y_mean, Y_std) = train_bc_on_arrays(
                    Xtr_agg, Ytr_agg, Xte, Yte,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                )

                # save each iter
                torch.save(model.state_dict(), os.path.join("asset", "dagger_policy.pt"))
                np.savez(os.path.join("asset", "dagger_norm.npz"),
                         X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
                np.savez(os.path.join("asset", "dagger_agg.npz"),
                         Xtr=Xtr_agg, Ytr=Ytr_agg, Xte=Xte, Yte=Yte)
                print("[DAgger] Saved model, norm, and aggregated dataset.")

        finally:
            disconnect_arm(arm)

    elif args.mode == "inference":
        # Load model
        # NOTE: we need obs_dim/act_dim; easiest is to infer from initial dataset shapes
        model = BCPolicy(obs_dim=Xtr0.shape[1], act_dim=Ytr0.shape[1]).to(device)
        model.load_state_dict(torch.load(os.path.join("asset", "dagger_policy.pt"), map_location=device))
        model.eval()

        norm = np.load(os.path.join("asset", "dagger_norm.npz"))
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

            print("\n=== Inference ===")

            for ep in range(args.episodes):
                code, initial_joints = arm.get_initial_point()
                arm.set_servo_angle(angle=initial_joints, speed=20.0, wait=True, is_radian=False)

                pose = get_tcp_pose(arm)
                pose[:3] += np.random.uniform(-5, 5, size=3)
                joint_angles = ik_from_pose(arm, pose)
                arm.set_servo_angle(angle=joint_angles, speed=20.0, wait=True, is_radian=True)
                arm.set_gripper_position(600, wait=True, speed=0.1)

                states, actions, eefs = [], [], []
                obs_buffer = deque(maxlen=args.obs_horizon)

                for t in range(args.inf_steps):
                    q = get_joint_angles(arm)
                    g = np.float32(get_gripper_position(arm))
                    state = np.concatenate([q, [g]], axis=0).astype(np.float32)
                    eef_state = get_tcp_pose(arm)

                    obs_buffer.append(state)
                    if len(obs_buffer) < args.obs_horizon:
                        continue

                    obs_stack = np.concatenate(list(obs_buffer), axis=0).astype(np.float32)

                    x = (obs_stack - X_mean) / X_std
                    x = torch.tensor(x, dtype=torch.float32, device=device)

                    with torch.no_grad():
                        a_norm = model(x).cpu().numpy()

                    action = a_norm * Y_std + Y_mean

                    dq = action[:7]
                    arm.set_servo_angle(
                        angle=(q + dq).tolist(),
                        speed=0.5,
                        wait=False,
                        is_radian=True,
                    )

                    if len(action) >= 8:
                        arm.set_gripper_position(float(action[7]), wait=False, speed=0.1)

                    states.append(state)
                    actions.append(action.astype(np.float32))
                    eefs.append(eef_state)

                ep_states_list.append(np.asarray(states, dtype=np.float32))
                ep_actions_list.append(np.asarray(actions, dtype=np.float32))

                plot_3d_positions(np.array(eefs)[:, :3])

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
