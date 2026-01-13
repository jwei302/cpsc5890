"""
Record a joint-space demonstration.
"""

import argparse, time
from xarm_lab.arm_utils import connect_arm, disconnect_arm, get_joint_angles, ArmConfig
from xarm_lab.traj_io import JointTrajectory, save_traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    arm = connect_arm(ArmConfig(ip=args.ip))

    try:
        t, q = [], []
        start = time.time()

        while time.time() - start < args.seconds:
            # TODO: append timestamp
            # TODO: append joint angles
            pass

        traj = JointTrajectory(t=t, q=q)
        save_traj(args.out, traj)

    finally:
        disconnect_arm(arm)


if __name__ == "__main__":
    main()