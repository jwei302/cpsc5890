import argparse
from xarm_lab.arm_utils import connect_arm, disconnect_arm, ArmConfig
from xarm_lab.traj_io import load_traj
from xarm_lab.safety import enable_basic_safety, clear_faults

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--traj", required=True)
    args = ap.parse_args()

    traj = load_traj(args.traj)
    arm = connect_arm(ArmConfig(ip=args.ip))

    try:
        # TODO: clear faults
        # TODO: enable safety

        for qi in traj.q:
            # TODO: send joint command (low speed)
            pass

    finally:
        disconnect_arm(arm)

if __name__ == "__main__":
    main()