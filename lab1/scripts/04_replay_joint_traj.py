"""
Replay a recorded xArm trajectory file (.traj) using the SDK trajectory playback APIs.

Workflow:
  1) connect
  2) enable motion
  3) set normal mode (Mode 0)
  4) load_trajectory(<traj>)
  5) playback_trajectory()

IMPORTANT:
- Stand clear
- Keep speeds conservative (trajectory playback is controller-defined)
- Be ready to hit E-stop
"""

import argparse
from xarm.wrapper import XArmAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--traj", required=True, help="Name of the .traj file recorded in teach mode")
    args = ap.parse_args()

    # TODO: initialize XArmAPI
    arm = XArmAPI(args.ip, is_radian=True)

    # TODO: connect
    arm.connect()
    arm.set_self_collision_detection(on_off=True)

    try:
        # TODO: enable motion
        arm.motion_enable(enable=True)
        # TODO: set normal mode (Mode 0)
        arm.set_mode(0)
        # TODO: set state ready
        arm.set_state(0)

        # TODO: load_trajectory(args.traj)
        arm.load_trajectory(args.traj)
        # TODO: playback_trajectory()
        arm.playback_trajectory()

        print("[OK] Playback command sent.")

    finally:
        # TODO: disconnect
        arm.disconnect()
        pass


if __name__ == "__main__":
    main()
