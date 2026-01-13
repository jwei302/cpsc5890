"""
Trajectory save/load utilities.
"""

import json
from dataclasses import dataclass
from typing import List


@dataclass
class JointTrajectory:
    t: List[float]           # timestamps (seconds)
    q: List[List[float]]     # joint angles


def save_traj(path: str, traj: JointTrajectory):
    # TODO: save trajectory to JSON
    raise NotImplementedError


def load_traj(path: str) -> JointTrajectory:
    # TODO: load trajectory from JSON
    raise NotImplementedError