# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

@dataclass
class PolicyOut:
    action: np.ndarray
    info: Optional[Dict[str, Any]] = None

class UniversalPolicy:
    """
    Students implement:
      - reset()
      - step(obs) -> PolicyOut

    obs (from RobotEnv) will typically include:
      obs["joint_positions"] : (D,)
      obs["base_rgb"]        : (H,W,3) uint8
      obs["wrist_rgb"]       : (H,W,3) uint8
    """
    def __init__(self):
        # TODO: load model, init buffers, etc.
        pass

    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        pass

    def step(self, obs: Dict[str, Any]) -> PolicyOut:
        joints = np.asarray(obs["joint_positions"], dtype=np.float32)

        # TODO: replace with your model inference
        action = joints.copy()  # safe default: hold

        return PolicyOut(action=action, info=None)