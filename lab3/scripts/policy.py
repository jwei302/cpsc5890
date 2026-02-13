# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from bc import BCConvMLPPolicy
import os
import torch

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
        ckpt_path = os.environ.get("asset/checkpoints/bcconv_final.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = BCConvMLPPolicy(**ckpt["model_kwargs"]).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        self.model.eval()
        self.stats = ckpt["stats"]

        
    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        pass

    def step(self, obs: Dict[str, Any]) -> PolicyOut:
        joints = np.asarray(obs["joint_positions"], dtype=np.float32)

        # TODO: replace with your model inference
        action = joints.copy()  # safe default: hold

        return PolicyOut(action=action, info=None)