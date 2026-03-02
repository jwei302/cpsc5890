# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from scripts.ddpm import DiffusionPolicyUNet  # your UNet DDPM/DDIM model
from scripts.crop import _random_crop_bhchw, _center_crop_bhchw


@dataclass
class PolicyOut:
    action: np.ndarray
    info: Optional[Dict[str, Any]] = None


class UniversalPolicy:
    """
    Universal policy using your 1D UNet diffusion model (DDPM/DDIM).
    Supports state + base/wrist image sequences.
    """
    def __init__(
        self,
        obs_low_dim: int = 8,
        action_dim: int = 8,
        action_horizon: int = 16,
        obs_horizon: int = 2,
        image_type: str = "both",
        **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_low_dim = obs_low_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.image_type = image_type

        # Buffers for past observations
        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []

        # Load diffusion model
        self.model = DiffusionPolicyUNet(
            obs_low_dim=obs_low_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            obs_horizon=obs_horizon,
            image_type=image_type,
            **kwargs
        ).to(self.device)
        self.model.eval()

    def reset(self) -> None:
        """Reset buffers."""
        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []

    def step(self, obs: Dict[str, Any]) -> PolicyOut:
        """Step the policy given current observation."""

        # Convert observations
        joints = np.asarray(obs["joint_positions"], dtype=np.float32)
        img = np.asarray(obs["base_rgb"], dtype=np.float32)
        wimg = np.asarray(obs["wrist_rgb"], dtype=np.float32)

        # Crop images depending on train/eval mode
        crop_fn = _random_crop_bhchw if self.model.training else _center_crop_bhchw

        # Add batch & time dims: (B=1, T=1, C, H, W)
        img_torch = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).unsqueeze(0).float()
        wimg_torch = torch.from_numpy(wimg).permute(2,0,1).unsqueeze(0).unsqueeze(0).float()

        # Crop images
        img_torch = crop_fn(img_torch)  # (1,1,3,Hc,Wc)
        wimg_torch = crop_fn(wimg_torch)

        # Append to buffers
        self.state_buffer.append(joints.copy())
        self.img_buffer.append(img_torch)
        self.wimg_buffer.append(wimg_torch)

        # Keep only last obs_horizon steps
        self.state_buffer = self.state_buffer[-self.obs_horizon:]
        self.img_buffer = self.img_buffer[-self.obs_horizon:]
        self.wimg_buffer = self.wimg_buffer[-self.obs_horizon:]

        # Stack sequences: (B=1, T, ...)
        state_seq = torch.from_numpy(np.stack(self.state_buffer, axis=0)[None, ...]).to(self.device)  # (1,T,D)
        img_seq = torch.cat(self.img_buffer, dim=1).to(self.device)  # (1,T,3,H,W)
        wimg_seq = torch.cat(self.wimg_buffer, dim=1).to(self.device)

        # Flatten/encode condition to match UNet expected tensor
        cond = torch.cat([
            state_seq.flatten(1),      # (1, T*D)
            img_seq.flatten(1),        # (1, T*3*H*W)
            wimg_seq.flatten(1)
        ], dim=-1)  # (1, cond_dim)

        # Initialize noisy action input for inference
        x = torch.randn((1, self.action_horizon, self.action_dim), device=self.device)

        # Denoising: use your DDPM/DDIM inference function
        with torch.no_grad():
            for t in reversed(range(self.model.num_timesteps)):  # full DDPM steps
                timestep = torch.tensor([t], device=self.device, dtype=torch.long)
                x = self.model(x, timestep, cond)

        action_pred = x.cpu().numpy()  # (1, K, action_dim)
        return PolicyOut(action=action_pred)