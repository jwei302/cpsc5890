# policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scripts.bc import BCConvMLPPolicy
from scripts.action_vae import BCConvMLPPolicyLatent, ActionVAE 
from scripts.crop import _center_crop_bhchw
from collections import deque
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.use_latent = False

        if self.use_latent:
            print("Using VAE policy")
            # VAE Policy
            # print("scripts/asset/checkpoints/bcconv_latent_final_{8}_{128}.pt")
            # ckpt_path = os.environ.get("asset/checkpoints/bcconv_latent_final_{8}_{128}.pt")
            # print(ckpt_path)
            ckpt_path = "scripts/asset/checkpoints/bcconv_latent_final_{8}_{128}.pt"
            ckpt_path = os.path.join(os.getcwd(), ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            # print(ckpt.keys())
            self.model = ActionVAE(**ckpt["action_ae_kwargs"]).to(self.device)
            self.model.load_state_dict(ckpt["action_ae_state_dict"])    
        else:
            print("Using BC policy")
            # BC Policy 
            ckpt_path = "scripts/bc_asset/checkpoints/bcconv_latent_final_{8}_{128}.pt"
            ckpt_path = os.path.join(os.getcwd(), ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            new_dict = ckpt["policy_kwargs"]
            del new_dict["z_dim"]
            self.model = BCConvMLPPolicy(**ckpt["policy_kwargs"]).to(self.device)
            # self.model.load_state_dict(ckpt["policy_state_dict"])
            state_dict = ckpt["policy_state_dict"]
            new_state_dict = {}

            for k, v in state_dict.items():
                # Convert old keys to new keys
                new_key = k.replace("net.0.", "conv.").replace("net.1.", "norm.")
                new_state_dict[new_key] = v

            self.model.load_state_dict(new_state_dict)



        self.model.eval()
        self.stats = ckpt["stats"]
        self.obs_horizon = self.model.obs_horizon
        self.obs_dim = self.model.obs_dim
        self.pred_horizon = self.model.pred_horizon

        self.buffer = deque()
        
    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        self.buffer.clear()

    def step(self, obs: Dict[str, Any]) -> PolicyOut:
        if len(self.buffer) == 0:
            for _ in range(self.obs_horizon):
                self.buffer.append(obs)
        
        self.buffer.popleft()
        self.buffer.append(obs)

        s_mean, s_std = self.stats["s_mean"], self.stats["s_std"]
        a_mean, a_std = self.stats["a_mean"], self.stats["a_std"]
        img_mean, img_std = self.stats["img_mean"], self.stats["img_std"]
        wimg_mean, wimg_std = self.stats["wimg_mean"], self.stats["wimg_std"]

        obs_states, obs_base_rgb, obs_wrist_rgb = [], [], []
        for obs in self.buffer:
            state = (obs["joint_positions"] - s_mean) / s_std
            obs_states.append(state)
            base_rgb = (obs["base_rgb"].astype(np.float32) - img_mean) / img_std
            obs_base_rgb.append(_center_crop_bhchw(base_rgb))
            wrist_rgb = (obs["wrist_rgb"].astype(np.float32) - wimg_mean) / wimg_std
            obs_wrist_rgb.append(_center_crop_bhchw(wrist_rgb))

        obs_joints = torch.tensor(np.expand_dims(obs_states, axis=0), dtype=torch.float32, device=self.device)  # (1, obs_horizon, obs_dim)
        obs_base_rgb = torch.tensor(np.expand_dims(obs_base_rgb, axis=0), dtype=torch.float32, device=self.device)
        obs_wrist_rgb = torch.tensor(np.expand_dims(obs_wrist_rgb, axis=0), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred = self.model(obs_joints, obs_base_rgb, obs_wrist_rgb)  
            if self.use_latent:
                vae = ActionVAE(self.stats["a_mean"], self.stats["a_std"])
                action = vae.decode(pred.cpu().numpy()).squeeze(0)  # (act_dim,)
            else:
                action = pred.cpu().numpy().squeeze(0)  # (act_dim,) 
            action = action * a_std + a_mean  # unnormalize

        return PolicyOut(action=action, info=None)
    
  #####
  # normalize observation, do prediction, unnormalize action