from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler

from scripts.unet import DiffusionPolicyUNet


@dataclass
class PolicyOut:
    action: np.ndarray
    info: Optional[Dict[str, Any]] = None


class UniversalPolicy:
    """
    Diffusion policy used by scripts.run_robot.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: str = "config/lift_cube.yaml",
        obs_low_dim: int = 8,
        action_dim: int = 8,
        action_horizon: int = 16,
        obs_horizon: int = 1,
        image_type: str = "both",
        num_diffusion_steps: int = 100,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        **kwargs: Any,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eta = float(eta)

        cfg = self._load_config(config_path)
        ckpt_path = self._resolve_checkpoint_path(checkpoint_path, cfg)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        meta = ckpt.get("meta", {})
        self.obs_low_dim = int(meta.get("obs_dim", obs_low_dim))
        self.action_dim = int(meta.get("action_dim", action_dim))
        self.action_horizon = int(meta.get("pred_horizon", action_horizon))
        self.obs_horizon = int(meta.get("obs_horizon", obs_horizon))
        self.image_type = str(cfg.get("image_type", image_type))
        self.num_diffusion_steps = int(meta.get("num_diffusion_steps", num_diffusion_steps))
        if num_inference_steps is None:
            num_inference_steps = int(cfg.get("num_inference_steps", self.num_diffusion_steps))
        self.num_inference_steps = int(num_inference_steps)

        self.model = DiffusionPolicyUNet(
            obs_low_dim=self.obs_low_dim,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            obs_horizon=self.obs_horizon,
            image_type=self.image_type,
            **kwargs,
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        stats = ckpt["stats"]
        self.obs_mean = self._to_tensor(stats["obs"]["mean"])
        self.obs_std = self._to_tensor(stats["obs"]["std"])
        self.act_min = self._to_tensor(stats["act"]["min"])
        self.act_max = self._to_tensor(stats["act"]["max"])
        self.img_ext_mean = self._to_tensor(stats["img_ext"]["mean"])
        self.img_ext_std = self._to_tensor(stats["img_ext"]["std"])
        self.img_wst_mean = self._to_tensor(stats["img_wst"]["mean"])
        self.img_wst_std = self._to_tensor(stats["img_wst"]["std"])

        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
        )
        self.scheduler.set_timesteps(self.num_inference_steps)

        self.state_buffer: List[np.ndarray] = []
        self.img_buffer: List[np.ndarray] = []
        self.wimg_buffer: List[np.ndarray] = []

    def _to_tensor(self, x: Any) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        p = Path(config_path)
        if not p.exists():
            return {}
        with p.open("r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _resolve_checkpoint_path(
        checkpoint_path: Optional[str], cfg: Dict[str, Any]
    ) -> str:
        candidates: List[Path] = []

        if checkpoint_path:
            candidates.append(Path(checkpoint_path))

        load_name = cfg.get("load_model_name")
        save_name = cfg.get("save_model_name")

        if load_name:
            candidates.extend(
                [
                    Path(load_name),
                    Path("asset/policy") / load_name,
                    Path("asset/policy") / f"{load_name}.pth",
                    Path("asset/policy") / f"DiT_state_{load_name}.pth",
                    Path("asset/policy") / f"DiT_state_{load_name}_final.pth",
                ]
            )

        if save_name:
            candidates.extend(
                [
                    Path(save_name),
                    Path("asset/policy") / save_name,
                    Path("asset/policy") / f"{save_name}.pth",
                    Path("asset/policy") / f"DiT_state_{save_name}.pth",
                    Path("asset/policy") / f"DiT_state_{save_name}_final.pth",
                ]
            )

        for c in candidates:
            if c.exists():
                return str(c)

        pretty = "\n".join(str(c) for c in candidates) or "(no candidates)"
        raise FileNotFoundError(
            "Could not find diffusion checkpoint. Tried:\n"
            f"{pretty}\n"
            "Pass `checkpoint_path=...` when constructing UniversalPolicy."
        )

    def _padded_history(self, hist: List[np.ndarray]) -> List[np.ndarray]:
        if not hist:
            raise RuntimeError("Policy history is empty.")
        if len(hist) >= self.obs_horizon:
            return hist[-self.obs_horizon :]
        pad_count = self.obs_horizon - len(hist)
        return [hist[0].copy() for _ in range(pad_count)] + hist

    def reset(self) -> None:
        # TODO: reset hidden state / buffers
        PolicyOut(action = np.deg2rad(np.array([-0.2,-45.4,-0.3,21.8,0.4,67.2,0.6])))
        self.state_buffer = []
        self.img_buffer = []
        self.wimg_buffer = []

    def step(self, obs: Dict[str, Any]) -> np.ndarray:
        joints = np.asarray(obs["joint_positions"], dtype=np.float32)
        img_ext = np.asarray(obs["base_rgb"], dtype=np.uint8)
        img_wst = np.asarray(obs["wrist_rgb"], dtype=np.uint8)

        if joints.shape[-1] != self.obs_low_dim:
            raise ValueError(
                f"joint_positions dim mismatch: got {joints.shape[-1]}, expected {self.obs_low_dim}"
            )

        self.state_buffer.append(joints.copy())
        self.img_buffer.append(img_ext.copy())
        self.wimg_buffer.append(img_wst.copy())
        self.state_buffer = self.state_buffer[-self.obs_horizon :]
        self.img_buffer = self.img_buffer[-self.obs_horizon :]
        self.wimg_buffer = self.wimg_buffer[-self.obs_horizon :]

        state_hist = np.stack(self._padded_history(self.state_buffer), axis=0)  # (T, D)
        ext_hist = np.stack(self._padded_history(self.img_buffer), axis=0)       # (T, H, W, C)
        wst_hist = np.stack(self._padded_history(self.wimg_buffer), axis=0)      # (T, H, W, C)

        obs_cond = torch.from_numpy(state_hist).to(self.device, dtype=torch.float32).unsqueeze(0)
        obs_cond = (obs_cond - self.obs_mean.view(1, 1, -1)) / self.obs_std.view(1, 1, -1)

        img_ext_cond = torch.from_numpy(ext_hist).to(self.device, dtype=torch.float32) / 255.0
        img_ext_cond = img_ext_cond.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
        img_ext_cond = (img_ext_cond - self.img_ext_mean.view(1, 1, -1, 1, 1)) / self.img_ext_std.view(
            1, 1, -1, 1, 1
        )

        img_wst_cond = torch.from_numpy(wst_hist).to(self.device, dtype=torch.float32) / 255.0
        img_wst_cond = img_wst_cond.permute(0, 3, 1, 2).unsqueeze(0)
        img_wst_cond = (img_wst_cond - self.img_wst_mean.view(1, 1, -1, 1, 1)) / self.img_wst_std.view(
            1, 1, -1, 1, 1
        )

        naction = torch.randn(
            (1, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            for t in self.scheduler.timesteps:
                noise_pred = self.model(
                    noisy_actions=naction,
                    timesteps=t,
                    observations=obs_cond,
                    img_ext=img_ext_cond if self.image_type in ("both", "external") else None,
                    img_wst=img_wst_cond if self.image_type in ("both", "wrist") else None,
                )
                naction = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=naction,
                    eta=self.eta,
                ).prev_sample

        naction = naction.clamp(-1.0, 1.0)
        naction = (naction + 1.0) * 0.5
        naction = naction * (self.act_max.view(1, 1, -1) - self.act_min.view(1, 1, -1)) + self.act_min.view(
            1, 1, -1
        )
        return naction[0].detach().cpu().numpy().astype(np.float32)