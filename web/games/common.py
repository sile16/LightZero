import copy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from ding.config import compile_config
from ding.policy import create_policy


@dataclass
class PlayerSpec:
    player_type: str
    bot_type: Optional[str] = None
    checkpoint: Optional[str] = None
    algo: Optional[str] = None
    num_simulations: Optional[int] = None


class PolicyAgent:
    def __init__(self, config_module: str, checkpoint_path: Optional[str], num_simulations: Optional[int] = None):
        self._config_module = config_module
        self._checkpoint_path = checkpoint_path
        self._num_simulations = num_simulations
        self._policy = None
        self._device = "cpu"
        self._policy_type = None
        self._compiled_cfg = None

    def _load(self) -> None:
        if self._policy is not None:
            return
        module = __import__(self._config_module, fromlist=["main_config", "create_config"])
        cfg = copy.deepcopy(module.main_config)
        create_cfg = copy.deepcopy(module.create_config)
        cfg.policy.cuda = False
        cfg.policy.device = "cpu"
        cfg.policy.model_path = None
        if create_cfg.policy.type in {"alphazero", "gumbel_alphazero", "sampled_alphazero"}:
            cfg.policy.mcts_ctree = False
            if hasattr(cfg, "env"):
                cfg.env.alphazero_mcts_ctree = False
            model_cfg = cfg.policy.model
            if hasattr(model_cfg, "downsample") and isinstance(model_cfg.downsample, dict):
                model_cfg.downsample = bool(model_cfg.downsample.get("is_downsample", False))
            if hasattr(model_cfg, "fc_value_layers") and not hasattr(model_cfg, "value_head_hidden_channels"):
                model_cfg.value_head_hidden_channels = list(model_cfg.fc_value_layers)
            if hasattr(model_cfg, "fc_policy_layers") and not hasattr(model_cfg, "policy_head_hidden_channels"):
                model_cfg.policy_head_hidden_channels = list(model_cfg.fc_policy_layers)
            if hasattr(model_cfg, "image_channel"):
                del model_cfg.image_channel
        if hasattr(cfg, "env"):
            cfg.env.battle_mode = "self_play_mode"
        if self._num_simulations is not None:
            if hasattr(cfg.policy, "num_simulations"):
                cfg.policy.num_simulations = int(self._num_simulations)
            if hasattr(cfg.policy, "mcts") and isinstance(cfg.policy.mcts, dict):
                cfg.policy.mcts["num_simulations"] = int(self._num_simulations)
        compiled_cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
        compiled_cfg.policy.create_cfg = create_cfg
        compiled_cfg.policy.full_cfg = compiled_cfg
        self._policy = create_policy(compiled_cfg.policy, model=None, enable_field=["eval", "learn"])
        if self._checkpoint_path:
            state_dict = torch.load(self._checkpoint_path, map_location=compiled_cfg.policy.device)
            self._policy.learn_mode.load_state_dict(state_dict)
        try:
            self._policy.eval_mode.reset()
        except Exception:
            pass
        self._device = compiled_cfg.policy.device
        self._policy_type = create_cfg.policy.type
        self._compiled_cfg = compiled_cfg

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray, to_play: int, eval_obs: Optional[Dict] = None) -> int:
        self._load()
        if self._policy_type == "alphazero":
            if eval_obs is None:
                raise ValueError("alphazero requires eval_obs")
            output = self._policy.eval_mode.forward({0: eval_obs})
            return int(output[0]["action"])
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
        mask = [action_mask.tolist()]
        output = self._policy.eval_mode.forward(obs_tensor, mask, to_play=[to_play])
        return int(output[0]["action"])
