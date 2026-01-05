"""
Pig Dice Game Environment for LightZero.

Rules:
- Two players take turns.
- On your turn, roll a 6-sided die:
  - Roll 1: Lose all turn points, turn ends.
  - Roll 2-6: Add to turn score. Choose to "roll" again or "hold".
- "Hold": Bank turn points to total score, end turn.
- First player to reach target_score (default 100) wins.

This is a stochastic 2-player game suitable for Stochastic MuZero.
"""

import copy
from typing import List

import gymnasium as gym
import numpy as np
from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from easydict import EasyDict


@ENV_REGISTRY.register('pig')
class PigEnv(BaseEnv):
    """Pig dice game environment."""

    config = dict(
        env_id="Pig",
        # Target score to win
        target_score=100,
        # Maximum steps per episode (prevents infinite games)
        max_episode_steps=200,
        # Battle mode: 'self_play_mode' or 'play_with_bot_mode'
        battle_mode='self_play_mode',
        # Bot type for play_with_bot_mode
        bot_action_type='random',  # 'random' or 'hold_at_20'
        # Starting player index (0 = model first, 1 = bot first in play_with_bot_mode)
        start_player_index=0,
        # Observation type
        channel_last=False,
        # Stochastic MuZero settings
        # Chance space: 7 outcomes (0 = no roll, 1-6 = die faces)
        chance_space_size=7,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None):
        default_config = self.default_config()
        if cfg is not None:
            default_config.update(cfg)
        self._cfg = default_config

        self.target_score = self._cfg.target_score
        self.max_episode_steps = self._cfg.max_episode_steps
        self.battle_mode = self._cfg.battle_mode
        self.bot_action_type = self._cfg.bot_action_type
        self._default_start_player_index = self._cfg.get('start_player_index', 0)

        # Action space: 0 = roll, 1 = hold
        self.action_space_size = 2
        self.total_num_actions = 2

        # Stochastic MuZero support
        self.chance_space_size = self._cfg.chance_space_size  # 0 = no roll, 1-6 = die faces
        self.chance = 0  # Last chance outcome (0 = no roll, 1-6 = die faces)

        # Players
        self.players = [1, 2]

        self._observation_space = gym.spaces.Box(
            low=0, high=self.target_score, shape=(5,), dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(2)
        self._reward_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, start_player_index=None, init_state=None, **kwargs):
        """Reset the game."""
        # Use default from config if not specified
        if start_player_index is None:
            start_player_index = self._default_start_player_index

        self.scores = [0, 0]  # Player 1 and 2 total scores
        self.turn_score = 0   # Current turn's accumulated score
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        self.done = False
        self.winner = -1
        self.chance = 0  # Reset chance (no roll)
        self._step_count = 0  # Track steps for max_episode_steps

        # In play_with_bot_mode, if bot goes first (start_player_index=1), let bot play
        if self.battle_mode == 'play_with_bot_mode' and start_player_index == 1:
            while self._current_player == 2 and not self.done:
                bot_action = self._bot_action()
                self._player_step(bot_action)

        obs = self._get_obs()
        return obs

    def _get_obs(self):
        """Get observation dict."""
        # Observation: [my_score, opponent_score, turn_score, am_i_player_1, can_hold]
        my_idx = 0 if self._current_player == 1 else 1
        opp_idx = 1 - my_idx

        # Normalize scores
        obs_array = np.array([
            self.scores[my_idx] / self.target_score,
            self.scores[opp_idx] / self.target_score,
            self.turn_score / self.target_score,
            1.0 if self._current_player == 1 else 0.0,
            1.0 if self.turn_score > 0 else 0.0,  # Can only hold if have turn points
        ], dtype=np.float32)

        # Action mask: can always roll (0), can only hold (1) if turn_score > 0
        action_mask = np.array([1, 1 if self.turn_score > 0 else 0], dtype=np.int8)

        if self.battle_mode == 'self_play_mode':
            to_play = self._current_player
        else:
            to_play = -1

        return {
            'observation': obs_array,
            'action_mask': action_mask,
            'to_play': to_play,
            'chance': self.chance,  # Stochastic MuZero support
        }

    def step(self, action):
        """Execute action."""
        if self.battle_mode == 'self_play_mode':
            timestep = self._player_step(action)
            if timestep.done:
                # Calculate reward from player 1's perspective
                timestep.info['eval_episode_return'] = timestep.reward if self._current_player == 1 else -timestep.reward
            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            # Player 1's turn
            timestep = self._player_step(action)
            if timestep.done:
                timestep.obs['to_play'] = -1
                return timestep

            # If turn switched to player 2 (bot), let bot play until turn ends
            while self._current_player == 2 and not self.done:
                bot_action = self._bot_action()
                timestep = self._player_step(bot_action)

            timestep.obs['to_play'] = -1
            if timestep.done:
                # From player 1's perspective
                timestep.info['eval_episode_return'] = 1.0 if self.winner == 1 else -1.0
                timestep = timestep._replace(reward=timestep.info['eval_episode_return'])
            return timestep

    def _player_step(self, action):
        """Execute a single player action."""
        reward = 0.0
        info = {}

        self._step_count += 1
        my_idx = 0 if self._current_player == 1 else 1

        # Check for max episode steps (game ends in draw)
        if self._step_count >= self.max_episode_steps:
            self.done = True
            self.winner = -1  # Draw
            obs = self._get_obs()
            info['eval_episode_return'] = 0.0  # Draw
            return BaseEnvTimestep(obs, np.float32(0.0), self.done, info)

        if action == 0:  # Roll
            # Roll the die (stochastic event)
            die_roll = np.random.randint(1, 7)  # 1-6
            self.chance = die_roll  # Store as 1-6; 0 reserved for no roll

            if die_roll == 1:
                # Bust! Lose turn points
                self.turn_score = 0
                self._switch_player()
            else:
                # Add to turn score
                self.turn_score += die_roll

        elif action == 1:  # Hold
            # Bank turn points
            self.scores[my_idx] += self.turn_score
            self.turn_score = 0
            self.chance = 0  # No dice roll on hold

            # Check for win
            if self.scores[my_idx] >= self.target_score:
                self.done = True
                self.winner = self._current_player
                reward = 1.0
            else:
                self._switch_player()

        obs = self._get_obs()

        if self.done:
            info['eval_episode_return'] = reward

        return BaseEnvTimestep(obs, np.float32(reward), self.done, info)

    def _switch_player(self):
        """Switch to the other player."""
        self._current_player = 2 if self._current_player == 1 else 1

    def _bot_action(self):
        """Get bot action."""
        if self.bot_action_type == 'random':
            if self.turn_score > 0 and np.random.random() < 0.5:
                return 1  # Hold
            return 0  # Roll
        elif self.bot_action_type == 'hold_at_20':
            # Simple strategy: hold when turn_score >= 20
            if self.turn_score >= 20:
                return 1  # Hold
            return 0  # Roll
        else:
            return 0  # Default: always roll

    @property
    def current_player(self):
        return self._current_player

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @property
    def current_player_index(self):
        return 0 if self._current_player == 1 else 1

    @property
    def legal_actions(self):
        """Get legal actions."""
        if self.turn_score > 0:
            return [0, 1]  # Can roll or hold
        return [0]  # Can only roll

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return "LightZero Pig Dice Game Env"

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.battle_mode = 'play_with_bot_mode'
        return [cfg for _ in range(evaluator_env_num)]
