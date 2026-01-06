from typing import Dict

import numpy as np
from zoo.board_games.pig.envs.pig_env import PigEnv

from web.games.common import PlayerSpec, PolicyAgent

SUPPORTED_POLICY_CONFIGS = {
    "stochastic_muzero": "zoo.board_games.pig.config.pig_stochastic_muzero_config",
}


class PigSession:
    def __init__(self, players: Dict[int, PlayerSpec], auto_play: bool = True):
        cfg = PigEnv.default_config()
        cfg.battle_mode = "self_play_mode"
        self.env = PigEnv(cfg)
        self.env.reset(start_player_index=0)
        self.players = players
        self.auto_play = auto_play
        self.done = False
        self.winner = None
        self.last_action = None
        self._policy_cache = {}

    def _player_spec(self) -> PlayerSpec:
        return self.players[int(self.env.current_player)]

    def _bot_action(self, spec: PlayerSpec) -> int:
        if spec.player_type == "bot":
            if spec.bot_type == "hold_at_20":
                return 1 if self.env.turn_score >= 20 else 0
            if spec.bot_type == "random":
                if self.env.turn_score > 0 and np.random.random() < 0.5:
                    return 1
                return 0
            return 0
        if spec.player_type == "model":
            if not spec.algo or spec.algo not in SUPPORTED_POLICY_CONFIGS:
                raise ValueError(f"Unsupported model algo: {spec.algo}")
            key = (spec.algo, spec.checkpoint)
            if key not in self._policy_cache:
                self._policy_cache[key] = PolicyAgent(
                    SUPPORTED_POLICY_CONFIGS[spec.algo],
                    spec.checkpoint,
                    num_simulations=spec.num_simulations,
                )
            obs = self.env._get_obs()
            return self._policy_cache[key].select_action(
                obs["observation"],
                np.array(obs["action_mask"], dtype=np.int8),
                int(obs["to_play"]),
                eval_obs=None,
            )
        raise ValueError(f"Unsupported player_type: {spec.player_type}")

    def _apply_action(self, action: int) -> None:
        timestep = self.env.step(action)
        self.last_action = int(action)
        self.done = bool(timestep.done)
        if self.done:
            self.winner = int(self.env.winner)

    def advance(self, max_steps: int = 200) -> None:
        steps = 0
        while not self.done and steps < max_steps:
            spec = self._player_spec()
            if spec.player_type == "human":
                break
            action = self._bot_action(spec)
            self._apply_action(action)
            steps += 1
            if not self.auto_play:
                break

    def apply_human_action(self, action: int) -> None:
        spec = self._player_spec()
        if spec.player_type != "human":
            raise ValueError("Current player is not human")
        self._apply_action(action)
        if self.auto_play and not self.done:
            self.advance()

    def state(self) -> Dict:
        obs = self.env._get_obs()
        return {
            "game": "pig",
            "scores": list(self.env.scores),
            "turn_score": int(self.env.turn_score),
            "current_player": int(self.env.current_player),
            "legal_actions": list(self.env.legal_actions),
            "action_mask": list(obs["action_mask"]),
            "done": self.done,
            "winner": self.winner,
            "last_action": self.last_action,
            "players": {
                1: self.players[1].__dict__,
                2: self.players[2].__dict__,
            },
        }


class PigGame:
    name = "pig"
    label = "Pig"
    supported_policy_configs = SUPPORTED_POLICY_CONFIGS

    def new_session(self, players: Dict[int, PlayerSpec], auto_play: bool = True) -> PigSession:
        session = PigSession(players, auto_play=auto_play)
        if auto_play:
            session.advance()
        return session
