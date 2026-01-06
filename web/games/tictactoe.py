import copy
from typing import Dict

import numpy as np
from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv

from web.games.common import PlayerSpec, PolicyAgent

SUPPORTED_POLICY_CONFIGS = {
    "stochastic_muzero": "zoo.board_games.tictactoe.config.tictactoe_stochastic_muzero_bot_mode_config",
    "gumbel_muzero": "zoo.board_games.tictactoe.config.tictactoe_gumbel_muzero_bot_mode_config",
    "muzero": "zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config",
    "alphazero": "zoo.board_games.tictactoe.config.tictactoe_alphazero_bot_mode_config",
}


class TicTacToeSession:
    def __init__(self, players: Dict[int, PlayerSpec], auto_play: bool = True):
        cfg = TicTacToeEnv.default_config()
        cfg.battle_mode = "self_play_mode"
        cfg.channel_last = False
        cfg.scale = True
        cfg.agent_vs_human = False
        self.env = TicTacToeEnv(cfg)
        self.env.reset(start_player_index=0)
        self.players = players
        self.auto_play = auto_play
        self.done = False
        self.winner = None
        self.last_action = None
        self._policy_cache = {}
        self._alpha_beta = AlphaBetaPruningBot(self.env, self.env._cfg, "alpha_beta_pruning_player")

    def _player_spec(self) -> PlayerSpec:
        return self.players[int(self.env.current_player)]

    def _bot_action(self, spec: PlayerSpec) -> int:
        if spec.player_type == "bot":
            if spec.bot_type == "random":
                return self.env.random_action()
            if spec.bot_type == "v0":
                return self.env.rule_bot_v0()
            if spec.bot_type == "heuristic_perfect":
                return self.env.rule_bot_heuristic_perfect()
            if spec.bot_type == "alpha_beta_pruning":
                return self._alpha_beta.get_best_action(self.env.board, player_index=self.env.current_player_index)
            raise ValueError(f"Unsupported bot_type: {spec.bot_type}")
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
            obs = self.env.current_state()[1]
            action_mask = np.zeros(self.env.total_num_actions, dtype=np.int8)
            action_mask[self.env.legal_actions] = 1
            eval_obs = {
                "board": copy.deepcopy(self.env.board),
                "current_player_index": int(self.env.current_player_index),
            }
            return self._policy_cache[key].select_action(
                obs, action_mask, int(self.env.current_player), eval_obs=eval_obs
            )
        raise ValueError(f"Unsupported player_type: {spec.player_type}")

    def _apply_action(self, action: int) -> None:
        timestep = self.env.step(action)
        self.last_action = int(action)
        self.done = bool(timestep.done)
        if self.done:
            _, winner = self.env.get_done_winner()
            self.winner = int(winner)

    def advance(self, max_steps: int = 20) -> None:
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
        done, winner = self.env.get_done_winner()
        self.done = bool(done)
        if done:
            self.winner = int(winner)
        action_mask = np.zeros(self.env.total_num_actions, dtype=np.int8)
        action_mask[self.env.legal_actions] = 1
        return {
            "game": "tictactoe",
            "board": self.env.board.tolist(),
            "current_player": int(self.env.current_player),
            "legal_actions": [int(x) for x in self.env.legal_actions],
            "done": self.done,
            "winner": self.winner,
            "last_action": self.last_action,
            "players": {
                1: self.players[1].__dict__,
                2: self.players[2].__dict__,
            },
            "action_mask": action_mask.tolist(),
        }


class TicTacToeGame:
    name = "tictactoe"
    label = "Tic Tac Toe"
    supported_policy_configs = SUPPORTED_POLICY_CONFIGS

    def new_session(self, players: Dict[int, PlayerSpec], auto_play: bool = True) -> TicTacToeSession:
        session = TicTacToeSession(players, auto_play=auto_play)
        if auto_play:
            session.advance()
        return session
