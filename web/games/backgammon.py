from typing import Dict, List

import numpy as np
from zoo.board_games.backgammon.envs.backgammon_bot import BackgammonRandomBot
from zoo.board_games.backgammon.envs.backgammon_env import BackgammonEnv

from web.games.common import PlayerSpec, PolicyAgent

SUPPORTED_POLICY_CONFIGS = {
    "stochastic_muzero": "zoo.board_games.backgammon.config.backgammon_stochastic_muzero_bot_mode_config",
    "alphazero": "zoo.board_games.backgammon.config.backgammon_alphazero_bot_mode_config",
}


def _format_point(idx: int) -> str:
    if idx == -1:
        return "off"
    if idx == 24:
        return "bar"
    return str(idx + 1)


class BackgammonSession:
    def __init__(self, players: Dict[int, PlayerSpec], auto_play: bool = True):
        cfg = BackgammonEnv.default_config()
        cfg.battle_mode = "self_play_mode"
        self.env = BackgammonEnv(cfg)
        self.env.reset(start_player_index=0)
        self.players = players
        self.auto_play = auto_play
        self.done = False
        self.winner = None
        self.last_action = None
        self._policy_cache = {}
        self._random_bot = BackgammonRandomBot(self.env)

    def _player_spec(self) -> PlayerSpec:
        return self.players[int(self.env._current_player + 1)]

    def _bot_action(self, spec: PlayerSpec) -> int:
        if spec.player_type == "bot":
            obs = self.env.observe()
            return self._random_bot.get_action(obs)
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
            obs = self.env.observe()
            eval_obs = None
            if spec.algo == "alphazero":
                eval_obs = {
                    "board": {
                        "board": self.env.game.get_board().tolist(),
                        "bar": self.env.game.get_bar().tolist(),
                        "off": self.env.game.get_beared_off().tolist(),
                        "turn": int(self.env.game.get_player_turn()),
                        "dice": list(self.env.game.get_remaining_dice()),
                        "nature_turn": bool(self.env.game.is_nature_turn()),
                    },
                    "current_player_index": int(self.env.current_player),
                }
            return self._policy_cache[key].select_action(
                obs["observation"],
                np.array(obs["action_mask"], dtype=np.int8),
                int(obs["to_play"]),
                eval_obs=eval_obs,
            )
        raise ValueError(f"Unsupported player_type: {spec.player_type}")

    def _apply_action(self, action: int) -> None:
        timestep = self.env.step(action)
        self.last_action = int(action)
        self.done = bool(timestep.done)
        if self.done:
            self.winner = int(self.env.game.get_winner())

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

    def _legal_action_details(self, legal_actions: List[int]) -> List[Dict]:
        slot0_die, slot1_die = self.env._get_dice_slots()
        slot_map = {0: int(slot0_die), 1: int(slot1_die)}
        moves = self.env.game.get_moves()
        details = []
        for action in legal_actions:
            src = action // 2
            die_slot = action % 2
            die_val = slot_map.get(die_slot, 0)
            dst = None
            for move in moves:
                if not move.is_movement_move:
                    continue
                move_src = move.src if move.src != -1 else 24
                if move_src == src and move.n == die_val:
                    dst = move.dst
                    break
            label = f"{_format_point(src)} -> {_format_point(dst) if dst is not None else '?'} (die {die_val})"
            details.append(
                {
                    "action": int(action),
                    "src": int(src),
                    "dst": int(dst) if dst is not None else None,
                    "die": int(die_val),
                    "label": label,
                }
            )
        return details

    def state(self) -> Dict:
        obs = self.env.observe()
        legal_actions = [int(i) for i, val in enumerate(obs["action_mask"]) if val == 1]
        raw_board = self.env.game.get_board()
        raw_bar = self.env.game.get_bar()
        raw_off = self.env.game.get_beared_off()
        return {
            "game": "backgammon",
            "current_player": int(self.env._current_player + 1),
            "legal_actions": legal_actions,
            "legal_action_details": self._legal_action_details(legal_actions),
            "action_mask": obs["action_mask"].tolist(),
            "done": self.done,
            "winner": self.winner,
            "last_action": self.last_action,
            "dice": {
                "remaining": list(self.env.game.get_remaining_dice()),
                "slots": list(self.env._get_dice_slots()),
            },
            "board": {
                "p1_points": raw_board[0].tolist(),
                "p2_points": raw_board[1].tolist(),
                "p1_bar": int(raw_bar[0]),
                "p2_bar": int(raw_bar[1]),
                "p1_off": int(raw_off[0]),
                "p2_off": int(raw_off[1]),
            },
            "players": {
                1: self.players[1].__dict__,
                2: self.players[2].__dict__,
            },
        }


class BackgammonGame:
    name = "backgammon"
    label = "Backgammon"
    supported_policy_configs = SUPPORTED_POLICY_CONFIGS

    def new_session(self, players: Dict[int, PlayerSpec], auto_play: bool = True) -> BackgammonSession:
        session = BackgammonSession(players, auto_play=auto_play)
        if auto_play:
            session.advance()
        return session
