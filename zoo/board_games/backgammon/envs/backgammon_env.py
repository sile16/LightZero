import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import random

# Import custom cython engine
current_dir = os.path.dirname(os.path.abspath(__file__))
# zoo/board_games/backgammon/envs -> zoo/board_games/backgammon/cython_backgammon
cython_dir = os.path.join(os.path.dirname(current_dir), 'cython_backgammon')
sys.path.append(cython_dir)
import state

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict
from zoo.board_games.backgammon.envs.backgammon_bot import BackgammonRandomBot

# Observation type constants
OBS_MINIMAL = 'minimal'
OBS_STANDARD = 'standard'
OBS_WITH_FEATURES = 'features'

# Observation shapes for each type
# Width 25 = 24 board points + bar at index 24 (off remains scalar)
# minimal: 12 board (6 my + 6 opp) + 24 dice + 2 off scalars + 2 legal_dice = 40
# standard: minimal + 1 contact + 2 total_pips + 2 checkers_in_home + 2 deltas = 47
# features: standard + 2 pips_outside + 2 stragglers + 1 delta_stragglers = 52
OBS_SHAPES = {
    OBS_MINIMAL: (40, 1, 25),
    OBS_STANDARD: (47, 1, 25),
    OBS_WITH_FEATURES: (52, 1, 25),
}

# Chance outcomes: unordered dice pairs mapped to [0, 20].
_CHANCE_OUTCOMES = [(i, j) for i in range(1, 7) for j in range(i, 7)]
_CHANCE_MAP = {pair: idx for idx, pair in enumerate(_CHANCE_OUTCOMES)}
_CHANCE_NO_ROLL = len(_CHANCE_OUTCOMES)


@ENV_REGISTRY.register('backgammon')
class BackgammonEnv(BaseEnv):
    config = dict(
        env_id="Backgammon",
        # self_play_mode: One agent controls both sides (for AlphaZero training)
        # play_with_bot_mode: Agent plays player 0, Bot plays player 1
        battle_mode='self_play_mode',
        # obs_type: 'minimal', 'standard', or 'features'
        obs_type=OBS_STANDARD,
        # reward_scale: normalize terminal rewards by this value
        reward_scale=3.0,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None) -> None:
        default_config = self.default_config()
        if cfg is not None:
            default_config.update(cfg)
        self.cfg = default_config
        self._init_flag = False
        self._current_player = 0  # White
        self.battle_mode = self.cfg.battle_mode
        self.obs_type = getattr(self.cfg, 'obs_type', OBS_STANDARD)
        self.reward_scale = float(getattr(self.cfg, 'reward_scale', 3.0))
        self._rng = random.Random()
        # 21 dice outcomes + 1 "no roll" outcome for deterministic transitions.
        self.chance_space_size = 22
        self._turn_dice = None

        # Initialize Bot if needed
        if self.battle_mode == 'play_with_bot_mode':
            self.bot = BackgammonRandomBot(self)

        # Action Space: 25 sources × 2 dice slots = 50 actions
        # Sources: 0-23 (board points) + 24 (bar)
        # Dice slots: 0 (first die) or 1 (second die)
        # Encoding: action = source * 2 + die_slot
        #
        # At any moment, player has at most 2 distinct dice values to choose from.
        # For doubles (e.g., [4,4,4,4]), both slots map to same value.
        # No pass action - env auto-handles turn transitions.
        self.action_space_size = 25 * 2  # 50
        self._action_space = spaces.Discrete(self.action_space_size)

        # Observation Space - shape depends on obs_type
        obs_shape = OBS_SHAPES[self.obs_type]
        self._observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

        self.game = state.State()

    def reset(self, start_player_index=0, init_state=None):
        self.game.reset()

        # Start game with player 0, roll dice (doubles allowed), ready for movement
        self.game.force_start(start_player=start_player_index)
        self._current_player = self.game.get_player_turn()
        self._update_turn_dice_from_remaining()

        # In bot mode, if no legal actions (auto-passed), advance until agent can play
        if self.battle_mode == 'play_with_bot_mode':
            while self._current_player != 0 and not self.game.game_ended():
                obs = self.observe()
                bot_action = self.bot.get_action(obs)
                self._step_core(bot_action)

        return self.observe()

    def set_dice(self, dice_values):
        """
        Test/training helper to set specific dice values.

        Args:
            dice_values: List of dice values, e.g., [3, 5] or [4, 4, 4, 4] for doubles

        This sets the dice, clears the nature turn flag, and regenerates movement moves.
        """
        self.game.set_dice(dice_values)
        self.game.set_nature_turn(False)
        self.game.generate_movement_moves()
        self._current_player = self.game.get_player_turn()
        self._set_turn_dice_from_values(dice_values)

    def set_current_player(self, player):
        """
        Test/training helper to change the current player.

        Args:
            player: 0 (White) or 1 (Black)

        This changes whose turn it is. Use with set_dice() to fully control game state.
        """
        assert player in [0, 1], f"Player must be 0 or 1, got {player}"
        self.game.set_turn(player)
        self._current_player = player
        # Regenerate moves for the new player
        if not self.game.is_nature_turn():
            self.game.generate_movement_moves()

    def step(self, action):
        if self.battle_mode == 'self_play_mode':
            timestep = self._step_core(action)
            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            # 1. Player (Agent) Step
            timestep = self._step_core(action)
            if timestep.done:
                return timestep

            # 2. Bot Step(s)
            # The bot must play until it is the Agent's turn again or game ends
            # Note: Backgammon turns consist of multiple atomic moves.
            # While it is NOT Agent's turn (i.e. it is Bot's turn), Bot plays.

            while self._current_player != 0 and not self.game.game_ended():
                bot_action = self.bot.get_action(timestep.obs)
                timestep = self._step_core(bot_action)
                if timestep.done:
                    # In bot mode, if bot wins, reward for agent is -1.
                    # _step_core handles reward assignment relative to winner.
                    # If winner is 1 (Bot), reward is -1. Correct.
                    return timestep

            return timestep

    def _step_core(self, action):
        # Map Action Index -> Move Object
        # Action = Source * 2 + die_slot
        # Action range: 0-49
        # Clear chance; only set if a new roll happens in this transition.
        self._turn_dice = None

        # Track who made this move for reward calculation
        acting_player = self._current_player

        legal_moves = self.game.get_moves()
        if len(legal_moves) == 0:
            self.game.advance_turn_if_no_moves()
            if not self.game.game_ended():
                if self.game.is_nature_turn():
                    self.game.auto_roll()
                    self._update_turn_dice_from_remaining()
            done = self.game.game_ended()
            winner = self.game.get_winner()
            reward = 0
            if done:
                win_value = self._get_win_value(winner)
                if self.battle_mode == 'play_with_bot_mode':
                    reward = win_value if winner == 0 else -win_value
                else:
                    reward = win_value if winner == acting_player else -win_value
            if done and self.reward_scale > 0:
                reward = reward / self.reward_scale
            self._current_player = self.game.get_player_turn()
            obs = self.observe()
            return BaseEnvTimestep(obs, reward, done, {'action_mask': obs['action_mask']})

        if isinstance(action, np.ndarray):
            action = int(action)
        legal_actions = [i for i, x in enumerate(self.observe()['action_mask']) if x == 1]
        if len(legal_actions) == 0:
            if self.game.is_nature_turn():
                self.game.auto_roll()
                self._update_turn_dice_from_remaining()
            else:
                self.game.advance_turn_if_no_moves()
                if not self.game.game_ended():
                    if self.game.is_nature_turn():
                        self.game.auto_roll()
                        self._update_turn_dice_from_remaining()
            done = self.game.game_ended()
            winner = self.game.get_winner()
            reward = 0
            if done:
                win_value = self._get_win_value(winner)
                if self.battle_mode == 'play_with_bot_mode':
                    reward = win_value if winner == 0 else -win_value
                else:
                    reward = win_value if winner == acting_player else -win_value
            if done and self.reward_scale > 0:
                reward = reward / self.reward_scale
            self._current_player = self.game.get_player_turn()
            obs = self.observe()
            return BaseEnvTimestep(obs, reward, done, {'action_mask': obs['action_mask']})
        if action not in legal_actions:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {legal_actions}. "
                f"Now we randomly choice a action from legal_actions."
            )
            action = np.random.choice(legal_actions)

        # Decode action
        src = action // 2
        die_slot = action % 2

        # Get die value from slot
        slot0_die, slot1_die = self._get_dice_slots()
        die_val = slot0_die if die_slot == 0 else slot1_die

        # Find matching move
        matching_move = None
        for m in legal_moves:
            if not m.is_movement_move:
                continue
            m_src = m.src
            if m_src == -1:
                m_src = 24  # Map Bar -1 to 24 for indexing

            if m_src == src and m.n == die_val:
                matching_move = m
                break

        if matching_move:
            self.game.do_move(matching_move)

            # After move, auto-roll if turn switched to nature turn
            if not self.game.game_ended():
                if self.game.is_nature_turn():
                    self.game.auto_roll()
                    self._update_turn_dice_from_remaining()

        # Check for game end
        done = self.game.game_ended()
        winner = self.game.get_winner()

        # Reward calculation
        # In self_play_mode: reward from perspective of the acting player
        # In play_with_bot_mode: reward always from agent's (player 0) perspective
        reward = 0
        if done:
            win_value = self._get_win_value(winner)
            if self.battle_mode == 'play_with_bot_mode':
                # Agent is always player 0
                reward = win_value if winner == 0 else -win_value
            else:
                # Self-play: from acting player's perspective
                reward = win_value if winner == acting_player else -win_value
            if self.reward_scale > 0:
                reward = reward / self.reward_scale

        self._current_player = self.game.get_player_turn()

        obs = self.observe()
        return BaseEnvTimestep(obs, reward, done, {'action_mask': obs['action_mask']})

    def _get_win_value(self, winner):
        """
        Return win value based on end state:
        1 = normal win, 2 = gammon, 3 = backgammon.
        """
        loser = 1 - winner
        beared_off = self.game.get_beared_off()
        if beared_off[loser] > 0:
            return 1

        bar = self.game.get_bar()
        if bar[loser] > 0:
            return 3

        board = self.game.get_board()
        if winner == 0:
            home_slice = board[loser, 0:6]
        else:
            home_slice = board[loser, 18:24]
        if np.any(home_slice > 0):
            return 3

        return 2

    def _get_dice_slots(self):
        """
        Get unique dice values mapped to slots, sorted high to low.
        Returns (slot0_value, slot1_value) where slot0 >= slot1.
        For doubles, both slots have the same value.
        If only one die remains, slot1 is 0 to avoid duplicate actions.
        """
        remaining = self.game.get_remaining_dice()
        if len(remaining) == 0:
            return (0, 0)
        if len(remaining) == 1:
            return (remaining[0], 0)
        unique = sorted(set(remaining), reverse=True)  # Higher die first
        if len(unique) == 1:
            return (unique[0], unique[0])
        return (unique[0], unique[1])

    def _get_relative_arrays(self):
        """
        Get board state as relative arrays from current player's perspective.

        Returns:
            my_arr: np.array of shape (24,) - my checker counts on points 0-23
            opp_arr: np.array of shape (24,) - opponent checker counts on points 0-23
            my_bar: int - my checkers on bar
            opp_bar: int - opponent checkers on bar
            my_off: int - my checkers borne off
            opp_off: int - opponent checkers borne off

        Point indexing is relative to current player:
            - Index 0 = my ace point (closest to bearing off)
            - Index 23 = my 24-point (furthest from bearing off)
        """
        me = self._current_player
        raw_board = self.game.get_board()  # (2, 24)
        raw_bar = self.game.get_bar()  # (2,)
        raw_off = self.game.get_beared_off()  # (2,)

        if me == 0:  # White (moves 23 -> 0)
            my_arr = raw_board[0, 0:24].copy()
            opp_arr = raw_board[1, 0:24].copy()
            my_bar = raw_bar[0]
            opp_bar = raw_bar[1]
            my_off = raw_off[0]
            opp_off = raw_off[1]
        else:  # Black (moves 0 -> 23)
            # Flip the board so index 0 is Black's ace point
            my_arr = raw_board[1, 0:24][::-1].copy()
            opp_arr = raw_board[0, 0:24][::-1].copy()
            my_bar = raw_bar[1]
            opp_bar = raw_bar[0]
            my_off = raw_off[1]
            opp_off = raw_off[0]

        return my_arr, opp_arr, my_bar, opp_bar, my_off, opp_off

    def observe(self):
        """Build observation dict with action mask."""
        # Action Mask - only for movement moves
        # Encoding: action = source * 2 + die_slot
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        legal_moves = self.game.get_moves()

        slot0_die, slot1_die = self._get_dice_slots()

        for m in legal_moves:
            if not m.is_movement_move:
                continue
            src = m.src if m.src != -1 else 24
            die = m.n

            # Map die value to slot(s)
            if die == slot0_die:
                idx = src * 2 + 0
                if 0 <= idx < self.action_space_size:
                    mask[idx] = 1
            if die == slot1_die and slot1_die != slot0_die:
                idx = src * 2 + 1
                if 0 <= idx < self.action_space_size:
                    mask[idx] = 1
            elif die == slot1_die and slot1_die == slot0_die:
                # Doubles - both slots valid for same die value
                idx = src * 2 + 1
                if 0 <= idx < self.action_space_size:
                    mask[idx] = 1

        # Compute legal dice slot flags from action mask
        # slot0_playable: any legal action uses slot 0 (action % 2 == 0)
        # slot1_playable: any legal action uses slot 1 (action % 2 == 1)
        slot0_playable = any(mask[i] == 1 for i in range(0, self.action_space_size, 2))
        slot1_playable = any(mask[i] == 1 for i in range(1, self.action_space_size, 2))

        # Select observation function based on obs_type
        if self.obs_type == OBS_MINIMAL:
            obs_vector = self._get_obs_minimal(slot0_playable, slot1_playable)
        elif self.obs_type == OBS_STANDARD:
            obs_vector = self._get_obs_standard(slot0_playable, slot1_playable)
        else:
            obs_vector = self._get_obs_with_features(slot0_playable, slot1_playable)

        return {
            'observation': obs_vector,
            'action_mask': mask,
            'to_play': (self._current_player + 1) if self.battle_mode == 'self_play_mode' else -1,
            'chance': self._get_chance_value(self._turn_dice),
        }

    def _get_chance_value(self, dice_values):
        if not dice_values:
            return _CHANCE_NO_ROLL
        die1, die2 = dice_values
        ordered = (die1, die2) if die1 <= die2 else (die2, die1)
        return _CHANCE_MAP[ordered]

    def _set_turn_dice_from_values(self, dice_values):
        if not dice_values:
            self._turn_dice = None
            return
        if len(dice_values) >= 2:
            if len(dice_values) == 4:
                self._turn_dice = (dice_values[0], dice_values[0])
            else:
                self._turn_dice = (dice_values[0], dice_values[1])

    def _update_turn_dice_from_remaining(self):
        remaining = self.game.get_remaining_dice()
        if len(remaining) == 0:
            self._turn_dice = None
            return
        if len(remaining) == 4:
            self._turn_dice = (remaining[0], remaining[0])
        elif len(remaining) == 2:
            self._turn_dice = (remaining[0], remaining[1])

    def _get_obs_minimal(self, slot0_playable=False, slot1_playable=False):
        """
        Minimal observation representation with separate planes for my/opponent checkers.

        Shape: (40, 1, 25)
        Width 25 = 24 board points + bar at index 24 (off remains scalar)

        - Channels 0-5: My checker counts (>=1, >=2, ..., >=6 thresholds)
        - Channels 6-11: Opponent checker counts (>=1, >=2, ..., >=6 thresholds)
        - Channels 12-35: Dice (4 slots × 6 one-hot), deterministically ordered high→low
        - Channels 36-37: Scalar features broadcasted:
            - 36: My off / 15
            - 37: Opponent off / 15
        - Channels 38-39: Legal dice slot flags (aligned with action encoding):
            - 38: Slot 0 playable (1 if any legal action uses slot 0)
            - 39: Slot 1 playable (1 if any legal action uses slot 1)
        """
        obs = np.zeros((40, 1, 25), dtype=np.float32)

        my_arr, opp_arr, my_bar, opp_bar, my_off, opp_off = self._get_relative_arrays()

        # A. Board Topology (Channels 0-11) - Separate planes for my/opponent
        # Each player gets 6 channels for count thresholds (>=1, >=2, ..., >=6)
        for i in range(1, 7):
            # My checkers on board (points 0-23)
            obs[i - 1, 0, :24][my_arr >= i] = 1.0
            # My checkers on bar (index 24)
            if my_bar >= i:
                obs[i - 1, 0, 24] = 1.0
            # Opponent checkers on board (points 0-23)
            obs[i + 5, 0, :24][opp_arr >= i] = 1.0
            # Opponent checkers on bar (index 24)
            if opp_bar >= i:
                obs[i + 5, 0, 24] = 1.0

        # B. Dice (Channels 12-35)
        slot0_die, slot1_die = self._get_dice_slots()
        remaining = self.game.get_remaining_dice()

        dice_ordered = []
        if slot0_die > 0:
            dice_ordered.append(slot0_die)
        if slot1_die > 0 and slot1_die != slot0_die:
            dice_ordered.append(slot1_die)
        elif slot1_die > 0 and slot1_die == slot0_die:
            count = remaining.count(slot0_die)
            dice_ordered = [slot0_die] * count

        dice_ordered = dice_ordered + [0] * (4 - len(dice_ordered))

        for slot_idx, die_val in enumerate(dice_ordered[:4]):
            if die_val > 0:
                channel = 12 + slot_idx * 6 + (die_val - 1)
                obs[channel, :, :] = 1.0

        # C. Scalar Features (Channels 36-37) - Bar is now spatial, only off as scalar
        obs[36, :, :] = my_off / 15.0
        obs[37, :, :] = opp_off / 15.0

        # D. Legal Dice Slot Flags (Channels 38-39)
        # These tell the network which dice slots have playable moves
        obs[38, :, :] = 1.0 if slot0_playable else 0.0
        obs[39, :, :] = 1.0 if slot1_playable else 0.0

        return obs

    def _compute_contact(self, my_arr, opp_arr, my_bar, opp_bar):
        """
        Compute whether there is contact between opposing checkers.

        Contact exists when any of my checkers could potentially interact with
        opponent's checkers (i.e., they haven't fully passed each other).

        In relative indexing:
        - I move toward 0 (bearing off), opponent moves toward 23
        - Contact = any of my checkers at point i AND opponent has checkers at point j < i
        - Also contact if either player has checkers on bar

        Returns:
            bool: True if contact exists, False if pure race
        """
        # Bar always means contact (need to re-enter through opponent's home)
        if my_bar > 0 or opp_bar > 0:
            return True

        # Find my rearmost checker (highest index with my checkers)
        my_rear = -1
        for i in range(23, -1, -1):
            if my_arr[i] > 0:
                my_rear = i
                break

        # Find opponent's rearmost checker from their perspective (lowest index)
        opp_front = 24  # From my perspective, opponent's "front" is at low indices
        for i in range(24):
            if opp_arr[i] > 0:
                opp_front = i
                break

        # Contact if my rear is ahead of (greater than) opponent's front
        # i.e., we haven't passed each other yet
        return my_rear > opp_front if my_rear >= 0 and opp_front < 24 else False

    def _get_obs_standard(self, slot0_playable=False, slot1_playable=False):
        """
        Standard observation with total pip counts and home checker counts.

        Shape: (47, 1, 25)
        Width 25 = 24 board points + bar at index 24 (off remains scalar)

        - Channels 0-39: Same as minimal observation (including legal dice flags)
        - Channel 40: Contact indicator (1 if any opposing checkers in front of each other)
        - Channels 41-46: Race position features:
            - 41: My total pips / 200
            - 42: Opponent total pips / 200
            - 43: My checkers in home / 15
            - 44: Opponent checkers in home / 15
            - 45: Delta pips (clipped to [-1, 1])
            - 46: Delta checkers in home / 15 (clipped to [-1, 1])
        """
        obs = np.zeros((47, 1, 25), dtype=np.float32)

        my_arr, opp_arr, my_bar, opp_bar, my_off, opp_off = self._get_relative_arrays()

        # A. Board Topology (Channels 0-11) - Separate planes for my/opponent
        for i in range(1, 7):
            # My checkers on board (points 0-23)
            obs[i - 1, 0, :24][my_arr >= i] = 1.0
            # My checkers on bar (index 24)
            if my_bar >= i:
                obs[i - 1, 0, 24] = 1.0
            # Opponent checkers on board (points 0-23)
            obs[i + 5, 0, :24][opp_arr >= i] = 1.0
            # Opponent checkers on bar (index 24)
            if opp_bar >= i:
                obs[i + 5, 0, 24] = 1.0

        # B. Dice (Channels 12-35)
        slot0_die, slot1_die = self._get_dice_slots()
        remaining = self.game.get_remaining_dice()

        dice_ordered = []
        if slot0_die > 0:
            dice_ordered.append(slot0_die)
        if slot1_die > 0 and slot1_die != slot0_die:
            dice_ordered.append(slot1_die)
        elif slot1_die > 0 and slot1_die == slot0_die:
            count = remaining.count(slot0_die)
            dice_ordered = [slot0_die] * count

        dice_ordered = dice_ordered + [0] * (4 - len(dice_ordered))

        for slot_idx, die_val in enumerate(dice_ordered[:4]):
            if die_val > 0:
                channel = 12 + slot_idx * 6 + (die_val - 1)
                obs[channel, :, :] = 1.0

        # C. Scalar Features (Channels 36-37) - Bar is now spatial, only off as scalar
        obs[36, :, :] = my_off / 15.0
        obs[37, :, :] = opp_off / 15.0

        # D. Legal Dice Slot Flags (Channels 38-39)
        obs[38, :, :] = 1.0 if slot0_playable else 0.0
        obs[39, :, :] = 1.0 if slot1_playable else 0.0

        # E. Contact Indicator (Channel 40)
        contact = self._compute_contact(my_arr, opp_arr, my_bar, opp_bar)
        obs[40, :, :] = 1.0 if contact else 0.0

        # F. Race Position Features (Channels 41-46)

        # Pip weights for correct direction calculation
        pip_weights_me = np.arange(1, 25, dtype=np.int32)  # [1, 2, ..., 24]
        pip_weights_opp = np.arange(24, 0, -1, dtype=np.int32)  # [24, 23, ..., 1]

        # 41: My total pips (including bar)
        my_total_pips = np.dot(my_arr, pip_weights_me) + my_bar * 25
        obs[41, :, :] = my_total_pips / 200.0

        # 42: Opponent total pips (including bar)
        opp_total_pips = np.dot(opp_arr, pip_weights_opp) + opp_bar * 25
        obs[42, :, :] = opp_total_pips / 200.0

        # 43: My checkers in home (indices 0-5)
        my_home_checkers = np.sum(my_arr[0:6])
        obs[43, :, :] = my_home_checkers / 15.0

        # 44: Opponent checkers in home (indices 18-23 from my perspective = their 0-5)
        opp_home_checkers = np.sum(opp_arr[18:24])
        obs[44, :, :] = opp_home_checkers / 15.0

        # 45: Delta pips - clipped for training stability
        delta_pips = my_total_pips - opp_total_pips
        obs[45, :, :] = np.clip(delta_pips / 200.0, -1.0, 1.0)

        # 46: Delta checkers in home - clipped for training stability
        delta_home = my_home_checkers - opp_home_checkers
        obs[46, :, :] = np.clip(delta_home / 15.0, -1.0, 1.0)

        return obs

    def _get_obs_with_features(self, slot0_playable=False, slot1_playable=False):
        """
        Full-featured observation representation with separate planes.

        Shape: (52, 1, 25)
        Width 25 = 24 board points + bar at index 24 (off remains scalar)

        - Channels 0-39: Same as minimal observation (including legal dice flags)
        - Channel 40: Contact indicator (1 if any opposing checkers in front of each other)
        - Channels 41-46: Standard race features:
            - 41: My total pips / 200
            - 42: Opponent total pips / 200
            - 43: My checkers in home / 15
            - 44: Opponent checkers in home / 15
            - 45: Delta pips (clipped to [-1, 1])
            - 46: Delta checkers in home / 15 (clipped to [-1, 1])
        - Channels 47-51: Feature-rich race features:
            - 47: My pips outside home / 100
            - 48: Opponent pips outside home / 100
            - 49: My stragglers / 15 (checkers outside home + bar; 0 = can bear off)
            - 50: Opponent stragglers / 15
            - 51: Delta stragglers (clipped to [-1, 1])
        """
        obs = np.zeros((52, 1, 25), dtype=np.float32)

        my_arr, opp_arr, my_bar, opp_bar, my_off, opp_off = self._get_relative_arrays()

        # A. Board Topology (Channels 0-11) - Separate planes for my/opponent
        for i in range(1, 7):
            # My checkers on board (points 0-23)
            obs[i - 1, 0, :24][my_arr >= i] = 1.0
            # My checkers on bar (index 24)
            if my_bar >= i:
                obs[i - 1, 0, 24] = 1.0
            # Opponent checkers on board (points 0-23)
            obs[i + 5, 0, :24][opp_arr >= i] = 1.0
            # Opponent checkers on bar (index 24)
            if opp_bar >= i:
                obs[i + 5, 0, 24] = 1.0

        # B. Dice (Channels 12-35)
        slot0_die, slot1_die = self._get_dice_slots()
        remaining = self.game.get_remaining_dice()

        dice_ordered = []
        if slot0_die > 0:
            dice_ordered.append(slot0_die)
        if slot1_die > 0 and slot1_die != slot0_die:
            dice_ordered.append(slot1_die)
        elif slot1_die > 0 and slot1_die == slot0_die:
            count = remaining.count(slot0_die)
            dice_ordered = [slot0_die] * count

        dice_ordered = dice_ordered + [0] * (4 - len(dice_ordered))

        for slot_idx, die_val in enumerate(dice_ordered[:4]):
            if die_val > 0:
                channel = 12 + slot_idx * 6 + (die_val - 1)
                obs[channel, :, :] = 1.0

        # C. Scalar Features (Channels 36-37) - Bar is now spatial, only off as scalar
        obs[36, :, :] = my_off / 15.0
        obs[37, :, :] = opp_off / 15.0

        # D. Legal Dice Slot Flags (Channels 38-39)
        obs[38, :, :] = 1.0 if slot0_playable else 0.0
        obs[39, :, :] = 1.0 if slot1_playable else 0.0

        # E. Contact Indicator (Channel 40)
        contact = self._compute_contact(my_arr, opp_arr, my_bar, opp_bar)
        obs[40, :, :] = 1.0 if contact else 0.0

        # F. Standard Race Features (Channels 41-46)

        # Pip weights for correct direction calculation
        pip_weights_me = np.arange(1, 25, dtype=np.int32)  # [1, 2, ..., 24]
        pip_weights_opp = np.arange(24, 0, -1, dtype=np.int32)  # [24, 23, ..., 1]

        # 41: My total pips (including bar)
        my_total_pips = np.dot(my_arr, pip_weights_me) + my_bar * 25
        obs[41, :, :] = my_total_pips / 200.0

        # 42: Opponent total pips (including bar)
        opp_total_pips = np.dot(opp_arr, pip_weights_opp) + opp_bar * 25
        obs[42, :, :] = opp_total_pips / 200.0

        # 43: My checkers in home (indices 0-5)
        my_home_checkers = np.sum(my_arr[0:6])
        obs[43, :, :] = my_home_checkers / 15.0

        # 44: Opponent checkers in home (indices 18-23 from my perspective)
        opp_home_checkers = np.sum(opp_arr[18:24])
        obs[44, :, :] = opp_home_checkers / 15.0

        # 45: Delta pips - clipped for training stability
        delta_pips = my_total_pips - opp_total_pips
        obs[45, :, :] = np.clip(delta_pips / 200.0, -1.0, 1.0)

        # 46: Delta checkers in home - clipped for training stability
        delta_home = my_home_checkers - opp_home_checkers
        obs[46, :, :] = np.clip(delta_home / 15.0, -1.0, 1.0)

        # G. Feature-rich Race Features (Channels 47-51)

        # 47: My pips outside home (indices 6-23 + bar)
        my_pips_outside = np.dot(my_arr[6:24], pip_weights_me[6:24]) + my_bar * 25
        obs[47, :, :] = my_pips_outside / 100.0

        # 48: Opponent pips outside home (correct direction)
        opp_pips_outside = np.dot(opp_arr[0:18], pip_weights_opp[0:18]) + opp_bar * 25
        obs[48, :, :] = opp_pips_outside / 100.0

        # 49: My stragglers (0 = can bear off)
        my_stragglers = np.sum(my_arr[6:24]) + my_bar
        obs[49, :, :] = my_stragglers / 15.0

        # 50: Opponent stragglers
        opp_stragglers = np.sum(opp_arr[0:18]) + opp_bar
        obs[50, :, :] = opp_stragglers / 15.0

        # 51: Delta stragglers - clipped for training stability
        delta_stragglers = my_stragglers - opp_stragglers
        obs[51, :, :] = np.clip(delta_stragglers / 15.0, -1.0, 1.0)

        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def legal_actions(self):
        # Return list of indices
        mask = self.observe()['action_mask']
        return [i for i, x in enumerate(mask) if x == 1]

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """Set the random seed for the environment."""
        self._rng.seed(seed)
        np.random.seed(seed)
        self.game.seed(seed)

    def close(self) -> None:
        """Clean up resources. No-op for this environment."""
        pass

    def __repr__(self) -> str:
        return "LightZero Backgammon Env"
