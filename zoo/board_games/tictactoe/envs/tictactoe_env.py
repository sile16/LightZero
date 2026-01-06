import copy
import os
import sys
from datetime import datetime
from functools import lru_cache
from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from ditk import logging
from easydict import EasyDict
from zoo.board_games.tictactoe.envs.get_done_winner_cython import get_done_winner_cython
from zoo.board_games.tictactoe.envs.legal_actions_cython import legal_actions_cython

from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot


@lru_cache(maxsize=512)
def _legal_actions_func_lru(board_tuple):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return legal_actions_cython(board_view)


@lru_cache(maxsize=512)
def _get_done_winner_func_lru(board_tuple):
    # Convert tuple to NumPy array.
    board_array = np.array(board_tuple, dtype=np.int32)
    # Convert NumPy array to memory view.
    board_view = board_array.view(dtype=np.int32).reshape(board_array.shape)
    return get_done_winner_cython(board_view)


@ENV_REGISTRY.register('tictactoe')
class TicTacToeEnv(BaseEnv):

    config = dict(
        # (str): The name of the environment.
        env_id="TicTacToe",
        # (bool) If True, means that the game is not a zero-sum game.
        non_zero_sum=False,
        # (str): The mode of the battle. Choices are 'self_play_mode' or 'alpha_beta_pruning'.
        battle_mode='self_play_mode',
        # (str): The mode of Monte Carlo Tree Search. This is only used in AlphaZero.
        battle_mode_in_simulation_env='self_play_mode',
        # (str): The type of action the bot should take. Choices are 'v0' or 'alpha_beta_pruning'.
        bot_action_type='v0',
        # (list): Optional eval bot definitions for eval_mode.
        eval_bots=None,
        # (list): Optional eval matchups for eval_mode.
        eval_matchups=None,
        # (int): The index of the player who starts. 0 = model first, 1 = bot first (in play_with_bot_mode).
        start_player_index=0,
        # (str): The folder path where replay video saved, if None, will not save replay video.
        replay_path=None,
        # (bool): If True, the agent will play against a human.
        agent_vs_human=False,
        # (int): The probability of the random agent.
        prob_random_agent=0,
        # (int): The probability of the expert agent.
        prob_expert_agent=0,
        # (bool): If True, print boards and decisions for heuristic_perfect bot.
        heuristic_print=False,
        # (bool): If True, the channel will be the last dimension.
        channel_last=False,
        # (bool): If True, the pixel values will be scaled.
        scale=True,
        # (int): The value to stop the game.
        stop_value=1,
        # (bool): If True, the Monte Carlo Tree Search from AlphaZero is used.
        alphazero_mcts_ctree=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None):
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        self.channel_last = self._cfg.channel_last
        self.scale = self._cfg.scale
        self.battle_mode = self._cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.battle_mode_in_simulation_env = 'self_play_mode'
        self.board_size = 3
        self.players = [1, 2]
        self.total_num_actions = 9
        self.prob_random_agent = self._cfg.prob_random_agent
        self.prob_expert_agent = self._cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'
        self._env = self
        self.agent_vs_human = self._cfg.agent_vs_human
        self.bot_action_type = self._cfg.bot_action_type
        self.alphazero_mcts_ctree = self._cfg.alphazero_mcts_ctree
        self._replay_path = self._cfg.replay_path if hasattr(self._cfg, "replay_path") and self._cfg.replay_path is not None else None
        self._save_replay_count = 0
        # Default start_player_index from config (0 = model first, 1 = bot first)
        self._default_start_player_index = self._cfg.get('start_player_index', 0)
        # Eval bot scheduling (eval_mode only)
        self._eval_bots = self._cfg.get('eval_bots', None) or []
        self._eval_matchups = self._cfg.get('eval_matchups', None) or []
        self._eval_matchup_index = 0
        self._eval_bot_action_type = None
        self._eval_agent_role = 'first'
        self._eval_matchup_name = None
        needs_alpha_beta = 'alpha_beta_pruning' in self.bot_action_type
        if self._eval_matchups:
            needs_alpha_beta = needs_alpha_beta or any(
                m.get('action_type') == 'alpha_beta_pruning' for m in self._eval_matchups
            )
        if needs_alpha_beta:
            self.alpha_beta_pruning_player = AlphaBetaPruningBot(self, self._cfg, 'alpha_beta_pruning_player')
        # Stochastic MuZero support: TicTacToe is deterministic, so chance is always 0
        self.chance = 0
        self.chance_space_size = 1  # Only one "chance" outcome (no randomness)

    @property
    def legal_actions(self):
        # Convert NumPy arrays to nested tuples to make them hashable.
        return _legal_actions_func_lru(tuple(map(tuple, self.board)))

    # only for evaluation speed
    @property
    def legal_actions_cython(self):
        return legal_actions_cython(list(self.board))

    # only for evaluation speed
    @property
    def legal_actions_cython_lru(self):
        # Convert NumPy arrays to nested tuples to make them hashable.
        return _legal_actions_func_lru(tuple(map(tuple, self.board)))

    def get_done_winner(self):
        """
        Overview:
             Check if the game is over and who the winner is. Return 'done' and 'winner'.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
                - if player 1 win,     'done' = True, 'winner' = 1
                - if player 2 win,     'done' = True, 'winner' = 2
                - if draw,             'done' = True, 'winner' = -1
                - if game is not over, 'done' = False, 'winner' = -1
        """
        # Convert NumPy arrays to nested tuples to make them hashable.
        return _get_done_winner_func_lru(tuple(map(tuple, self.board)))

    def reset(self, start_player_index=None, init_state=None, katago_policy_init=False, katago_game_state=None):
        """
        Overview:
            This method resets the environment and optionally starts with a custom state specified by 'init_state'.
        Arguments:
            - start_player_index (:obj:`int`, optional): Specifies the starting player. The players are [1,2] and
                their corresponding indices are [0,1]. Defaults to 0.
            - init_state (:obj:`Any`, optional): The custom starting state. If provided, the game starts from this state.
                Defaults to None.
            - katago_policy_init (:obj:`bool`, optional): This parameter is used to maintain compatibility with the
                handling of 'katago' related parts in 'alphazero_mcts_ctree' in Go. Defaults to False.
            - katago_game_state (:obj:`Any`, optional): This parameter is similar to 'katago_policy_init' and is used to
                maintain compatibility with 'katago' in 'alphazero_mcts_ctree'. Defaults to None.
        """
        # Use default from config if not specified
        if start_player_index is None:
            start_player_index = self._default_start_player_index
        # Eval mode always starts from player 1; agent role decides who moves first.
        # Skip eval matchup selection if init_state is provided (simulator mode for MCTS/alpha-beta).
        if self.battle_mode == 'eval_mode' and init_state is None:
            start_player_index = 0
            self._select_eval_matchup()

        if self.alphazero_mcts_ctree and init_state is not None:
            # Convert byte string to np.ndarray
            init_state = np.frombuffer(init_state, dtype=np.int32)

        if self.scale:
            self._observation_space = gym.spaces.Box(
                low=0, high=1, shape=(self.board_size, self.board_size, 3), dtype=np.float32
            )
        else:
            self._observation_space = gym.spaces.Box(
                low=0, high=2, shape=(self.board_size, self.board_size, 3), dtype=np.uint8
            )
        self._action_space = gym.spaces.Discrete(self.board_size ** 2)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.board = np.array(copy.deepcopy(init_state), dtype="int32")
            if self.alphazero_mcts_ctree:
                self.board = self.board.reshape((self.board_size, self.board_size))
        else:
            self.board = np.zeros((self.board_size, self.board_size), dtype="int32")

        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1

        if self.battle_mode == 'play_with_bot_mode' or (self.battle_mode == 'eval_mode' and init_state is None):
            # In eval_mode, if agent plays second, bot moves first.
            # Skip if init_state is provided (simulator mode).
            if self.battle_mode == 'eval_mode' and self._eval_agent_role == 'second':
                bot_action = self.bot_action()
                self._player_step(bot_action)
                # Update action mask after bot's move
                action_mask = np.zeros(self.total_num_actions, 'int8')
                action_mask[self.legal_actions] = 1
            # In play_with_bot_mode, if start_player_index=1 (bot goes first), have the bot make the first move
            if self.battle_mode == 'play_with_bot_mode' and self.start_player_index == 1:
                bot_action = self.bot_action()
                self._player_step(bot_action)
                # Update action mask after bot's move
                action_mask = np.zeros(self.total_num_actions, 'int8')
                action_mask[self.legal_actions] = 1

            # In ``play_with_bot_mode`` and ``eval_mode``, we need to set the "to_play" parameter in the "obs" dict to -1,
            # because we don't take into account the alternation between players.
            # The "to_play" parameter is used in the MCTS algorithm.
            obs = {
                'observation': self.current_state()[1],
                'action_mask': action_mask,
                'board': copy.deepcopy(self.board),
                'current_player_index': self.players.index(self._current_player),
                'to_play': -1,
                'chance': self.chance  # Stochastic MuZero support (always 0 for deterministic game)
            }
        else:
            # self_play_mode, or eval_mode with init_state (simulator mode)
            # In the "self_play_mode", we set to_play=self.current_player in the "obs" dict,
            # which is used to differentiate the alternation of 2 players in the game when calculating Q in the MCTS algorithm.
            obs = {
                'observation': self.current_state()[1],
                'action_mask': action_mask,
                'board': copy.deepcopy(self.board),
                'current_player_index': self.players.index(self._current_player),
                'to_play': self.current_player,
                'chance': self.chance  # Stochastic MuZero support (always 0 for deterministic game)
            }
        if self._replay_path is not None:
            self._frames = []

        return obs

    def reset_v2(self, start_player_index=0, init_state=None):
        """
        Overview:
            only used in alpha-beta pruning bot.
        """
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.board = np.array(init_state, dtype="int32")
        else:
            self.board = np.zeros((self.board_size, self.board_size), dtype="int32")

    def step(self, action):
        if self.battle_mode == 'self_play_mode':
            if self.prob_random_agent > 0:
                if np.random.rand() < self.prob_random_agent:
                    action = self.random_action()
            elif self.prob_expert_agent > 0:
                if np.random.rand() < self.prob_expert_agent:
                    action = self.bot_action()

            timestep = self._player_step(action)
            if timestep.done:
                # The eval_episode_return is calculated from Player 1's perspective。
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs[
                                                                               'to_play'] == 1 else timestep.reward
            return timestep
        elif self.battle_mode == 'play_with_bot_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            timestep_player1 = self._player_step(action)
            # self.env.render()
            if timestep_player1.done:
                # NOTE: in play_with_bot_mode, we must set to_play as -1, because we don't consider the alternation between players.
                # And the to_play is used in MCTS.
                timestep_player1.obs['to_play'] = -1
                return timestep_player1

            # player 2's turn
            bot_action = self.bot_action()
            # print('player 2 (computer player): ' + self.action_to_string(bot_action))
            timestep_player2 = self._player_step(bot_action)
            # the eval_episode_return is calculated from Player 1's perspective
            timestep_player2.info['eval_episode_return'] = -timestep_player2.reward
            timestep_player2 = timestep_player2._replace(reward=-timestep_player2.reward)

            timestep = timestep_player2
            # NOTE: in play_with_bot_mode, we must set to_play as -1, because we don't consider the alternation between players.
            # And the to_play is used in MCTS.
            timestep.obs['to_play'] = -1

            return timestep
        elif self.battle_mode == 'eval_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            if self._replay_path is not None:
                self._frames.append(self._env.render(mode='rgb_array'))
            timestep_player1 = self._player_step(action)
            # self.env.render()
            if timestep_player1.done:
                # NOTE: in eval_mode, we must set to_play as -1, because we don't consider the alternation between players.
                # And the to_play is used in MCTS.
                timestep_player1.obs['to_play'] = -1
                timestep_player1.info['eval_episode_return'] = self._eval_agent_return()
                timestep_player1 = timestep_player1._replace(reward=timestep_player1.info['eval_episode_return'])
                self._set_eval_episode_info(timestep_player1)

                if self._replay_path is not None:
                    if not os.path.exists(self._replay_path):
                        os.makedirs(self._replay_path)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    path = os.path.join(
                        self._replay_path,
                        'tictactoe_{}_{}_{}.mp4'.format(os.getpid(), timestamp, self._save_replay_count)
                    )
                    self.display_frames_as_mp4(self._frames, path)
                    print(f'replay {path} saved!')
                    self._save_replay_count += 1

                return timestep_player1

            # player 2's turn
            if self.agent_vs_human:
                bot_action = self.human_to_action()
            else:
                bot_action = self.bot_action()
            # print('player 2 (computer player): ' + self.action_to_string(bot_action))
            if self._replay_path is not None:
                self._frames.append(self._env.render(mode='rgb_array'))
            timestep_player2 = self._player_step(bot_action)
            if self._replay_path is not None:
                self._frames.append(self._env.render(mode='rgb_array'))
            # The eval_episode_return is calculated from agent's perspective.
            if timestep_player2.done:
                timestep_player2.info['eval_episode_return'] = self._eval_agent_return()
                timestep_player2 = timestep_player2._replace(reward=timestep_player2.info['eval_episode_return'])
            self._set_eval_episode_info(timestep_player2)

            timestep = timestep_player2
            # NOTE: in eval_mode, we must set to_play as -1, because we don't consider the alternation between players.
            # And the to_play is used in MCTS.
            timestep.obs['to_play'] = -1

            if timestep_player2.done:
                if self._replay_path is not None:
                    if not os.path.exists(self._replay_path):
                        os.makedirs(self._replay_path)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    path = os.path.join(
                        self._replay_path,
                        'tictactoe_{}_{}_{}.mp4'.format(os.getpid(), timestamp, self._save_replay_count)
                    )
                    self.display_frames_as_mp4(self._frames, path)
                    print(f'replay {path} saved!')
                    self._save_replay_count += 1

            return timestep

    def _player_step(self, action):

        if action in self.legal_actions:
            row, col = self.action_to_coord(action)
            self.board[row, col] = self.current_player
        else:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = np.random.choice(self.legal_actions)
            row, col = self.action_to_coord(action)
            self.board[row, col] = self.current_player

        # Check whether the game is ended or not and give the winner
        done, winner = self.get_done_winner()

        reward = np.array(float(winner == self.current_player)).astype(np.float32)
        info = {'next player to play': self.next_player}
        """
        NOTE: here exchange the player
        """
        self.current_player = self.next_player

        if done:
            info['eval_episode_return'] = reward
            # print('tictactoe one episode done: ', info)
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs = {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'board': copy.deepcopy(self.board),
            'current_player_index': self.players.index(self.current_player),
            'to_play': self.current_player,
            'chance': self.chance  # Stochastic MuZero support (always 0 for deterministic game)
        }
        return BaseEnvTimestep(obs, reward, done, info)

    def current_state(self):
        """
        Overview:
            obtain the state from the view of current player.
            self.board is nd-array, 0 indicates that no stones is placed here,
            1 indicates that player 1's stone is placed here, 2 indicates player 2's stone is placed here
        Returns:
            - current_state (:obj:`array`):
                the 0 dim means which positions is occupied by self.current_player,
                the 1 dim indicates which positions are occupied by self.next_player,
                the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2
        """
        board_curr_player = np.where(self.board == self.current_player, 1, 0)
        board_opponent_player = np.where(self.board == self.next_player, 1, 0)
        board_to_play = np.full((self.board_size, self.board_size), self.current_player)
        raw_obs = np.array([board_curr_player, board_opponent_player, board_to_play], dtype=np.float32)
        if self.scale:
            scale_obs = copy.deepcopy(raw_obs / 2)
        else:
            scale_obs = copy.deepcopy(raw_obs)
        if self.channel_last:
            # move channel dim to last axis
            # (C, W, H) -> (W, H, C)
            return np.transpose(raw_obs, [1, 2, 0]), np.transpose(scale_obs, [1, 2, 0])
        else:
            # (C, W, H)
            return raw_obs, scale_obs

    def get_done_reward(self):
        """
        Overview:
             Check if the game is over and what is the reward in the perspective of player 1.
             Return 'done' and 'reward'.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'reward',
                - if player 1 win,     'done' = True, 'reward' = 1
                - if player 2 win,     'done' = True, 'reward' = -1
                - if draw,             'done' = True, 'reward' = 0
                - if game is not over, 'done' = False,'reward' = None
        """
        done, winner = self.get_done_winner()
        if winner == 1:
            reward = 1
        elif winner == 2:
            reward = -1
        elif winner == -1 and done:
            reward = 0
        elif winner == -1 and not done:
            # episode is not done
            reward = None
        return done, reward

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def bot_action(self):
        action_type = self._get_bot_action_type()
        if action_type == 'random':
            return self.random_action()
        if action_type == 'v0':
            return self.rule_bot_v0()
        if action_type == 'heuristic_perfect':
            return self.rule_bot_heuristic_perfect()
        elif action_type == 'alpha_beta_pruning':
            return self.bot_action_alpha_beta_pruning()
        else:
            raise NotImplementedError

    def bot_action_alpha_beta_pruning(self):
        action = self.alpha_beta_pruning_player.get_best_action(self.board, player_index=self.current_player_index)
        return action

    def rule_bot_v0(self):
        """
        Overview:
            Hard coded expert agent for tictactoe env.
            First random sample a action from legal_actions, then take the action that will lead a connect3 of current player's pieces.
        Returns:
            - action (:obj:`int`): the expert action to take in the current game state.
        """
        # To easily calculate expert action, we convert the chessboard notation:
        # from player 1:  1, player 2: 2
        # to   player 1: -1, player 2: 1
        # TODO: more elegant implementation
        board = copy.deepcopy(self.board)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] == 1:
                    board[i][j] = -1
                elif board[i][j] == 2:
                    board[i][j] = 1

        # first random sample a action from legal_actions
        action = np.random.choice(self.legal_actions)

        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                # if i-th horizontal line has two same pieces and one empty position
                # find the index in the i-th horizontal line
                ind = np.where(board[i, :] == 0)[0][0]
                # convert ind to action
                action = np.ravel_multi_index((np.array([i]), np.array([ind])), (3, 3))[0]
                if self.current_player_to_compute_bot_action * sum(board[i, :]) > 0:
                    # only take the action that will lead a connect3 of current player's pieces
                    return action

            if abs(sum(board[:, i])) == 2:
                # if i-th vertical line has two same pieces and one empty position
                # find the index in the i-th vertical line
                ind = np.where(board[:, i] == 0)[0][0]
                # convert ind to action
                action = np.ravel_multi_index((np.array([ind]), np.array([i])), (3, 3))[0]
                if self.current_player_to_compute_bot_action * sum(board[:, i]) > 0:
                    # only take the action that will lead a connect3 of current player's pieces
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = np.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            # if diagonal has two same pieces and one empty position
            # find the index in the diag vector
            ind = np.where(diag == 0)[0][0]
            # convert ind to action
            action = np.ravel_multi_index((np.array([ind]), np.array([ind])), (3, 3))[0]
            if self.current_player_to_compute_bot_action * sum(diag) > 0:
                # only take the action that will lead a connect3 of current player's pieces
                return action

        if abs(sum(anti_diag)) == 2:
            # if anti-diagonal has two same pieces and one empty position
            # find the index in the anti_diag vector
            ind = np.where(anti_diag == 0)[0][0]
            # convert ind to action
            action = np.ravel_multi_index((np.array([ind]), np.array([2 - ind])), (3, 3))[0]
            if self.current_player_to_compute_bot_action * sum(anti_diag) > 0:
                # only take the action that will lead a connect3 of current player's pieces
                return action

        return action

    def _find_winning_moves(self, board, player):
        moves = []
        for action in self.legal_actions:
            row, col = self.action_to_coord(action)
            if board[row, col] != 0:
                continue
            board_copy = copy.deepcopy(board)
            board_copy[row, col] = player
            done, winner = _get_done_winner_func_lru(tuple(map(tuple, board_copy)))
            if done and winner == player:
                moves.append(action)
        return moves

    def _find_fork_moves(self, board, player):
        fork_moves = []
        for action in self.legal_actions:
            row, col = self.action_to_coord(action)
            if board[row, col] != 0:
                continue
            board_copy = copy.deepcopy(board)
            board_copy[row, col] = player
            winning_moves = self._find_winning_moves(board_copy, player)
            if len(winning_moves) >= 2:
                fork_moves.append(action)
        return fork_moves

    def _symmetry_maps(self):
        if hasattr(self, "_symmetry_maps_cache"):
            return self._symmetry_maps_cache
        index_board = np.arange(9).reshape(3, 3)
        maps = []
        for k in range(4):
            rot = np.rot90(index_board, k)
            maps.append(rot)
            maps.append(np.fliplr(rot))
        transform_maps = []
        for transformed in maps:
            flat = transformed.flatten()
            forward = np.empty(9, dtype=np.int64)
            for i in range(9):
                forward[i] = int(np.where(flat == i)[0][0])
            inverse = np.empty(9, dtype=np.int64)
            for i in range(9):
                inverse[forward[i]] = i
            transform_maps.append(
                {
                    "forward": forward,
                    "inverse": inverse,
                }
            )
        self._symmetry_maps_cache = transform_maps
        return transform_maps

    def _apply_symmetry(self, board, transform_index):
        k = transform_index // 2
        rot = np.rot90(board, k)
        if transform_index % 2 == 1:
            return np.fliplr(rot)
        return rot

    def _normalize_board_for_player(self, board, player):
        if player == 1:
            return board
        board_copy = copy.deepcopy(board)
        board_copy[board_copy == 1] = 3
        board_copy[board_copy == 2] = 1
        board_copy[board_copy == 3] = 2
        return board_copy

    def _canonicalize_board(self, board):
        best = None
        best_index = 0
        for i in range(8):
            transformed = self._apply_symmetry(board, i)
            flat = tuple(transformed.flatten().tolist())
            if best is None or flat < best:
                best = flat
                best_index = i
        best_board = np.array(best, dtype=np.int32).reshape(3, 3)
        return best_board, best_index

    def _heuristic_opening_move(self):
        normalized = self._normalize_board_for_player(self.board, self.current_player)
        player_count = int(np.sum(normalized == 1))
        opponent_count = int(np.sum(normalized == 2))
        if player_count != 1 or opponent_count != 1:
            return None

        canonical, transform_index = self._canonicalize_board(normalized)
        flat = canonical.flatten()
        pos_player = int(np.where(flat == 1)[0][0])
        pos_opponent = int(np.where(flat == 2)[0][0])

        center = 4
        corners = [0, 2, 6, 8]
        edges = [1, 3, 5, 7]
        opposite_corner = {0: 8, 2: 6, 6: 2, 8: 0}
        edge_to_corners = {
            1: [0, 2],
            3: [0, 6],
            5: [2, 8],
            7: [6, 8],
        }
        corner_to_edges = {
            0: [1, 3],
            2: [1, 5],
            6: [3, 7],
            8: [5, 7],
        }

        action_canonical = None
        if pos_player == center:
            if pos_opponent in corners:
                action_canonical = opposite_corner[pos_opponent]
            elif pos_opponent in edges:
                action_canonical = edge_to_corners[pos_opponent][0]
        elif pos_player in corners:
            if pos_opponent == center:
                opp = opposite_corner[pos_player]
                preferred_edges = corner_to_edges[opp]
                action_canonical = preferred_edges[0]
            elif pos_opponent in corners:
                if pos_opponent == opposite_corner[pos_player]:
                    row = pos_player // 3
                    action_canonical = row * 3 + (2 - (pos_player % 3))
                else:
                    action_canonical = center
            elif pos_opponent in edges:
                adjacent_edges = corner_to_edges[pos_player]
                if pos_opponent in adjacent_edges:
                    candidates = edge_to_corners[pos_opponent]
                    candidates = [c for c in candidates if c != pos_player]
                    action_canonical = candidates[0] if candidates else None
                else:
                    adjacent_corners = edge_to_corners[pos_opponent]
                    candidates = [c for c in corners if c not in adjacent_corners and c != pos_player]
                    action_canonical = candidates[0] if candidates else None

        if action_canonical is None:
            return None
        transform_maps = self._symmetry_maps()
        action = int(transform_maps[transform_index]["inverse"][action_canonical])
        if action in self.legal_actions:
            return action
        return None

    def _maybe_print_heuristic(self, action, reason):
        if not self._cfg.get('heuristic_print', False):
            return
        def _render(board):
            symbol_map = {0: ".", 1: "X", 2: "O"}
            lines = []
            for row in board:
                line = " ".join(symbol_map[int(cell)] for cell in row)
                lines.append(line)
            return "\n".join(lines)
        print("TicTacToe heuristic board:")
        print(_render(self.board))
        print(f"player={self.current_player} action={action} reason={reason}")

    def rule_bot_heuristic_perfect(self):
        """
        Overview:
            Deterministic heuristic for perfect TicTacToe play.
            Strategy order: win, block, fork, block fork, center, opposite corner, empty corner, empty side.
        """
        opening_action = self._heuristic_opening_move()
        if opening_action is not None:
            self._maybe_print_heuristic(opening_action, "opening_book")
            return opening_action
        player = self.current_player
        opponent = self.next_player

        winning_moves = self._find_winning_moves(self.board, player)
        if winning_moves:
            action = winning_moves[0]
            self._maybe_print_heuristic(action, "win")
            return action

        blocking_moves = self._find_winning_moves(self.board, opponent)
        if blocking_moves:
            action = blocking_moves[0]
            self._maybe_print_heuristic(action, "block")
            return action

        fork_moves = self._find_fork_moves(self.board, player)
        if fork_moves:
            action = fork_moves[0]
            self._maybe_print_heuristic(action, "fork")
            return action

        opponent_forks = self._find_fork_moves(self.board, opponent)
        if opponent_forks:
            center = self.coord_to_action(1, 1)
            corners = [self.coord_to_action(0, 0), self.coord_to_action(0, 2),
                       self.coord_to_action(2, 0), self.coord_to_action(2, 2)]
            if (center not in self.legal_actions and self.board[1, 1] == opponent and
                    any(self.board[r, c] == player for r, c in [(0, 0), (0, 2), (2, 0), (2, 2)]) and
                    any(self.board[r, c] == opponent for r, c in [(0, 0), (0, 2), (2, 0), (2, 2)])):
                for corner in corners:
                    if corner in self.legal_actions:
                        self._maybe_print_heuristic(corner, "block_fork_corner")
                        return corner
            if len(opponent_forks) == 1:
                action = opponent_forks[0]
                self._maybe_print_heuristic(action, "block_fork_single")
                return action
            # Try to create a threat to force a response.
            for action in self.legal_actions:
                row, col = self.action_to_coord(action)
                if self.board[row, col] != 0:
                    continue
                board_copy = copy.deepcopy(self.board)
                board_copy[row, col] = player
                if self._find_winning_moves(board_copy, player):
                    self._maybe_print_heuristic(action, "block_fork_threat")
                    return action
            action = opponent_forks[0]
            self._maybe_print_heuristic(action, "block_fork_fallback")
            return action

        center = self.coord_to_action(1, 1)
        if center in self.legal_actions:
            self._maybe_print_heuristic(center, "center")
            return center

        corners = [self.coord_to_action(0, 0), self.coord_to_action(0, 2),
                   self.coord_to_action(2, 0), self.coord_to_action(2, 2)]
        opposite = {
            self.coord_to_action(0, 0): self.coord_to_action(2, 2),
            self.coord_to_action(2, 2): self.coord_to_action(0, 0),
            self.coord_to_action(0, 2): self.coord_to_action(2, 0),
            self.coord_to_action(2, 0): self.coord_to_action(0, 2),
        }
        for corner in corners:
            r, c = self.action_to_coord(corner)
            if self.board[r, c] == opponent:
                opp_corner = opposite[corner]
                if opp_corner in self.legal_actions:
                    self._maybe_print_heuristic(opp_corner, "opposite_corner")
                    return opp_corner

        for corner in corners:
            if corner in self.legal_actions:
                self._maybe_print_heuristic(corner, "empty_corner")
                return corner

        sides = [self.coord_to_action(0, 1), self.coord_to_action(1, 0),
                 self.coord_to_action(1, 2), self.coord_to_action(2, 1)]
        for side in sides:
            if side in self.legal_actions:
                self._maybe_print_heuristic(side, "empty_side")
                return side

        action = self.random_action()
        self._maybe_print_heuristic(action, "random_fallback")
        return action

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_player_index(self):
        """
        Overview:
            current_player_index = 0, current_player = 1
            current_player_index = 1, current_player = 2
        """
        return 0 if self._current_player == 1 else 1

    @property
    def next_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def current_player_to_compute_bot_action(self):
        """
        Overview: to compute expert action easily.
        """
        return -1 if self.current_player == 1 else 1

    def set_eval_bot(self, bot_action_type: str) -> None:
        """Set bot type for eval_mode only."""
        self._eval_bot_action_type = bot_action_type

    def set_eval_agent_role(self, role: str) -> None:
        """Set agent role for eval_mode only: 'first' or 'second'."""
        if role not in ['first', 'second']:
            raise ValueError(f"Invalid eval agent role: {role}")
        self._eval_agent_role = role

    def _get_bot_action_type(self) -> str:
        if self.battle_mode == 'eval_mode' and self._eval_bot_action_type is not None:
            return self._eval_bot_action_type
        return self.bot_action_type

    def _select_eval_matchup(self) -> None:
        if not self._eval_matchups:
            self._eval_bot_action_type = self.bot_action_type
            self._eval_agent_role = 'first'
            self._eval_matchup_name = f"{self._eval_bot_action_type}_first"
            return
        matchup = self._eval_matchups[self._eval_matchup_index % len(self._eval_matchups)]
        self._eval_matchup_index += 1
        bot_name = matchup.get('bot', 'bot')
        bot_action_type = matchup.get('action_type', self.bot_action_type)
        agent_role = matchup.get('agent_role', 'first')
        self.set_eval_bot(bot_action_type)
        self.set_eval_agent_role(agent_role)
        self._eval_matchup_name = f"{bot_name}_{agent_role}"

    def _eval_agent_return(self) -> float:
        done, winner = self.get_done_winner()
        if not done or winner == -1:
            return 0.0
        agent_player = 1 if self._eval_agent_role == 'first' else 2
        return 1.0 if winner == agent_player else -1.0

    def _set_eval_episode_info(self, timestep: BaseEnvTimestep) -> None:
        if self.battle_mode != 'eval_mode' or self._eval_matchup_name is None:
            return
        episode_info = timestep.info.get('episode_info', {})
        episode_info['eval_matchup'] = self._eval_matchup_name
        episode_info['eval_bot'] = self._get_bot_action_type()
        episode_info['eval_agent_role'] = self._eval_agent_role
        timestep.info['episode_info'] = episode_info

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        print(self.board)
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1, 2, or 3, from up to bottom) to play for the player {self.current_player}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3, from left to right) to play for the player {self.current_player}: "
                    )
                )
                choice = self.coord_to_action(row - 1, col - 1)
                if (choice in self.legal_actions and 1 <= row and 1 <= col and row <= self.board_size
                        and col <= self.board_size):
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def coord_to_action(self, i, j):
        """
        Overview:
            convert coordinate i, j to action index a in [0, board_size**2)
        """
        return i * self.board_size + j

    def action_to_coord(self, a):
        """
        Overview:
            convert action index a in [0, board_size**2) to coordinate (i, j)
        """
        return a // self.board_size, a % self.board_size

    def action_to_string(self, action_number):
        """
        Overview:
            Convert an action number to a string representing the action.
        Arguments:
            - action_number: an integer from the action space.
        Returns:
            - String representing the action.
        """
        row = action_number // self.board_size + 1
        col = action_number % self.board_size + 1
        return f"Play row {row}, column {col}"

    def simulate_action(self, action):
        """
        Overview:
            execute action and get next_simulator_env. used in AlphaZero.
        Arguments:
            - action: an integer from the action space.
        Returns:
            - next_simulator_env: next simulator env after execute action.
        """
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        new_board = copy.deepcopy(self.board)
        row, col = self.action_to_coord(action)
        new_board[row, col] = self.current_player
        if self.start_player_index == 0:
            start_player_index = 1  # self.players = [1, 2], start_player = 2, start_player_index = 1
        else:
            start_player_index = 0  # self.players = [1, 2], start_player = 1, start_player_index = 0
        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.reset(start_player_index, init_state=new_board)
        return next_simulator_env

    def simulate_action_v2(self, board, start_player_index, action):
        """
        Overview:
            execute action from board and get new_board, new_legal_actions. used in alphabeta_pruning_bot.
        Arguments:
            - board (:obj:`np.array`): current board
            - start_player_index (:obj:`int`): start player index
            - action (:obj:`int`): action
        Returns:
            - new_board (:obj:`np.array`): new board
            - new_legal_actions (:obj:`list`): new legal actions
        """
        self.reset(start_player_index, init_state=board)
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        row, col = self.action_to_coord(action)
        self.board[row, col] = self.current_player
        new_legal_actions = copy.deepcopy(self.legal_actions)
        new_board = copy.deepcopy(self.board)

        return new_board, new_legal_actions

    def render(self, mode="human"):
        """
        Render the game state, either as a string (mode='human') or as an RGB image (mode='rgb_array').

        Arguments:
            - mode (:obj:`str`): The mode to render with. Valid modes are:
                - 'human': render to the current display or terminal and
                - 'rgb_array': Return an numpy.ndarray with shape (x, y, 3),
                  representing RGB values for an image of the board
        Returns:
            if mode is:
            - 'human': returns None
            - 'rgb_array': return a numpy array representing the rendered image.
        Raises:
            ValueError: If the provided mode is unknown.
        """
        if mode == 'human':
            print(self.board)
        elif mode == 'rgb_array':
            dpi = 80
            fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

            # """Piece is in the cross point of row and col"""
            # # Draw a black background, white grid
            # ax.imshow(np.zeros((self.board_size, self.board_size, 3)), origin='lower')
            # ax.grid(color='white', linewidth=2)
            #
            # # Draw the 'X' and 'O' symbols for each player
            # for i in range(self.board_size):
            #     for j in range(self.board_size):
            #         if self.board[i, j] == 1:  # Player 1
            #             ax.text(j, i, 'X', ha='center', va='center', color='white', fontsize=24)
            #         elif self.board[i, j] == 2:  # Player 2
            #             ax.text(j, i, 'O', ha='center', va='center', color='white', fontsize=24)

            # # Setup the axes
            # ax.set_xticks(np.arange(self.board_size))
            # ax.set_yticks(np.arange(self.board_size))

            """Piece is in the center point of grid"""
            # Draw a peachpuff background, black grid
            ax.imshow(np.ones((self.board_size, self.board_size, 3)) * np.array([255, 218, 185]) / 255, origin='lower')
            ax.grid(color='black', linewidth=2)

            # Draw the 'X' and 'O' symbols for each player
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i, j] == 1:  # Player 1
                        ax.text(j, i, 'X', ha='center', va='center', color='black', fontsize=24)
                    elif self.board[i, j] == 2:  # Player 2
                        ax.text(j, i, 'O', ha='center', va='center', color='white', fontsize=24)

            # Setup the axes
            ax.set_xticks(np.arange(0.5, self.board_size, 1))
            ax.set_yticks(np.arange(0.5, self.board_size, 1))

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')

            # Set the title of the game
            plt.title('TicTacToe: ' + ('Black Turn' if self.current_player == 1 else 'White Turn'))

            fig.canvas.draw()

            # Get the width and height of the figure
            width, height = fig.get_size_inches() * fig.get_dpi()
            width = int(width)
            height = int(height)

            # Use the width and height values to reshape the numpy array
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(height, width, 3)

            plt.close(fig)

            return img
        else:
            raise ValueError(f"Unknown mode '{mode}', it should be either 'human' or 'rgb_array'.")

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        import imageio
        imageio.mimsave(path, frames, fps=20)

    @staticmethod
    def display_frames_as_mp4(frames: list, path: str, fps=5) -> None:
        assert path.endswith('.mp4'), f'path must end with .mp4, but got {path}'
        import imageio
        imageio.mimwrite(path, frames, fps=fps)

    def clone(self):
        return copy.deepcopy(self)

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

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # In eval phase, we use ``eval_mode`` to make agent play with the built-in bot to
        # evaluate the performance of the current agent.
        cfg.battle_mode = 'eval_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero TicTacToe Env"

    def close(self) -> None:
        pass
