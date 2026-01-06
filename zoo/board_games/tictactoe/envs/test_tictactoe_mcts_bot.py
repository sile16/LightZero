from easydict import EasyDict
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
from zoo.board_games.mcts_bot import MCTSBot

import pytest

cfg = dict(
    prob_random_agent=0,
    prob_expert_agent=0,
    battle_mode='self_play_mode',
    agent_vs_human=False,
    channel_last=True,
    scale=True,
    bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
)


@pytest.mark.envtest
class TestTicTacToeBot:

    def test_tictactoe_self_play_mode_player0_win(self):
        # player_0  num_simulation=1000
        # player_1  num_simulation=1
        # player_0 will win in principle
        env = TicTacToeEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 1', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 2', 1)  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        step = 0
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action, _, _ = player_0.get_actions(state, step=step, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action, _, _ = player_1.get_actions(state, step=step, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            step += 1
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.get_done_winner()[1] == 1

    def test_tictactoe_self_play_mode_player1_win(self):
        # player_0  num_simulation=1
        # player_1  num_simulation=1000
        # player_1 will win in principle
        env = TicTacToeEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 1', 1)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 2', 1000)  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        step = 0
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action, _, _ = player_0.get_actions(state, step=step, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action, _, _ = player_1.get_actions(state, step=step, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            step += 1
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.get_done_winner()[1] == 2

    def test_tictactoe_self_play_mode_draw(self):
        # player_0  num_simulation=1000
        # player_1  num_simulation=1000,
        # two players will draw in principle
        env = TicTacToeEnv(EasyDict(cfg))
        env.reset()
        state = env.board
        player_0 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 1', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 2', 1000)  # player_index = 1, player = 2

        player_index = 0  # player 1 fist
        step = 0
        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action, _, _ = player_0.get_actions(state, step=step, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action, _, _ = player_1.get_actions(state, step=step, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            step += 1
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.get_done_winner()[1] == -1

    def test_tictactoe_self_play_mode_half_case_1(self):
        env = TicTacToeEnv(EasyDict(cfg))
        init_state = [[1, 1, 0], [0, 2, 2], [0, 0, 0]]
        player_0 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 1', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 2', 1000)  # player_index = 1, player = 2
        player_index = 0  # player 1 fist
        step = 0

        env.reset(player_index, init_state)
        state = env.board

        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action, _, _ = player_0.get_actions(state, step=step, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action, _, _ = player_1.get_actions(state, step=step, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            step += 1
            print('#' * 15)
            print(state)
            print('#' * 15)
            row, col = env.action_to_coord(action)
        assert env.get_done_winner()[1] == 1
        assert row == 0, col == 2

    def test_tictactoe_self_play_mode_half_case_2(self):
        env = TicTacToeEnv(EasyDict(cfg))
        init_state = [[1, 0, 1], [0, 0, 2], [2, 0, 1]]
        player_0 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 1', 1000)  # player_index = 0, player = 1
        player_1 = MCTSBot(TicTacToeEnv(EasyDict(cfg)), 'player 2', 1000)  # player_index = 1, player = 2
        player_index = 1  # player 1 fist
        step = 0

        env.reset(player_index, init_state)
        state = env.board

        print('#' * 15)
        print(state)
        print('#' * 15)
        print('\n')
        while not env.get_done_reward()[0]:
            if player_index == 0:
                action, _, _ = player_0.get_actions(state, step=step, player_index=player_index)
                player_index = 1
            else:
                print('-' * 40)
                action, _, _ = player_1.get_actions(state, step=step, player_index=player_index)
                player_index = 0
                print('-' * 40)
            env.step(action)
            state = env.board
            step += 1
            print('#' * 15)
            print(state)
            print('#' * 15)
        assert env.get_done_winner()[1] == 1
