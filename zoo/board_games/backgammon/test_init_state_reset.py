import numpy as np

from zoo.board_games.backgammon.envs.backgammon_env import BackgammonEnv


def test_reset_with_init_state_dict():
    env = BackgammonEnv(BackgammonEnv.default_config())
    env.reset(start_player_index=0)
    env.set_dice([3, 5])
    init_state = {
        "board": env.game.get_board().tolist(),
        "bar": env.game.get_bar().tolist(),
        "off": env.game.get_beared_off().tolist(),
        "turn": int(env.game.get_player_turn()),
        "dice": list(env.game.get_remaining_dice()),
        "nature_turn": bool(env.game.is_nature_turn()),
    }
    env.reset(start_player_index=0, init_state=init_state)
    np.testing.assert_array_equal(env.game.get_board(), np.array(init_state["board"], dtype=np.int8))
    np.testing.assert_array_equal(env.game.get_bar(), np.array(init_state["bar"], dtype=np.int8))
    np.testing.assert_array_equal(env.game.get_beared_off(), np.array(init_state["off"], dtype=np.int8))
    assert env.game.get_player_turn() == init_state["turn"]
    assert list(env.game.get_remaining_dice()) == init_state["dice"]
