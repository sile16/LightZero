import numpy as np

from zoo.board_games.pig.envs.pig_env import PigEnv


def test_chance_space_size_and_reset_chance():
    env = PigEnv()
    obs = env.reset()
    assert env.chance_space_size == 7
    assert obs['chance'] == 0


def test_roll_sets_nonzero_chance():
    np.random.seed(123)
    env = PigEnv()
    env.reset()
    timestep = env.step(0)  # roll
    assert 1 <= timestep.obs['chance'] <= 6


def test_hold_sets_no_roll_chance():
    env = PigEnv()
    env.reset()
    env.turn_score = 5
    timestep = env.step(1)  # hold
    assert timestep.obs['chance'] == 0
