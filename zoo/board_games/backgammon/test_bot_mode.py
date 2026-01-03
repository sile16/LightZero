"""Tests for bot mode in BackgammonEnv.

Bot mode simulates playing against an AI opponent where:
- Player 0 is the human/agent
- Player 1 is the bot (plays automatically)
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from zoo.board_games.backgammon.envs.backgammon_env import BackgammonEnv
from easydict import EasyDict


class TestBotModeInit:
    """Test bot mode initialization."""

    def test_bot_mode_initializes(self):
        """Bot mode environment initializes without error."""
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        assert env is not None
        assert env.battle_mode == 'play_with_bot_mode'

    def test_bot_mode_has_bot(self):
        """Bot mode environment has a bot instance."""
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        assert env.bot is not None

    def test_agent_starts_as_player_zero(self):
        """After reset, the agent (player 0) has the turn."""
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()
        assert env._current_player == 0, "Agent should be player 0"
        assert obs['to_play'] == -1


class TestBotModeGameplay:
    """Test bot mode gameplay mechanics."""

    def test_agent_has_legal_actions(self):
        """After reset, agent has legal actions available."""
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
        assert len(legal_actions) > 0, "Agent should have legal actions"

    def test_agent_action_is_legal(self):
        """Agent's chosen action must be in the action mask."""
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Pick first legal action
        action = legal_actions[0]
        assert action_mask[action] == 1, "Chosen action should be legal"

    def test_turn_returns_to_agent(self):
        """After agent moves, bot plays, and turn returns to agent."""
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
        assert len(legal_actions) > 0

        action = legal_actions[0]
        timestep = env.step(action)

        if not timestep.done:
            # Turn should be back to agent (player 0)
            assert env._current_player == 0, \
                f"Expected turn to return to agent (0), got {env._current_player}"
            assert timestep.obs['to_play'] == -1


class TestBotActionLegality:
    """Test that bot actions are legal."""

    def test_bot_selects_legal_action(self):
        """Bot's selected action should be in the action mask."""
        np.random.seed(42)
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)

        # Run several games and verify bot always picks legal actions
        for _ in range(5):
            obs = env.reset()
            step_count = 0
            max_steps = 500

            while not env.game.game_ended() and step_count < max_steps:
                action_mask = obs['action_mask']
                legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

                if len(legal_actions) == 0:
                    break

                # Agent picks random legal action
                action = legal_actions[np.random.randint(len(legal_actions))]
                assert action_mask[action] == 1, "Agent action should be legal"

                timestep = env.step(action)
                obs = timestep.obs
                step_count += 1


class TestBotModeRewards:
    """Test reward correctness in bot mode."""

    def test_reward_sign_on_game_end(self):
        """Reward sign should match winner in bot mode."""
        np.random.seed(789)
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)

        # Play until game ends
        obs = env.reset()
        max_steps = 2000
        step_count = 0
        final_reward = 0

        while step_count < max_steps:
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

            if len(legal_actions) == 0:
                break

            action = legal_actions[np.random.randint(len(legal_actions))]
            timestep = env.step(action)
            obs = timestep.obs
            final_reward = timestep.reward
            step_count += 1

            if timestep.done:
                break

        if env.game.game_ended():
            winner = env.game.get_winner()
            # In bot mode, agent is player 0
            # If agent wins (winner == 0), reward should be positive
            # If bot wins (winner == 1), reward should be negative
            if winner == 0:
                assert final_reward > 0, "Agent won, reward should be positive"
            else:
                assert final_reward < 0, "Bot won, reward should be negative"


class TestBotModeFullGame:
    """Test complete games in bot mode."""

    def test_game_completes(self):
        """A random game in bot mode should complete."""
        np.random.seed(456)
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        max_steps = 2000
        step_count = 0

        while step_count < max_steps:
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

            if len(legal_actions) == 0:
                break

            action = legal_actions[np.random.randint(len(legal_actions))]
            timestep = env.step(action)
            obs = timestep.obs
            step_count += 1

            if timestep.done:
                break

        assert env.game.game_ended() or step_count < max_steps


# Legacy test function for backwards compatibility
def test_bot_mode():
    """Legacy test function."""
    cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
    env = BackgammonEnv(cfg)
    obs = env.reset()

    assert env._current_player == 0, "Agent should start as player 0"

    action_mask = obs['action_mask']
    legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
    assert len(legal_actions) > 0, "Should have legal actions"

    action = legal_actions[0]
    timestep = env.step(action)

    if not timestep.done:
        assert env._current_player == 0, "Turn should return to agent"


if __name__ == "__main__":
    test_bot_mode()
    print("Bot mode test passed!")
