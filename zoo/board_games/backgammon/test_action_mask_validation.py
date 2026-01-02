"""
Action mask validation tests against known backgammon positions.

These tests verify that the action encoding/decoding matches expected
backgammon rules for specific board positions.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from easydict import EasyDict
from zoo.board_games.backgammon.envs.backgammon_env import BackgammonEnv


def decode_action(action):
    """Convert action index to (source, die_slot) tuple."""
    src = action // 2
    die_slot = action % 2
    return src, die_slot


def get_legal_moves_by_slot(legal_actions):
    """Group legal actions by die slot."""
    moves_by_slot = {0: set(), 1: set()}
    for a in legal_actions:
        src, slot = decode_action(a)
        moves_by_slot[slot].add(src)
    return moves_by_slot


def get_legal_sources(legal_actions):
    """Get set of all legal source points."""
    sources = set()
    for a in legal_actions:
        src, _ = decode_action(a)
        sources.add(src)
    return sources


class TestOpeningPosition:
    """Test action mask for standard opening position."""

    def test_opening_with_6_1(self):
        """Opening position with dice [6, 1] - classic opening roll."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Standard opening: White pieces at points 24(2), 13(5), 8(3), 6(5)
        # In 0-indexed: 23(2), 12(5), 7(3), 5(5)
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 23] = 2  # White 24-point
        board[0, 12] = 5  # White 13-point
        board[0, 7] = 3   # White 8-point
        board[0, 5] = 5   # White 6-point

        board[1, 0] = 2   # Black 1-point
        board[1, 11] = 5  # Black 12-point
        board[1, 16] = 3  # Black 17-point
        board[1, 18] = 5  # Black 19-point

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([6, 1])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Get legal source points
        legal_sources = get_legal_sources(legal_actions)

        # Points with pieces that can move
        points_with_pieces = {23, 12, 7, 5}

        # All moves should be from points with pieces
        assert legal_sources.issubset(points_with_pieces)

        # Should have moves (not empty)
        assert len(legal_actions) > 0

    def test_opening_with_3_1(self):
        """Opening position with dice [3, 1] - common opening."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 23] = 2
        board[0, 12] = 5
        board[0, 7] = 3
        board[0, 5] = 5

        board[1, 0] = 2
        board[1, 11] = 5
        board[1, 16] = 3
        board[1, 18] = 5

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([3, 1])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Should have valid moves
        assert len(legal_actions) > 0

        # Verify all moves are from valid source points
        legal_sources = get_legal_sources(legal_actions)
        assert legal_sources.issubset({23, 12, 7, 5, 24})  # Points + Bar


class TestBarEntry:
    """Test that pieces on bar must enter first."""

    def test_must_enter_from_bar(self):
        """When piece is on bar, all moves must be from bar."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 5] = 14  # 14 white pieces on 6-point
        board[1, 18] = 15  # Black far away

        bar = np.zeros((2,), dtype=np.int8)
        bar[0] = 1  # One white on bar

        env.game.debug_reset_board(board, bar)
        env.game.set_turn(0)
        env.set_dice([3, 2])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # All moves MUST be from bar (src=24)
        for a in legal_actions:
            src, die = decode_action(a)
            assert src == 24, f"With piece on bar, must enter from bar, got src={src}"

    def test_bar_blocked_entry(self):
        """When bar entry points are blocked, no moves available."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 5] = 14
        # Block all entry points for dice [1, 2]
        board[1, 23] = 2  # Block 24-point (entry for die 1)
        board[1, 22] = 2  # Block 23-point (entry for die 2)
        board[1, 18] = 11

        bar = np.zeros((2,), dtype=np.int8)
        bar[0] = 1

        env.game.debug_reset_board(board, bar)
        env.game.set_turn(0)
        env.set_dice([1, 2])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Should have no legal moves (both entry points blocked)
        assert len(legal_actions) == 0

    def test_bar_only_higher_die_enters(self):
        """If only one die can enter from bar, it must be used."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 5] = 14
        # Block entry for die 1 (24-point), leave entry for die 5 open.
        board[1, 23] = 2
        board[1, 18] = 13

        bar = np.zeros((2,), dtype=np.int8)
        bar[0] = 1

        env.game.debug_reset_board(board, bar)
        env.game.set_turn(0)
        env.set_dice([5, 1])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        slot0_die, slot1_die = env._get_dice_slots()
        assert slot0_die == 5 and slot1_die == 1

        # All legal actions should use the die-5 slot only.
        used_slots = {decode_action(a)[1] for a in legal_actions}
        assert used_slots == {0}


class TestBlockedPoints:
    """Test that moves to blocked points are not allowed."""

    def test_cannot_land_on_blocked_point(self):
        """Cannot move to a point with 2+ opponent pieces."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 12] = 5   # White at 13-point
        board[1, 8] = 2    # Black blocks 9-point
        board[1, 7] = 2    # Black blocks 8-point
        board[1, 18] = 11

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([4, 5])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # With dice [4, 5] from point 12:
        # 12-4=8 (blocked), 12-5=7 (blocked)
        # So point 12 should NOT be in legal sources
        legal_sources = get_legal_sources(legal_actions)
        assert 12 not in legal_sources, "Should not allow moves from 12 when both destinations blocked"

    def test_can_hit_blot(self):
        """Can land on a point with exactly 1 opponent piece (hit)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 12] = 5   # White at 13-point
        board[1, 8] = 1    # Black blot at 9-point (hittable!)
        board[1, 18] = 14

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([4, 3])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # 12-4=8, hitting the blot - should be legal
        # Point 12 should be in legal sources
        legal_sources = get_legal_sources(legal_actions)
        assert 12 in legal_sources, "Should allow hitting blot from point 12"


class TestBearingOff:
    """Test bearing off mechanics."""

    def test_can_bear_off_when_all_in_home(self):
        """Can bear off when all pieces are in home board."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 5] = 5   # White at 6-point
        board[0, 4] = 5   # White at 5-point
        board[0, 3] = 5   # White at 4-point
        board[1, 23] = 15

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([6, 5])

        assert env.game.can_bear_off(), "Should be able to bear off"

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Should have legal moves
        assert len(legal_actions) > 0

    def test_cannot_bear_off_with_pieces_outside(self):
        """Cannot bear off when pieces are outside home board."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 5] = 5   # White at 6-point (home)
        board[0, 4] = 5   # White at 5-point (home)
        board[0, 10] = 5  # White at 11-point (NOT home!)
        board[1, 23] = 15

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([6, 5])

        assert not env.game.can_bear_off(), "Should NOT be able to bear off"

    def test_can_overshoot_bear_off_if_no_pieces_behind(self):
        """Can bear off with a higher die if no pieces are behind."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        # White home board only, no pieces on 6-point (index 5).
        board[0, 4] = 3  # White at 5-point
        board[0, 0] = 12  # White stacked at 1-point
        board[1, 23] = 15

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([6, 1])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Die 6 is slot 0; allow bearing off from point 4 (index 4).
        expected_action = 4 * 2 + 0
        assert expected_action in legal_actions


class TestHigherDieRule:
    """Test the 'must play higher die' rule."""

    def test_must_play_higher_die_when_only_one_playable(self):
        """When only one die can be played, must play the higher one.

        The higher die rule applies when you can only play ONE of your dice
        (not both) due to blocking. In this case, you must play the higher one.
        """
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Setup: Single white piece at point 12
        # - Die 4 destination (point 8) is blocked by 2 black pieces
        # - Die 5 destination (point 7) is open
        # So only the higher die (5) can be played
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 12] = 1   # Single white piece at point 13 (index 12)
        board[0, 0] = 14   # Rest of white in home (doesn't affect this move)
        board[1, 8] = 2    # Block point 9 (index 8) - 4-move destination
        board[1, 18] = 13  # Other black pieces

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([5, 4])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Should have legal moves from point 12
        legal_sources = get_legal_sources(legal_actions)
        assert 12 in legal_sources, "Should be able to move from point 12"

        # Verify dice slots
        slot0_die, slot1_die = env._get_dice_slots()
        assert slot0_die == 5 and slot1_die == 4

        # Only moves from point 12 should use slot 0 (higher die 5)
        moves_from_12 = [a for a in legal_actions if decode_action(a)[0] == 12]
        slots_from_12 = {decode_action(a)[1] for a in moves_from_12}
        assert slots_from_12 == {0}, f"Point 12 should only use slot 0, got slots {slots_from_12}"

    def test_higher_die_with_lookahead(self):
        """When both dice can move initially but only one allows a sequence.

        This tests the look-ahead version of the higher die rule:
        - Single piece can move with either die 3 or die 4
        - But after either move, the second die is blocked
        - So only ONE die can be played total
        - Engine must offer only the higher die (4)
        """
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Setup: White piece at index 10
        # - Move 4: 10->6, then move 3: 6->3 (blocked)
        # - Move 3: 10->7, then move 4: 7->3 (blocked)
        # Both initial moves are possible, but second die always blocked at index 3
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 10] = 1   # Single white piece at point 11 (index 10)
        board[0, 0] = 14   # Rest of white in home
        board[1, 3] = 2    # Block point 4 (index 3) - blocks second move in both sequences
        board[1, 20] = 13  # Other black pieces

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([4, 3])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Verify dice slots
        slot0_die, slot1_die = env._get_dice_slots()
        assert slot0_die == 4 and slot1_die == 3

        # Should have exactly one legal action: move from 10 with higher die (4)
        assert len(legal_actions) == 1, f"Expected 1 legal action, got {len(legal_actions)}"

        # That action should be source=10, slot=0 (die 4)
        expected_action = 10 * 2 + 0  # = 20
        assert legal_actions[0] == expected_action, \
            f"Expected action {expected_action}, got {legal_actions[0]}"


class TestSingleDieOnly:
    """Test positions where only one die is playable."""

    def test_only_lower_die_playable(self):
        """When higher die is blocked, only lower die actions should appear."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 6] = 1   # White at 7-point
        board[1, 2] = 2   # Block 7-4=3-point (index 2)
        board[1, 18] = 13

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([4, 1])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        slot0_die, slot1_die = env._get_dice_slots()
        assert slot0_die == 4 and slot1_die == 1
        used_slots = {decode_action(a)[1] for a in legal_actions}
        assert used_slots == {1}


class TestDoubles:
    """Test doubles handling in the action mask."""

    def test_doubles_expose_both_slots(self):
        """For doubles, both slots should be valid for the same die."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 10] = 1
        board[1, 18] = 15

        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([4, 4, 4, 4])

        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # Both slots should be enabled for any legal source.
        sources = get_legal_sources(legal_actions)
        for src in sources:
            assert (src * 2 + 0) in legal_actions
            assert (src * 2 + 1) in legal_actions
        assert len(legal_actions) > 0, "Should have legal moves"


class TestActionEncoding:
    """Test action encoding/decoding consistency."""

    def test_action_roundtrip(self):
        """Action encoding is reversible."""
        for src in range(25):  # 0-23 points + 24 bar
            for slot in range(2):  # 2 dice slots
                action = src * 2 + slot
                decoded_src, decoded_slot = decode_action(action)
                assert decoded_src == src
                assert decoded_slot == slot

    def test_action_range(self):
        """All valid actions are in range [0, 50)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)

        for _ in range(10):
            env.reset()
            obs = env.observe()
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

            for a in legal_actions:
                assert 0 <= a < 50, f"Action {a} out of range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
