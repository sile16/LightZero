"""Tests for backgammon rule compliance in the Cython engine.

These tests verify core backgammon rules:
1. Higher die rule: When only one die can be played, must play the higher one
2. Must play both rule: Maximize dice usage when possible
"""
import sys
import os
import pytest
import numpy as np

# Add the cython_backgammon directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
cython_dir = os.path.join(current_dir, 'cython_backgammon')
sys.path.insert(0, cython_dir)

import state

WHITE = 0
BLACK = 1


class TestHigherDieRule:
    """Test the 'play higher die when only one can be played' rule."""

    def test_higher_die_required_when_both_blocked_after(self):
        """When both dice lead to blocked positions, must play higher die.

        Setup:
        - White piece at point 21 (index 20)
        - Black blocks point 12 (index 11)
        - Dice: [5, 4]

        Analysis:
        - Playing 4: 20->16, then 5: 16->11 blocked
        - Playing 5: 20->15, then 4: 15->11 blocked
        Both sequences play exactly 1 die, so higher die (5) must be used.
        """
        s = state.State()
        board = np.zeros((2, 24), dtype=np.int8)

        board[WHITE, 20] = 1   # White at index 20
        board[BLACK, 11] = 2   # Block index 11

        s.debug_reset_board(board)
        s.set_turn(WHITE)
        s.set_nature_turn(False)
        s.set_dice([5, 4])
        s.generate_movement_moves()

        moves = s.get_moves()
        die_values_used = {m.n for m in moves}

        # Should only allow die 5 (higher), not die 4
        assert 5 in die_values_used, "Higher die (5) should be playable"
        assert 4 not in die_values_used, "Lower die (4) should NOT be playable when only one die works"


class TestMustPlayBothRule:
    """Test the 'maximize number of dice played' rule."""

    def test_lower_die_allowed_when_enables_both_dice(self):
        """Lower die may be played first if it enables using both dice.

        Setup:
        - White piece at point 7 (index 6)
        - Black blocks point 3 (index 2)
        - Dice: [4, 1]

        Analysis:
        - Playing 4 first: 6->2 blocked (0 moves total)
        - Playing 1 first: 6->5, then 4: 5->1 open (2 moves total)
        Must play 1 first to maximize dice usage.
        """
        s = state.State()
        board = np.zeros((2, 24), dtype=np.int8)

        board[WHITE, 6] = 1    # White at index 6 (point 7)
        board[BLACK, 2] = 2    # Block index 2 (point 3)

        s.debug_reset_board(board)
        s.set_turn(WHITE)
        s.set_nature_turn(False)
        s.set_dice([4, 1])
        s.generate_movement_moves()

        moves = s.get_moves()
        die_values_used = {m.n for m in moves}

        # Should only allow die 1, because playing 4 first blocks everything
        assert 1 in die_values_used, "Die 1 should be playable (enables both dice)"
        assert 4 not in die_values_used, "Die 4 should NOT be playable (blocks continuation)"


class TestCombinedRules:
    """Test interaction between rules."""

    def test_maximize_moves_overrides_higher_die(self):
        """The 'maximize moves' rule takes priority over 'higher die' rule.

        If playing the lower die first allows using both dice,
        but playing the higher die first only allows one die,
        the lower die must be played first.
        """
        s = state.State()
        board = np.zeros((2, 24), dtype=np.int8)

        # Setup where only lower die path works
        board[WHITE, 6] = 1    # White at index 6
        board[BLACK, 2] = 2    # Block index 2 (blocks 4-move from 6)

        s.debug_reset_board(board)
        s.set_turn(WHITE)
        s.set_nature_turn(False)
        s.set_dice([4, 1])
        s.generate_movement_moves()

        moves = s.get_moves()
        die_values_used = {m.n for m in moves}

        # Lower die (1) should be the only option because it maximizes moves
        assert 1 in die_values_used
        assert 4 not in die_values_used


# Legacy functions for backwards compatibility
def setup_board_for_higher_die_rule():
    """Legacy test function."""
    s = state.State()
    board = np.zeros((2, 24), dtype=np.int8)
    board[WHITE, 20] = 1
    board[BLACK, 11] = 2

    s.debug_reset_board(board)
    s.set_turn(WHITE)
    s.set_nature_turn(False)
    s.set_dice([5, 4])
    s.generate_movement_moves()

    moves = s.get_moves()
    die_values = {m.n for m in moves}
    assert 5 in die_values and 4 not in die_values, "Higher die rule failed"


def setup_board_for_must_play_both_rule():
    """Legacy test function."""
    s = state.State()
    board = np.zeros((2, 24), dtype=np.int8)
    board[WHITE, 6] = 1
    board[BLACK, 2] = 2

    s.debug_reset_board(board)
    s.set_turn(WHITE)
    s.set_nature_turn(False)
    s.set_dice([4, 1])
    s.generate_movement_moves()

    moves = s.get_moves()
    die_values = {m.n for m in moves}
    assert 1 in die_values and 4 not in die_values, "Must play both rule failed"


if __name__ == "__main__":
    setup_board_for_higher_die_rule()
    print("Higher die rule: PASS")
    setup_board_for_must_play_both_rule()
    print("Must play both rule: PASS")
    print("All tests passed!")
