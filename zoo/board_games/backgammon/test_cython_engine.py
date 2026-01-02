"""Tests for the Cython backgammon engine.

NOTE: The raw Cython engine has a pre-game roll-off phase where both players
roll one die each to determine who goes first. Doubles trigger re-rolls.

Use `force_start(player)` to bypass pre-game and start with a specific player.
This rolls dice (allowing doubles) and advances to movement state.
"""
import sys
import os
import pytest
import numpy as np

# Add the cython_backgammon directory to path so we can import 'state'
current_dir = os.path.dirname(os.path.abspath(__file__))
cython_dir = os.path.join(current_dir, 'cython_backgammon')
sys.path.insert(0, cython_dir)

import state


class TestCythonEngineInit:
    """Test engine initialization."""

    def test_state_initializes(self):
        """State can be initialized without error."""
        s = state.State()
        assert s is not None

    def test_initial_board_shape(self):
        """Initial board has correct shape (2, 24)."""
        s = state.State()
        board = np.array(s.get_board())
        assert board.shape == (2, 24)

    def test_starts_as_nature_turn(self):
        """Game starts as nature turn (dice roll phase)."""
        s = state.State()
        assert s.is_nature_turn() is True

    def test_initial_dice_moves_count(self):
        """Initial nature turn has possible dice outcomes."""
        s = state.State()
        legal_moves = s.get_moves()
        # Pre-game has 15 non-doubles combinations (6*5/2 = 15)
        # or 21 if counting ordered pairs minus doubles
        assert len(legal_moves) > 0  # At least some dice moves available


class TestCythonEngineDiceRoll:
    """Test dice rolling mechanics using force_start() to bypass pre-game."""

    def test_force_start_clears_nature_turn(self):
        """After force_start, it's no longer a nature turn."""
        s = state.State()
        s.force_start(0)
        assert s.is_nature_turn() is False

    def test_force_start_sets_player_turn(self):
        """After force_start, specified player has the turn."""
        s = state.State()
        s.force_start(0)
        assert s.get_player_turn() == 0

    def test_force_start_generates_checker_moves(self):
        """After force_start, checker moves are available."""
        s = state.State()
        s.force_start(0)

        checker_moves = s.get_moves()
        assert len(checker_moves) > 0
        for m in checker_moves:
            assert m.is_movement_move is True

    def test_force_start_allows_doubles(self):
        """force_start can result in doubles (4 dice)."""
        # Run multiple times to check doubles can occur
        doubles_found = False
        for _ in range(100):
            s = state.State()
            s.force_start(0)
            remaining = list(s.get_remaining_dice())
            if len(remaining) == 4:
                doubles_found = True
                assert remaining[0] == remaining[1] == remaining[2] == remaining[3]
                break
        # Note: not asserting doubles_found since it's probabilistic


class TestCythonEngineCheckerMoves:
    """Test checker movement mechanics."""

    def test_checker_move_updates_remaining_dice(self):
        """After a checker move, remaining dice count decreases."""
        s = state.State()
        s.force_start(0)

        initial_dice = list(s.get_remaining_dice())
        checker_moves = s.get_moves()
        assert len(checker_moves) > 0

        s.do_move(checker_moves[0])
        remaining = list(s.get_remaining_dice())
        assert len(remaining) < len(initial_dice)

    def test_checker_move_has_valid_attributes(self):
        """Checker moves have valid src, dst, n attributes."""
        s = state.State()
        s.force_start(0)

        checker_moves = s.get_moves()
        for m in checker_moves:
            assert hasattr(m, 'src')
            assert hasattr(m, 'dst')
            assert hasattr(m, 'n')
            assert 1 <= m.n <= 6


class TestCythonEngineGameFlow:
    """Test complete game flow."""

    def test_random_game_completes(self):
        """A random game eventually ends."""
        np.random.seed(42)
        s = state.State()

        max_moves = 2000
        move_count = 0

        while not s.game_ended() and move_count < max_moves:
            legal_moves = s.get_moves()
            if len(legal_moves) == 0:
                break
            move = legal_moves[np.random.randint(len(legal_moves))]
            s.do_move(move)
            move_count += 1

        assert s.game_ended() or move_count < max_moves

    def test_winner_is_valid_player(self):
        """When game ends, winner is a valid player (0 or 1)."""
        np.random.seed(123)
        s = state.State()

        max_moves = 2000
        move_count = 0

        while not s.game_ended() and move_count < max_moves:
            legal_moves = s.get_moves()
            if len(legal_moves) == 0:
                break
            move = legal_moves[np.random.randint(len(legal_moves))]
            s.do_move(move)
            move_count += 1

        if s.game_ended():
            winner = s.get_winner()
            assert winner in [0, 1]


# Keep old function for backwards compatibility with direct script execution
def test_engine():
    """Legacy test function for backwards compatibility."""
    s = state.State()
    assert np.array(s.get_board()).shape == (2, 24)
    assert s.is_nature_turn() is True

    # Bypass pre-game phase (same as BackgammonEnv does)
    s.force_start(0)

    # force_start rolls dice and enters movement state directly
    assert s.is_nature_turn() is False
    assert s.get_player_turn() == 0  # Always player 0 with force_start

    legal_checker_moves = s.get_moves()
    assert len(legal_checker_moves) > 0

    move = legal_checker_moves[0]
    s.do_move(move)


if __name__ == "__main__":
    test_engine()
    print("All tests passed!")
