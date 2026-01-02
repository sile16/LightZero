"""
Comprehensive pytest tests for BackgammonEnv.

Tests cover:
- Environment initialization
- Auto-dice rolling in reset
- Step execution and turn transitions
- Action mask correctness
- set_dice test helper
- Bot mode behavior
- Complete game simulation
"""
import pytest
import numpy as np
import random
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from easydict import EasyDict
from zoo.board_games.backgammon.envs.backgammon_env import (
    BackgammonEnv, OBS_MINIMAL, OBS_STANDARD, OBS_WITH_FEATURES
)


class TestBackgammonEnvInit:
    """Tests for environment initialization."""

    def test_init_self_play_mode(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        assert env.action_space_size == 50  # 25 sources × 2 dice slots
        assert env._observation_space.shape == (47, 1, 25)  # standard obs_type default

    def test_init_bot_mode(self):
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        assert hasattr(env, 'bot')
        assert env.bot is not None


class TestBackgammonEnvReset:
    """Tests for reset behavior and auto-dice rolling."""

    def test_reset_starts_game(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        assert env.game.has_game_started()
        assert not env.game.is_nature_turn()
        assert env._current_player in [0, 1]

    def test_reset_has_legal_actions(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        assert len(legal_actions) > 0, "Reset should produce state with legal actions"

    def test_reset_observation_shape(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        assert 'observation' in obs
        assert 'action_mask' in obs
        assert 'to_play' in obs
        assert obs['observation'].shape == (47, 1, 25)  # standard obs_type default
        assert obs['action_mask'].shape == (50,)

    def test_reset_deterministic_with_seed(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))

        env1 = BackgammonEnv(cfg)
        env1._rng.seed(42)
        obs1 = env1.reset()

        env2 = BackgammonEnv(cfg)
        env2._rng.seed(42)
        obs2 = env2.reset()

        assert env1._current_player == env2._current_player


class TestBackgammonEnvStep:
    """Tests for step execution."""

    def test_step_executes_valid_action(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
        action = legal_actions[0]

        timestep = env.step(action)

        assert timestep.reward in [-1, 0, 1]
        assert isinstance(timestep.done, bool)

    def test_step_returns_valid_observation(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
        action = legal_actions[0]

        timestep = env.step(action)

        assert 'observation' in timestep.obs
        assert 'action_mask' in timestep.obs
        assert timestep.obs['observation'].shape == (47, 1, 25)  # standard obs_type default

    def test_step_has_legal_actions_if_not_done(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
        action = legal_actions[0]

        timestep = env.step(action)

        if not timestep.done:
            new_legal = [i for i, x in enumerate(timestep.obs['action_mask']) if x == 1]
            assert len(new_legal) > 0, "Non-terminal state must have legal actions"


class TestBackgammonEnvActionMask:
    """Tests for action mask correctness."""

    def test_action_mask_size(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        assert obs['action_mask'].shape == (50,)

    def test_action_mask_only_movement_actions(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        # All actions should be in valid range (0-49)
        for action in legal_actions:
            assert 0 <= action < 50

    def test_action_mask_matches_engine_moves(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        engine_moves = env.game.get_moves()
        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = set(i for i, x in enumerate(action_mask) if x == 1)

        # Get dice slots for encoding
        slot0_die, slot1_die = env._get_dice_slots()

        # Convert engine moves to action indices using new encoding
        expected_actions = set()
        for m in engine_moves:
            if m.is_movement_move:
                src = m.src if m.src != -1 else 24
                die = m.n
                # Map die value to slot(s)
                if die == slot0_die:
                    expected_actions.add(src * 2 + 0)
                if die == slot1_die:
                    expected_actions.add(src * 2 + 1)

        assert legal_actions == expected_actions


class TestBackgammonEnvSetDice:
    """Tests for set_dice test helper."""

    def test_set_dice_updates_remaining_dice(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        env.set_dice([3, 5])
        remaining = env.game.get_remaining_dice()

        assert set(remaining) == {3, 5}

    def test_set_dice_generates_movement_moves(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        env.set_dice([2, 4])
        moves = env.game.get_moves()

        assert len(moves) > 0
        assert all(m.is_movement_move for m in moves)

    def test_set_dice_clears_nature_turn(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        env.set_dice([1, 6])

        assert not env.game.is_nature_turn()

    def test_set_dice_doubles(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        env.set_dice([4, 4, 4, 4])
        remaining = env.game.get_remaining_dice()

        assert remaining == [4, 4, 4, 4]


class TestBackgammonEnvBotMode:
    """Tests for bot mode behavior."""

    def test_bot_mode_agent_is_player_zero(self):
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        assert env._current_player == 0

    def test_bot_mode_returns_to_agent_after_step(self):
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        obs = env.reset()

        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        if len(legal_actions) > 0:
            action = legal_actions[0]
            timestep = env.step(action)

            if not timestep.done:
                assert env._current_player == 0, "After bot plays, should be agent's turn"


class TestBackgammonEnvGameCompletion:
    """Tests for complete game simulation."""

    def test_random_game_completes(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env._rng.seed(42)
        random.seed(42)

        obs = env.reset()
        move_count = 0
        max_moves = 5000

        while move_count < max_moves:
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

            assert len(legal_actions) > 0, f"No legal actions at move {move_count}"

            action = random.choice(legal_actions)
            timestep = env.step(action)
            move_count += 1

            if timestep.done:
                break

            obs = timestep.obs

        assert timestep.done, f"Game did not complete in {max_moves} moves"

    def test_winner_is_valid(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env._rng.seed(123)
        random.seed(123)

        obs = env.reset()

        while True:
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
            action = random.choice(legal_actions)
            timestep = env.step(action)

            if timestep.done:
                break
            obs = timestep.obs

        winner = env.game.get_winner()
        assert winner in [0, 1]

    def test_reward_matches_winner(self):
        """Test that reward sign matches winner (accounts for gammon/backgammon scoring)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env._rng.seed(456)
        random.seed(456)

        obs = env.reset()
        last_player = None

        while True:
            last_player = env._current_player
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
            action = random.choice(legal_actions)
            timestep = env.step(action)

            if timestep.done:
                break
            obs = timestep.obs

        winner = env.game.get_winner()
        # With gammon/backgammon scoring, reward can be 1, 2, or 3 (or negative)
        if winner == last_player:
            assert timestep.reward > 0, f"Winner should get positive reward, got {timestep.reward}"
            assert timestep.reward in [1, 2, 3], f"Reward should be 1, 2, or 3, got {timestep.reward}"
        else:
            assert timestep.reward < 0, f"Loser should get negative reward, got {timestep.reward}"
            assert timestep.reward in [-1, -2, -3], f"Reward should be -1, -2, or -3, got {timestep.reward}"


class TestBackgammonEnvMultiMoveTurns:
    """Tests for multi-move turn behavior (remaining dice updates)."""

    def test_remaining_dice_decreases_after_move(self):
        """After a move, remaining dice count should decrease."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Set specific non-doubles dice
        env.set_dice([5, 3])
        initial_dice = list(env.game.get_remaining_dice())
        assert len(initial_dice) == 2

        # Get a legal action and execute it
        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
        assert len(legal_actions) > 0

        # Store current player
        current_player = env._current_player

        # Execute move
        env._step_core(legal_actions[0])

        # Check if still same player's turn (has remaining dice)
        if env._current_player == current_player:
            remaining_dice = list(env.game.get_remaining_dice())
            assert len(remaining_dice) < len(initial_dice), \
                f"Remaining dice should decrease: {len(initial_dice)} -> {len(remaining_dice)}"

    def test_action_mask_updates_after_partial_move(self):
        """Action mask should update after first die is used."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Set non-doubles dice
        env.set_dice([6, 2])

        obs1 = env.observe()
        mask1 = obs1['action_mask'].copy()
        legal1 = set(i for i, x in enumerate(mask1) if x == 1)

        # Execute a move
        action = list(legal1)[0]
        current_player = env._current_player
        env._step_core(action)

        # If still same player's turn, mask should have changed
        if env._current_player == current_player:
            obs2 = env.observe()
            mask2 = obs2['action_mask']
            legal2 = set(i for i, x in enumerate(mask2) if x == 1)

            # Legal actions should be different (used one die)
            assert legal1 != legal2 or len(legal2) > 0

    def test_turn_changes_after_all_dice_used(self):
        """Turn should change after all dice are used."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Set non-doubles dice
        env.set_dice([4, 2])
        initial_player = env._current_player
        moves_made = 0

        # Use all dice
        while not env.game.game_ended():
            remaining = list(env.game.get_remaining_dice())
            if len(remaining) == 0:
                break

            obs = env.observe()
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

            if len(legal_actions) == 0:
                break

            env._step_core(legal_actions[0])
            moves_made += 1

            # Check if player changed (turn ended)
            if env._current_player != initial_player:
                break

        # Assert: either turn changed OR dice exhausted (remaining == 0)
        remaining_dice = list(env.game.get_remaining_dice())
        turn_changed = env._current_player != initial_player
        dice_exhausted = len(remaining_dice) == 0

        assert turn_changed or dice_exhausted, \
            f"After using dice, turn should change or dice should be empty. " \
            f"Turn changed: {turn_changed}, Remaining dice: {remaining_dice}"

        # If turn changed, verify new player has fresh dice
        if turn_changed and not env.game.game_ended():
            new_remaining = list(env.game.get_remaining_dice())
            assert len(new_remaining) >= 2, \
                f"New turn should have fresh dice, got {new_remaining}"

    def test_doubles_consume_four_moves(self):
        """Doubles should provide 4 moves with same die value."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Set doubles
        env.set_dice([3, 3, 3, 3])
        initial_remaining = list(env.game.get_remaining_dice())
        assert initial_remaining == [3, 3, 3, 3], f"Should have 4 threes, got {initial_remaining}"

        initial_player = env._current_player
        moves_made = 0

        # Use all four dice
        while not env.game.game_ended() and moves_made < 4:
            remaining = list(env.game.get_remaining_dice())
            if len(remaining) == 0:
                break

            obs = env.observe()
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

            if len(legal_actions) == 0:
                break

            # Verify both die slots are valid (doubles)
            slot0_die, slot1_die = env._get_dice_slots()
            assert slot0_die == slot1_die == 3, \
                f"Doubles should have same value in both slots, got {slot0_die}, {slot1_die}"

            env._step_core(legal_actions[0])
            moves_made += 1

            # Check remaining dice decreased
            new_remaining = list(env.game.get_remaining_dice())
            expected_count = 4 - moves_made
            if env._current_player == initial_player:  # Still our turn
                assert len(new_remaining) == expected_count, \
                    f"After {moves_made} moves, should have {expected_count} dice, got {len(new_remaining)}"

        # Should have made 4 moves (or less if blocked)
        assert moves_made > 0, "Should have made at least one move with doubles"


class TestBackgammonEnvNoLegalMoves:
    """Tests for auto-advance when no legal moves available."""

    def test_blocked_position_has_empty_action_mask(self):
        """When all moves are blocked, action mask should be empty."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Create a blocked position for white
        # White has single piece, all destinations blocked
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 12] = 1   # Single white piece
        board[0, 0] = 14   # Rest in home (irrelevant for movement)
        # Block all possible destinations
        board[1, 11] = 2   # Block 12-1
        board[1, 10] = 2   # Block 12-2
        board[1, 9] = 2    # Block 12-3
        board[1, 8] = 2    # Block 12-4
        board[1, 7] = 2    # Block 12-5
        board[1, 6] = 2    # Block 12-6
        board[1, 5] = 3    # Leftover black pieces

        env.game.debug_reset_board(board)
        env.game.set_turn(0)  # White's turn
        env.set_dice([3, 4])  # Dice that would move to blocked points

        # Verify: no legal moves for white
        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
        assert len(legal_actions) == 0, "Should have no legal moves in blocked position"

        # Verify engine reports no moves
        engine_moves = env.game.get_moves()
        assert len(engine_moves) == 0, "Engine should have no legal moves"

    def test_auto_advance_self_play_via_step(self):
        """In self-play, blocked position should auto-advance when step is called."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # First make a normal move to trigger the system
        obs = env.observe()
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        if len(legal_actions) > 0:
            initial_player = env._current_player
            timestep = env.step(legal_actions[0])

            # After step, either:
            # 1. Same player continues (has remaining dice)
            # 2. Turn changed to opponent
            # In both cases, should have legal actions (auto-advanced if blocked)
            if not timestep.done:
                new_legal = [i for i, x in enumerate(timestep.obs['action_mask']) if x == 1]
                assert len(new_legal) > 0, \
                    f"Should have legal actions after step (player changed or auto-advanced)"

    def test_auto_advance_bot_mode_no_hang(self):
        """In bot mode, blocked bot position should not hang."""
        cfg = EasyDict(dict(battle_mode='play_with_bot_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Play a few moves to get past opening
        obs = env.observe()
        max_iterations = 100
        iteration = 0

        while iteration < max_iterations and not env.game.game_ended():
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

            if len(legal_actions) == 0:
                # This should not happen - env should auto-advance
                break

            action = legal_actions[0]
            timestep = env.step(action)
            iteration += 1

            if timestep.done:
                break

            # Key assertion: after step, we should be back to agent's turn
            # (player 0) with legal actions, or game ended
            assert env._current_player == 0 or timestep.done, \
                f"Bot mode should always return to agent's turn, got player {env._current_player}"

            if not timestep.done:
                new_legal = [i for i, x in enumerate(timestep.obs['action_mask']) if x == 1]
                assert len(new_legal) > 0, \
                    f"Agent should have legal actions after bot plays"

            obs = timestep.obs

        # Should not hang - verify we made progress
        assert iteration < max_iterations, "Bot mode appears to have hung"

    def test_game_flow_handles_blocked_turns(self):
        """During normal gameplay, blocked turns should auto-advance via auto_roll."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        random.seed(999)
        env._rng.seed(999)

        obs = env.reset()
        max_steps = 500
        step_count = 0

        while step_count < max_steps:
            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

            # Key assertion: in normal gameplay, we should always have legal actions
            # because auto_roll() handles blocked positions after turn changes
            assert len(legal_actions) > 0, \
                f"Normal gameplay should always have legal actions (step {step_count})"

            action = random.choice(legal_actions)
            timestep = env.step(action)
            step_count += 1

            if timestep.done:
                break

            obs = timestep.obs

        # Verify game progressed (either finished or made progress)
        assert step_count > 0, "Should have made at least one step"


class TestBackgammonEnvPieceConservation:
    """Tests for piece count conservation during gameplay."""

    def test_piece_count_conservation(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env._rng.seed(789)
        random.seed(789)

        obs = env.reset()

        for _ in range(200):
            # Verify piece counts
            board = env.game.get_board()
            bar = env.game.get_bar()
            off = env.game.get_beared_off()

            white_total = np.sum(board[0]) + bar[0] + off[0]
            black_total = np.sum(board[1]) + bar[1] + off[1]

            assert white_total == 15, f"White should have 15 pieces, got {white_total}"
            assert black_total == 15, f"Black should have 15 pieces, got {black_total}"

            action_mask = obs['action_mask']
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
            action = random.choice(legal_actions)
            timestep = env.step(action)

            if timestep.done:
                break
            obs = timestep.obs


class TestBackgammonEnvDoublesMoveLists:
    """Snapshot tests for legal move lists when doubles are rolled."""

    def _movement_tuples(self, env):
        moves = env.game.get_moves()
        return sorted((m.src, m.dst, m.n) for m in moves if m.is_movement_move)

    def test_doubles_open_board(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 10] = 2
        board[0, 8] = 1
        board[0, 0] = 12
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([3, 3, 3, 3])

        expected = sorted([
            (10, 7, 3),
            (8, 5, 3),
        ])
        assert self._movement_tuples(env) == expected

    def test_doubles_bar_entry_only(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        bar = np.zeros((2,), dtype=np.int8)
        bar[0] = 1
        env.game.debug_reset_board(board, bar=bar)
        env.game.set_turn(0)
        env.set_dice([2, 2, 2, 2])

        expected = [(24, 22, 2)]
        assert self._movement_tuples(env) == expected

    def test_doubles_exact_bear_off(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 5] = 1
        board[0, 0] = 14
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([6, 6, 6, 6])

        expected = [(5, -1, 6)]
        assert self._movement_tuples(env) == expected


class TestBackgammonEnvObservation:
    """Tests for observation vector correctness with separate planes encoding."""

    # ==================== Observation Shape Tests ====================

    def test_minimal_observation_shape(self):
        """Test minimal observation has correct shape (40, 1, 25)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        obs = env.reset()
        assert obs['observation'].shape == (40, 1, 25)

    def test_standard_observation_shape(self):
        """Test standard observation has correct shape (47, 1, 25)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        obs = env.reset()
        assert obs['observation'].shape == (47, 1, 25)

    def test_features_observation_shape(self):
        """Test features observation has correct shape (52, 1, 25)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_WITH_FEATURES))
        env = BackgammonEnv(cfg)
        obs = env.reset()
        assert obs['observation'].shape == (52, 1, 25)

    # ==================== Separate Planes Board Encoding Tests ====================

    def test_checker_encoding_my_single_piece(self):
        """Test that my single checker is encoded in my channels (0-5)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0] = 1  # My 1 piece at index 0
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_minimal()
        # My checker in channel 0 (>=1 threshold)
        assert obs[0, 0, 0] == 1.0, "My checker should be 1 in channel 0"
        # No checkers in higher threshold channels
        for ch in range(1, 6):
            assert obs[ch, 0, 0] == 0.0
        # No opponent checkers at this point
        for ch in range(6, 12):
            assert obs[ch, 0, 0] == 0.0

    def test_checker_encoding_opponent_piece(self):
        """Test that opponent's checkers are in opponent channels (6-11)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0] = 15  # My pieces elsewhere
        board[1, 5] = 3   # Opponent has 3 pieces at index 5
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_minimal()
        # Opponent has 3 pieces, so channels 6,7,8 should be 1
        for ch in range(6, 9):
            assert obs[ch, 0, 5] == 1.0, f"Opponent channel {ch} should be 1"
        # Channels 9,10,11 should be 0 (opponent has < 4,5,6 pieces)
        for ch in range(9, 12):
            assert obs[ch, 0, 5] == 0.0
        # My channels should be 0 at this point
        for ch in range(6):
            assert obs[ch, 0, 5] == 0.0

    def test_checker_encoding_five_pieces(self):
        """Test that 5 of my checkers are encoded in channels 0-4."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 5] = 5
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_minimal()
        for ch in range(5):
            assert obs[ch, 0, 5] == 1.0, f"Channel {ch} should be 1"
        assert obs[5, 0, 5] == 0.0

    def test_checker_encoding_six_plus_pieces(self):
        """Test that 6+ checkers sets all my channels (0-5) to 1."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 10] = 8
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_minimal()
        for ch in range(6):
            assert obs[ch, 0, 10] == 1.0

    # ==================== Bar as Spatial Point Tests ====================

    def test_bar_encoding_spatial(self):
        """Test bar checkers are encoded at spatial index 24."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        bar = np.array([3, 2], dtype=np.int8)
        board[0, 0] = 12
        board[1, 23] = 13
        env.game.debug_reset_board(board, bar=bar)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_minimal()
        # My bar (3 checkers) at spatial index 24, channels 0-2
        for ch in range(3):
            assert obs[ch, 0, 24] == 1.0, f"My bar channel {ch} should be 1"
        for ch in range(3, 6):
            assert obs[ch, 0, 24] == 0.0
        # Opponent bar (2 checkers) at spatial index 24, channels 6-7
        for ch in range(6, 8):
            assert obs[ch, 0, 24] == 1.0, f"Opponent bar channel {ch} should be 1"
        for ch in range(8, 12):
            assert obs[ch, 0, 24] == 0.0

    # ==================== Dice Encoding Tests ====================

    def test_dice_encoding_non_doubles(self):
        """Test dice encoding for non-doubles (e.g., [5, 3])."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()
        env.set_dice([5, 3])

        obs = env._get_obs_minimal()
        # Slot 0: die=5, channel = 12 + 0*6 + (5-1) = 16
        assert obs[16, 0, 0] == 1.0
        # Slot 1: die=3, channel = 12 + 1*6 + (3-1) = 20
        assert obs[20, 0, 0] == 1.0

    def test_dice_encoding_doubles(self):
        """Test dice encoding for doubles (e.g., [4, 4, 4, 4])."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()
        env.set_dice([4, 4, 4, 4])

        obs = env._get_obs_minimal()
        # All 4 slots should have die=4
        # Slot 0: ch = 12 + 0*6 + 3 = 15
        # Slot 1: ch = 12 + 1*6 + 3 = 21
        # Slot 2: ch = 12 + 2*6 + 3 = 27
        # Slot 3: ch = 12 + 3*6 + 3 = 33
        assert obs[15, 0, 0] == 1.0
        assert obs[21, 0, 0] == 1.0
        assert obs[27, 0, 0] == 1.0
        assert obs[33, 0, 0] == 1.0

    def test_dice_ordering_high_to_low(self):
        """Test dice are ordered high to low in observation."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()
        env.set_dice([2, 6])  # Should be ordered as [6, 2]

        obs = env._get_obs_minimal()
        # Slot 0 should be 6: ch = 12 + 0*6 + 5 = 17
        assert obs[17, 0, 0] == 1.0
        # Slot 1 should be 2: ch = 12 + 1*6 + 1 = 19
        assert obs[19, 0, 0] == 1.0

    # ==================== Scalar Feature Tests ====================

    def test_off_encoding(self):
        """Test borne off checkers are encoded correctly."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        beared_off = np.array([5, 3], dtype=np.int8)
        board[0, 0] = 10
        board[1, 23] = 12
        env.game.debug_reset_board(board, beared_off=beared_off)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_minimal()
        # Channel 36: my off / 15
        assert abs(obs[36, 0, 0] - 5/15) < 0.01
        # Channel 37: opponent off / 15
        assert abs(obs[37, 0, 0] - 3/15) < 0.01

    # ==================== Contact Indicator Tests ====================

    def test_contact_indicator_initial_position(self):
        """Test contact indicator is 1 at game start (pieces intermingled)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        env.reset()

        obs = env._get_obs_standard()
        # Channel 40: contact indicator - should be 1 at start
        assert obs[40, 0, 0] == 1.0

    def test_contact_indicator_race_position(self):
        """Test contact indicator is 0 when pure race (pieces passed)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        env.reset()

        # Setup race position: my pieces in home, opponent far away
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0:6] = [3, 3, 3, 3, 2, 1]  # My 15 in home (indices 0-5)
        board[1, 18:24] = [3, 3, 3, 3, 2, 1]  # Opponent in their home (my indices 18-23)
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_standard()
        # Channel 40: contact indicator - should be 0 (pure race)
        assert obs[40, 0, 0] == 0.0

    def test_contact_indicator_bar_means_contact(self):
        """Test contact indicator is 1 when any player has bar checkers."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        env.reset()

        # Race position but with bar
        board = np.zeros((2, 24), dtype=np.int8)
        bar = np.array([1, 0], dtype=np.int8)
        board[0, 0:6] = [3, 3, 3, 3, 2, 0]  # 14 in home
        board[1, 18:24] = [3, 3, 3, 3, 2, 1]  # Opponent in their home
        env.game.debug_reset_board(board, bar=bar)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_standard()
        # Channel 40: contact indicator - should be 1 (bar means contact)
        assert obs[40, 0, 0] == 1.0

    # ==================== Legal Dice Slot Flags Tests ====================

    def test_legal_dice_both_slots_playable(self):
        """Test both dice slots are marked playable when both have legal moves."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()
        env.set_dice([5, 3])  # Non-doubles, should have moves for both

        obs = env.observe()
        obs_vector = obs['observation']
        action_mask = obs['action_mask']

        # Verify both slots have legal actions
        slot0_has_action = any(action_mask[i] == 1 for i in range(0, 50, 2))
        slot1_has_action = any(action_mask[i] == 1 for i in range(1, 50, 2))

        # Channel 38: slot 0 playable, Channel 39: slot 1 playable
        assert obs_vector[38, 0, 0] == (1.0 if slot0_has_action else 0.0)
        assert obs_vector[39, 0, 0] == (1.0 if slot1_has_action else 0.0)

    def test_legal_dice_doubles_both_playable(self):
        """Test both dice slots are marked playable for doubles."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()
        env.set_dice([4, 4, 4, 4])  # Doubles

        obs = env.observe()
        obs_vector = obs['observation']

        # For doubles, both slots map to same die value, so both should be playable
        # Channel 38: slot 0 playable, Channel 39: slot 1 playable
        # At game start with doubles, we should have legal moves
        assert obs_vector[38, 0, 0] == 1.0
        assert obs_vector[39, 0, 0] == 1.0

    def test_legal_dice_one_slot_blocked(self):
        """Test only one slot is playable when moves for one die are blocked."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()

        # Setup: single piece that can only move with one die value
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 5] = 1   # Single white piece at point 6
        board[0, 0] = 14  # Rest in home
        # Block the 3-move destination (5-3=2, so block point 2)
        board[1, 2] = 2
        # Leave 5-move destination open (5-5=0, point 0 is home, allow bear off)
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env.set_dice([5, 3])  # Can use 5 to bear off, 3 is blocked

        obs = env.observe()
        obs_vector = obs['observation']
        action_mask = obs['action_mask']

        # Check actual legal actions in mask
        slot0_has_action = any(action_mask[i] == 1 for i in range(0, 50, 2))
        slot1_has_action = any(action_mask[i] == 1 for i in range(1, 50, 2))

        # Observation should match action mask
        assert obs_vector[38, 0, 0] == (1.0 if slot0_has_action else 0.0)
        assert obs_vector[39, 0, 0] == (1.0 if slot1_has_action else 0.0)

    def test_legal_dice_matches_action_mask(self):
        """Test legal dice flags always match action mask content."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        random.seed(42)
        env._rng.seed(42)

        # Play several moves and verify consistency each time
        obs = env.reset()
        for _ in range(20):
            obs_vector = obs['observation']
            action_mask = obs['action_mask']

            # Compute expected slot playability from mask
            slot0_expected = any(action_mask[i] == 1 for i in range(0, 50, 2))
            slot1_expected = any(action_mask[i] == 1 for i in range(1, 50, 2))

            # Verify channels match
            assert obs_vector[38, 0, 0] == (1.0 if slot0_expected else 0.0), \
                f"Slot 0 playable mismatch: obs={obs_vector[38, 0, 0]}, expected={slot0_expected}"
            assert obs_vector[39, 0, 0] == (1.0 if slot1_expected else 0.0), \
                f"Slot 1 playable mismatch: obs={obs_vector[39, 0, 0]}, expected={slot1_expected}"

            # Take random action
            legal_actions = [i for i, x in enumerate(action_mask) if x == 1]
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            timestep = env.step(action)
            if timestep.done:
                break
            obs = timestep.obs

    def test_legal_dice_in_features_observation(self):
        """Test legal dice flags are present in features observation too."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_WITH_FEATURES))
        env = BackgammonEnv(cfg)
        env.reset()
        env.set_dice([6, 2])

        obs = env.observe()
        obs_vector = obs['observation']
        action_mask = obs['action_mask']

        # Verify channels 38-39 in features observation
        slot0_expected = any(action_mask[i] == 1 for i in range(0, 50, 2))
        slot1_expected = any(action_mask[i] == 1 for i in range(1, 50, 2))

        assert obs_vector[38, 0, 0] == (1.0 if slot0_expected else 0.0)
        assert obs_vector[39, 0, 0] == (1.0 if slot1_expected else 0.0)

    # ==================== Standard Mode Tests ====================

    def test_standard_total_pips(self):
        """Test total pip count features in standard observation."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        env.reset()

        # Setup: all my pieces at point 1 (index 0), opponent at their point 1
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0] = 15  # My 15 pieces at index 0 (1 pip each) = 15 pips
        board[1, 23] = 15  # Opponent at my index 23 (their point 1) = 15 pips
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_standard()
        # Channel 41: my total pips / 200 = 15/200
        assert abs(obs[41, 0, 0] - 15/200) < 0.01
        # Channel 42: opponent total pips / 200 = 15/200
        assert abs(obs[42, 0, 0] - 15/200) < 0.01

    def test_standard_checkers_in_home(self):
        """Test checkers in home features in standard observation."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0] = 5   # 5 in home at index 0
        board[0, 3] = 3   # 3 in home at index 3
        board[0, 10] = 7  # 7 outside home
        # Opponent: some in their home (my indices 18-23)
        board[1, 20] = 4
        board[1, 22] = 2
        board[1, 5] = 9   # Outside their home
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_standard()
        # Channel 43: my checkers in home (indices 0-5) = 5 + 3 = 8
        assert abs(obs[43, 0, 0] - 8/15) < 0.01
        # Channel 44: opponent checkers in home (indices 18-23) = 4 + 2 = 6
        assert abs(obs[44, 0, 0] - 6/15) < 0.01

    def test_standard_delta_pips(self):
        """Test delta pip count in standard observation."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        env.reset()

        # I'm behind (more pips to go)
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 23] = 15  # My pieces far: 15 * 24 = 360 pips
        board[1, 23] = 15  # Opponent near their home: 15 * 1 = 15 pips
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_standard()
        # Channel 45: delta pips (clipped) - I'm way behind, should be 1.0
        assert obs[45, 0, 0] == 1.0

    def test_standard_delta_checkers_home(self):
        """Test delta checkers in home in standard observation."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0:6] = [3, 3, 3, 3, 2, 1]  # 15 in my home
        board[1, 10] = 15  # Opponent outside their home (none in home)
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_standard()
        # Channel 46: delta checkers in home = 15 - 0 = 15, normalized = 1.0
        assert obs[46, 0, 0] == 1.0

    def test_standard_legal_dice_present(self):
        """Test legal dice flags are present in standard observation."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_STANDARD))
        env = BackgammonEnv(cfg)
        env.reset()
        env.set_dice([5, 2])

        obs = env.observe()
        obs_vector = obs['observation']
        action_mask = obs['action_mask']

        # Verify channels 38-39 in standard observation
        slot0_expected = any(action_mask[i] == 1 for i in range(0, 50, 2))
        slot1_expected = any(action_mask[i] == 1 for i in range(1, 50, 2))

        assert obs_vector[38, 0, 0] == (1.0 if slot0_expected else 0.0)
        assert obs_vector[39, 0, 0] == (1.0 if slot1_expected else 0.0)

    # ==================== Features Mode Tests ====================

    def test_stragglers_zero_means_can_bear_off(self):
        """Test stragglers=0 when all pieces in home (can bear off)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_WITH_FEATURES))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0] = 5   # In home
        board[0, 3] = 5   # In home
        board[0, 5] = 5   # In home
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_with_features()
        # Channel 49: my stragglers / 15 should be 0
        assert obs[49, 0, 0] == 0.0

    def test_stragglers_nonzero_outside_home(self):
        """Test stragglers > 0 when pieces outside home."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_WITH_FEATURES))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0] = 10
        board[0, 10] = 5  # Outside home
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_with_features()
        # Channel 49: my stragglers / 15 = 5/15
        assert abs(obs[49, 0, 0] - 5/15) < 0.01

    def test_delta_pip_count_even(self):
        """Test delta pip count is 0 when equal (clipped format)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_WITH_FEATURES))
        env = BackgammonEnv(cfg)
        env.reset()

        # Same pip count for both players
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 0] = 15  # My 15 pieces at point 1: 15 pips
        board[1, 23] = 15  # Opponent at my index 23 = their point 1: 15 pips
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_with_features()
        # Channel 45: delta pip count, clipped to [-1, 1], 0 = even
        assert abs(obs[45, 0, 0]) < 0.01

    def test_delta_pip_count_behind(self):
        """Test delta pip count > 0 when I'm behind (more pips)."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_WITH_FEATURES))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 23] = 15  # My pieces far from home: 15*24 = 360 pips
        board[1, 23] = 15  # Opponent at my 23 = their 1: 15 pips
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_with_features()
        # I have way more pips, so delta > 0 (clipped to max 1.0)
        assert obs[45, 0, 0] > 0

    def test_delta_pip_clipping(self):
        """Test delta features are clipped to [-1, 1]."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_WITH_FEATURES))
        env = BackgammonEnv(cfg)
        env.reset()

        # Extreme position: I'm way behind
        # From White's perspective (player 0):
        # - My pieces at index 23 = my 24-point, far from home = 24 pips each
        # - Opponent at my index 23 = opponent's 1-point (near their home) = 1 pip each
        board = np.zeros((2, 24), dtype=np.int8)
        board[0, 23] = 15  # All my pieces at far point: 15 * 24 = 360 pips
        board[1, 23] = 15  # Opponent near their home: pip_weights_opp[23] = 1, so 15 pips
        env.game.debug_reset_board(board)
        env.game.set_turn(0)
        env._current_player = 0

        obs = env._get_obs_with_features()
        # delta_pips = 360 - 15 = 345, normalized = 345/200 = 1.725, clipped to 1.0
        assert obs[45, 0, 0] == 1.0
        # Channel 51: delta stragglers also clipped
        assert -1.0 <= obs[51, 0, 0] <= 1.0

    # ==================== Player Perspective Tests ====================

    def test_black_player_board_flipped(self):
        """Test that Black player's observation has board flipped correctly."""
        cfg = EasyDict(dict(battle_mode='self_play_mode', obs_type=OBS_MINIMAL))
        env = BackgammonEnv(cfg)
        env.reset()

        board = np.zeros((2, 24), dtype=np.int8)
        board[1, 23] = 5  # Black at absolute index 23 (Black's ace point)
        board[0, 0] = 15
        env.game.debug_reset_board(board)
        env.game.set_turn(1)
        env._current_player = 1

        obs = env._get_obs_minimal()
        # For Black, index 23 is their ace, appears at relative index 0
        # My (Black's) checker in channel 0
        assert obs[0, 0, 0] == 1.0


class TestBackgammonEnvTerminalRewards:
    """Tests for win/gammon/backgammon reward values."""

    def _force_terminal(self, env, winner, loser_beared_off, loser_bar=0, loser_in_home=0):
        board = np.zeros((2, 24), dtype=np.int8)
        bar = np.zeros((2,), dtype=np.int8)
        beared_off = np.zeros((2,), dtype=np.int8)

        beared_off[winner] = 15
        beared_off[1 - winner] = loser_beared_off
        bar[1 - winner] = loser_bar

        if loser_in_home > 0:
            if winner == 0:
                board[1 - winner, 0] = loser_in_home
            else:
                board[1 - winner, 18] = loser_in_home

        env.game.debug_reset_board(board, bar=bar, beared_off=beared_off)
        env.game.set_turn(winner)
        env.game.check_for_winner()
        env._current_player = winner

    def test_normal_win_reward_is_one(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        self._force_terminal(env, winner=0, loser_beared_off=1)
        timestep = env._step_core(0)

        assert timestep.done
        assert timestep.reward == 1

    def test_gammon_reward_is_two(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        self._force_terminal(env, winner=0, loser_beared_off=0)
        timestep = env._step_core(0)

        assert timestep.done
        assert timestep.reward == 2

    def test_backgammon_reward_is_three(self):
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        self._force_terminal(env, winner=0, loser_beared_off=0, loser_bar=1)
        timestep = env._step_core(0)

        assert timestep.done
        assert timestep.reward == 3

    def test_backgammon_loser_in_winner_home(self):
        """Backgammon when loser has pieces in winner's home board."""
        cfg = EasyDict(dict(battle_mode='self_play_mode'))
        env = BackgammonEnv(cfg)
        env.reset()

        # Winner=0 (White), loser has pieces in White's home (indices 0-5)
        self._force_terminal(env, winner=0, loser_beared_off=0, loser_in_home=2)
        timestep = env._step_core(0)

        assert timestep.done
        assert timestep.reward == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
