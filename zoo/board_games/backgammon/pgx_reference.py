import numpy as np

# Reference: pgx/backgammon.py::_make_init_board_short
# https://github.com/sile16/pgx
PGX_SHORT_GAME_BOARD = np.array(
    [
        0, -1, -3, 0, 2, -3, 0, -3, -2, 0, 0, -1,
        1, 0, 0, 2, 3, 0, 3, -2, 0, 3, 1, 0,
        0, 0, 0, 0,
    ],
    dtype=np.int32,
)


def pgx_board_to_internal(pgx_board: np.ndarray, current_player: int = 0):
    """
    Convert a pgx backgammon board (signed, current-player perspective) to
    LightZero backgammon internal representation.

    pgx board layout:
      - indices 0-23: points (current player's 24..1)
      - 24: current player's bar, 25: opponent's bar
      - 26: current player's off, 27: opponent's off
      - positive counts: current player, negative: opponent
    """
    if pgx_board.shape != (28,):
        raise ValueError("pgx_board must have shape (28,)")
    if current_player not in (0, 1):
        raise ValueError("current_player must be 0 (white) or 1 (black)")

    board = np.zeros((2, 24), dtype=np.int8)
    bar = np.zeros((2,), dtype=np.int8)
    beared_off = np.zeros((2,), dtype=np.int8)

    def map_index(i: int) -> int:
        # pgx index 0 is current player's 24-point.
        # White uses reversed indexing vs pgx; Black aligns with pgx.
        return 23 - i if current_player == 0 else i

    for i in range(24):
        val = int(pgx_board[i])
        if val > 0:
            board[current_player, map_index(i)] = val
        elif val < 0:
            board[1 - current_player, map_index(i)] = -val

    bar[current_player] = int(pgx_board[24])
    bar[1 - current_player] = int(abs(pgx_board[25]))
    beared_off[current_player] = int(pgx_board[26])
    beared_off[1 - current_player] = int(abs(pgx_board[27]))

    return board, bar, beared_off


def pgx_short_game_to_internal(current_player: int = 0):
    return pgx_board_to_internal(PGX_SHORT_GAME_BOARD, current_player=current_player)
