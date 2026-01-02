# cython: profile=False
import array
import copy
import numpy as np
import numpy.random as np_random
cimport numpy as np
cimport numpy.random as np_random
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX
from cpython cimport array

cdef int WHITE = 0
cdef int BLACK = 1
cdef int NONE = 2

np.import_array()

cdef class Move:
    cdef public bint is_movement_move
    cdef public signed char src, dst, n, i, j
    
    def __hash__(self):
        if self.is_movement_move:
            return hash((self.src, self.dst, self.n))
        else:
            return hash((self.i, self.j))
            
    def __repr__(self):
        if self.is_movement_move:
            return f"Move(src={self.src}, dst={self.dst}, n={self.n})"
        else:
            return f"Dice(i={self.i}, j={self.j})"

cdef struct UndoInfo:
    signed char hit
    signed char moved_from_bar
    signed char moved_to_off
    Py_ssize_t removed_die_index

cdef class State:
    cdef signed char [:, ::1] board
    cdef signed char [::1] bar, beared_off
    cdef int turn, turn_number, winner
    cdef bint _is_nature_turn, game_has_started
    cdef list legal_moves, _piece_moves_left
    cdef array.array _array_template
    
    def __init__(self):
        self.board = np.zeros((2, 24), dtype=np.int8)
        self.bar = np.zeros((2,), dtype=np.int8)
        self.beared_off = np.zeros((2,), dtype=np.int8)
        self.turn = NONE
        self.turn_number = 1
        self.winner = NONE
        self._is_nature_turn = True
        self.game_has_started = False
        self.legal_moves = []
        self._piece_moves_left = []
        self._array_template = array.array('i', [])
        self.reset()

    cpdef void reset(self):
        self._reset()
        self.generate_pre_game_2d6_moves()
        
    cpdef void debug_reset_board(self, signed char [:, ::1] board, signed char [::1] bar=np.zeros((2,), dtype=np.int8), signed char [::1] beared_off=np.zeros((2,), dtype=np.int8)):
        self._reset()
        self.board = np.copy(board)
        self.bar = np.copy(bar)
        self.beared_off = np.copy(beared_off)
        self.generate_pre_game_2d6_moves()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    def __copy__(self):
        cdef State state
        cdef Move move 
        cdef signed char x
        cdef int i, j
        state = State()
        # Fast manual copy of arrays
        state.board = np.copy(self.board)
        state.bar = np.copy(self.bar)
        state.beared_off = np.copy(self.beared_off)
        
        state.turn = self.turn
        state.turn_number = self.turn_number
        state.winner = self.winner
        state._is_nature_turn = self._is_nature_turn
        state.game_has_started = self.game_has_started
        
        # Deep copy moves lists
        if self.legal_moves is not None:
             if len(self.legal_moves) > 0 and isinstance(self.legal_moves[0], Move):
                 if (<Move>self.legal_moves[0]).is_movement_move:
                     state.legal_moves = [self._new_movement_move(move.src, move.dst, move.n) for move in self.legal_moves]
                 else:
                     state.legal_moves = self.get_roll_2d6_moves() # Dice moves are generic
             else:
                 state.legal_moves = []
        
        state._piece_moves_left = [x for x in self._piece_moves_left]
        return state

    cdef Move _new_movement_move(self, signed char src, signed char dst, signed char n):
        cdef Move move = Move()
        move.src = src
        move.dst = dst
        move.n = n
        move.is_movement_move = True
        return move
    
    cdef void _reset(self):
        self.board = np.zeros((2, 24), dtype=np.int8)
        self.bar = np.zeros((2,), dtype=np.int8)
        self.beared_off = np.zeros((2,), dtype=np.int8)
        self.turn = NONE
        self.turn_number = 1
        self._is_nature_turn = True
        self.game_has_started = False
        self._piece_moves_left = []
        self.winner = NONE
        #Place white pieces
        self.board[WHITE, 23] = 2
        self.board[WHITE, 12] = 5
        self.board[WHITE, 7] = 3
        self.board[WHITE, 5] = 5
        #Place black pieces
        self.board[BLACK, 0] = 2
        self.board[BLACK, 11] = 5
        self.board[BLACK, 16] = 3
        self.board[BLACK, 18] = 5

    def get_board(self):
        return np.asarray(self.board)
        
    cpdef list get_remaining_dice(self):
        return [x for x in self._piece_moves_left]

    cpdef void set_dice(self, list dice):
        self._piece_moves_left = [x for x in dice]

    def get_bar(self):
        return np.asarray(self.bar)

    def get_beared_off(self):
        return np.asarray(self.beared_off)

    cpdef int get_player_turn(self):
        return self.turn

    cpdef list get_moves(self):
        return self.legal_moves

    cpdef int get_winner(self):
        return self.winner

    cpdef bint is_nature_turn(self):
        return self._is_nature_turn

    cpdef bint has_game_started(self):
        return self.game_has_started

    cpdef void set_game_started(self, bint started):
        self.game_has_started = started

    cpdef void set_nature_turn(self, bint is_nature_turn):
        self._is_nature_turn = is_nature_turn

    cpdef void set_turn(self, int player):
        self.turn = player

    cpdef void force_start(self, int start_player=0):
        """
        Force game to start with specified player, bypassing pre-game roll-off.

        This method:
        - Sets the starting player (default: player 0)
        - Marks game as started
        - Rolls dice (doubles allowed)
        - Advances to movement state ready for play

        Use this for training/evaluation where you want deterministic start.
        """
        self.turn = start_player
        self.game_has_started = True

        # Roll dice (any combination including doubles)
        cdef int d1 = (rand() % 6) + 1
        cdef int d2 = (rand() % 6) + 1

        if d1 == d2:
            # Doubles: 4 dice
            self._piece_moves_left = [d1, d1, d1, d1]
        else:
            self._piece_moves_left = [d1, d2]

        self._is_nature_turn = False
        self.generate_movement_moves()

    cpdef void auto_roll(self):
        """
        If in a nature turn, randomly roll dice and advance to movement state.

        Call this after a turn ends to automatically roll for the next player.
        Loops until there are legal movement moves or game ends.
        """
        cdef int d1, d2, iterations
        iterations = 0

        while self._is_nature_turn and not self.game_ended() and iterations < 100:
            iterations += 1

            # Roll dice randomly
            d1 = (rand() % 6) + 1
            d2 = (rand() % 6) + 1

            if d1 == d2:
                self._piece_moves_left = [d1, d1, d1, d1]
            else:
                self._piece_moves_left = [d1, d2]

            self._is_nature_turn = False
            self.generate_movement_moves()

            # If no moves available, turn auto-passes
            if len(self.legal_moves) == 0 and not self.game_ended():
                self._goto_next_turn()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void _do_move_fast(self, Move move, UndoInfo* undo):
        """
        Fast in-place move application for lookahead.
        Assumes move is a movement move and does not update turn or legal_moves.
        """
        undo.hit = 0
        undo.moved_from_bar = 1 if move.src in [-1, 24] else 0
        undo.moved_to_off = 1 if move.dst in [-1, 24] else 0

        if undo.moved_from_bar:
            self.bar[self.turn] -= 1
        else:
            self.board[self.turn, move.src] -= 1

        if undo.moved_to_off:
            self.beared_off[self.turn] += 1
        else:
            if self.board[self._other_player(), move.dst] == 1:
                undo.hit = 1
                self.bar[self._other_player()] += 1
                self.board[self._other_player(), move.dst] = 0
            self.board[self.turn, move.dst] += 1

        if move.n in self._piece_moves_left:
            undo.removed_die_index = self._piece_moves_left.index(move.n)
            del self._piece_moves_left[undo.removed_die_index]
        else:
            undo.removed_die_index = -1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef void _undo_move_fast(self, Move move, UndoInfo* undo):
        if undo.removed_die_index != -1:
            self._piece_moves_left.insert(undo.removed_die_index, move.n)

        if undo.moved_to_off:
            self.beared_off[self.turn] -= 1
        else:
            self.board[self.turn, move.dst] -= 1
            if undo.hit:
                self.board[self._other_player(), move.dst] = 1
                self.bar[self._other_player()] -= 1

        if undo.moved_from_bar:
            self.bar[self.turn] += 1
        else:
            self.board[self.turn, move.src] += 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef int play_game_to_end(self):
        cdef int i, n_moves
        cdef list moves
        while not self.game_ended():
            moves = self.get_moves()
            n_moves = len(moves)
            i = rand() % n_moves
            self.do_move(moves[i])
        return self.get_winner()

    @cython.wraparound(False)
    @cython.boundscheck(False)    
    @cython.nonecheck(False)
    cpdef int play_game_to_depth(self, int depth):
        cdef int i, ply, n_moves
        cdef list moves
        ply = 0
        while not self.game_ended() and ply < depth:
            moves = self.get_moves()
            n_moves = len(moves)
            i = rand() % n_moves
            self.do_move(moves[i])
            ply += 1
        return self.get_winner()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef bint _has_piece(self, int point) nogil:
        """Returns True if point has a piece belonging to the player whose turn it is"""
        return self.board[self.turn, point] > 0
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef int _forward(self, int point, int n) nogil:
        cdef int new_point
        """Gets point n steps forward from point from the perspective of the player whose turn it is. Returns -1 if no such point exists or is unmovable"""
        new_point = point - n if self.turn == WHITE else point + n
        new_point = new_point if new_point >= 0 and new_point < 24 else -1
        if new_point != -1:
            new_point = new_point if self.board[self._other_player(), new_point] <= 1 else -1
        return new_point

    cpdef int bar_point(self):
        return -1 if self.turn == BLACK else 24

    cpdef int bearing_off_point(self):
        return -1 if self.turn == WHITE else 24

    cdef int _bar_point(self) nogil:
        return -1 if self.turn == BLACK else 24

    cdef int _bearing_off_point(self) nogil:
        return -1 if self.turn == WHITE else 24

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef array.array _get_points_with_pieces(self):
        cdef array.array points = array.clone(self._array_template, 0, False)
        cdef int i, point
        i = 0
        for point in range(24):
            if self._has_piece(point):
                array.resize_smart(points, i + 1)
                points[i] = point
                i += 1
        return points

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef list _get_overshoot_bear_off_moves(self, int die_value):
        """
        Generate overshoot bear-off moves when die is higher than highest occupied point.

        Backgammon rule: If a player rolls a number higher than the highest point
        on which they have a checker, they may bear off from the highest occupied point.

        For WHITE: home is indices 0-5, moves toward -1
          - Die n corresponds to exact point n-1
          - If no piece at n-1 AND no pieces at indices >= n (up to 5),
            can bear off from highest occupied point (largest index in 0 to n-2)

        For BLACK: home is indices 18-23, moves toward 24
          - Die n corresponds to exact point 24-n
          - If no piece at 24-n AND no pieces at indices < 24-n (down to 18),
            can bear off from highest occupied point (smallest index in 24-n+1 to 23)
        """
        cdef list moves = []
        cdef int exact_point, check_start, check_end, search_start, search_end
        cdef int idx, highest_occupied
        cdef bint has_higher_piece

        if self.turn == WHITE:
            # WHITE: home is 0-5, exact point for die n is n-1
            exact_point = die_value - 1

            # Check if any pieces exist at indices >= die_value (higher than exact point)
            # These are indices die_value, die_value+1, ..., 5
            has_higher_piece = False
            for idx in range(die_value, 6):  # die_value to 5
                if self._has_piece(idx):
                    has_higher_piece = True
                    break

            if not has_higher_piece:
                # No pieces at or above exact point, find highest occupied in 0 to die_value-2
                highest_occupied = -1
                for idx in range(die_value - 2, -1, -1):  # die_value-2 down to 0
                    if self._has_piece(idx):
                        highest_occupied = idx
                        break

                if highest_occupied >= 0:
                    moves.append(self._new_movement_move(highest_occupied, self._bearing_off_point(), die_value))
        else:
            # BLACK: home is 18-23, exact point for die n is 24-n
            exact_point = 24 - die_value

            # Check if any pieces exist at indices < exact_point (lower than exact point in home)
            # These are indices 18, 19, ..., exact_point-1
            has_higher_piece = False
            for idx in range(18, exact_point):  # 18 to exact_point-1
                if self._has_piece(idx):
                    has_higher_piece = True
                    break

            if not has_higher_piece:
                # No pieces below exact point, find lowest occupied in exact_point+1 to 23
                highest_occupied = -1
                for idx in range(exact_point + 1, 24):  # exact_point+1 to 23
                    if self._has_piece(idx):
                        highest_occupied = idx
                        break

                if highest_occupied >= 0:
                    moves.append(self._new_movement_move(highest_occupied, self._bearing_off_point(), die_value))

        return moves

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef list _get_raw_movement_moves(self):
        """Returns a list of all possible immediate moves based on current dice"""
        cdef list legal_moves = []
        cdef int i, src, dest, n
        cdef array.array points_with_pieces = self._get_points_with_pieces()
        
        # We iterate over unique dice values available.
        # If we have [4, 4], set is {4}, which is correct for finding ONE move of 4.
        for n in set(self._piece_moves_left):
            if self.bar[self.turn] > 0:
                dest = self._forward(self._bar_point(), n)
                if dest > -1:
                    legal_moves.append(self._new_movement_move(self._bar_point(), dest, n))
            else:
                if self._can_bear_off():
                    # Exact bear off: die n bears off from exact point
                    target_idx = n - 1 if self.turn == WHITE else 24 - n
                    if 0 <= target_idx < 24 and self._has_piece(target_idx):
                         legal_moves.append(self._new_movement_move(target_idx, self._bearing_off_point(), n))
                    else:
                        # Overshoot bear-off: if die is higher than highest occupied point,
                        # can bear off from highest occupied point in home.
                        # Rule: "If roll is higher than any point with checkers, bear off from highest."
                        legal_moves.extend(self._get_overshoot_bear_off_moves(n))

                # Standard moves (moving within the board)
                for i in range(len(points_with_pieces)):
                    src = points_with_pieces[i]
                    dest = self._forward(src, n)
                    if dest > -1:
                        legal_moves.append(self._new_movement_move(src, dest, n))

        return legal_moves

    cpdef list get_movement_moves(self):
        """
        Refined move generation that implements:
        1. Maximize number of dice played.
        2. If equal moves, maximize die value (Play higher die).
        """
        # 1. Get immediate candidates
        cdef list candidates = self._get_raw_movement_moves()
        if len(candidates) == 0:
            return []

        # Optimization: If only 1 move left to play (last die), no recursion needed.
        if len(self._piece_moves_left) == 1:
            return candidates

        # Optimization: For doubles (all dice have same value), skip lookahead/recursion.
        # When all dice are identical:
        # - No "play higher die" rule applies (all dice have equal value)
        # - Depth achieved is independent of move order (all moves consume same die value)
        # - Recursive lookahead provides no additional filtering benefit
        if len(set(self._piece_moves_left)) == 1:
            return candidates

        # 2. Lookahead
        # We need to find the max depth achievable from each candidate.
        cdef int max_depth = -1
        cdef list scored_moves = [] # tuples of (move, depth, first_die_val)
        cdef int depth
        cdef Move move
        cdef UndoInfo undo
        
        for move in candidates:
            # We must verify if this move is just the START of a valid sequence.
            self._do_move_fast(move, &undo)
            depth = 1 + self._get_max_depth_recursive()
            self._undo_move_fast(move, &undo)
            
            scored_moves.append((move, depth, move.n))
            
            if depth > max_depth:
                max_depth = depth
                
        # 3. Filter
        cdef list final_moves = []
        for move, depth, val in scored_moves:
            if depth == max_depth:
                final_moves.append(move)
                
        # 4. Apply "Higher Die" rule
        # If max_depth < total_available_dice AND we had a choice of dice (and dice were different)
        # AND we are in a situation where we can only play one of them.
        # "If you can play one number but not both, you must play the higher one."
        # This implies max_depth is 1 (current move + 0 subsequent).
        # And we started with >= 2 dice.
        
        if max_depth == 1 and len(self._piece_moves_left) >= 2:
             # Check if we have different dice values in final_moves
             # Filter to keep only those with max 'n'
             max_val = -1
             for m in final_moves:
                 if (<Move>m).n > max_val:
                     max_val = (<Move>m).n
             
             # Filter in place
             final_moves = [m for m in final_moves if (<Move>m).n == max_val]
             
        return final_moves

    cdef int _get_max_depth_recursive(self):
        """
        DFS to find maximum number of moves playable from current state.
        """
        if len(self._piece_moves_left) == 0:
            return 0
            
        cdef list moves = self._get_raw_movement_moves()
        if len(moves) == 0:
            return 0
            
        cdef int max_d = 0
        cdef int d
        cdef Move m
        cdef UndoInfo undo
        
        for m in moves:
            self._do_move_fast(m, &undo)
            d = 1 + self._get_max_depth_recursive()
            self._undo_move_fast(m, &undo)
            if d > max_d:
                max_d = d
                
        return max_d

    cdef void _generate_piece_moves_from_dice(self, Move dice):
        self._piece_moves_left.clear()
        if dice.i == dice.j:
            self._piece_moves_left = [dice.i] * 4
        else:
            self._piece_moves_left = [dice.i, dice.j]

    cpdef void generate_movement_moves(self):
        self.legal_moves = self.get_movement_moves()

    cpdef int other_player(self):
        return WHITE if self.turn == BLACK else BLACK if self.turn == WHITE else NONE

    cdef int _other_player(self) nogil:
        return WHITE if self.turn == BLACK else BLACK if self.turn == WHITE else NONE

    cpdef bint can_bear_off(self):
        return self._can_bear_off()

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef bint _can_bear_off(self) nogil:
        """Returns True if current player can bear off pieces"""
        cdef int lower, upper, i
        if self.turn == WHITE:
            lower = 6
            upper = 24
        else:
            lower = 0
            upper = 18

        for i in range(lower, upper, 1):
            if self.board[self.turn, i] > 0:
                return False
        return True

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef void do_move(self, Move move):
        if self.is_nature_turn():
            if self.has_game_started():
                self._generate_piece_moves_from_dice(move)
                self.set_nature_turn(False)  # Must be set BEFORE generate_movement_moves for correct lookahead
                self.generate_movement_moves()
                if len(self.legal_moves) == 0:
                    self._goto_next_turn()
            else:
                if move.i == move.j:
                    self.generate_pre_game_2d6_moves()
                else:
                    if move.i > move.j:
                        self.set_turn(WHITE)
                    else:
                        self.set_turn(BLACK)
                    self.game_has_started = True
                    self._generate_piece_moves_from_dice(move)
                    self.set_nature_turn(False)  # Must be set BEFORE generate_movement_moves for correct lookahead
                    self.generate_movement_moves()
                    if len(self.legal_moves) == 0:
                        self._goto_next_turn()
        else:
            #Move piece
            #Moving off bar
            if move.src in [-1, 24]:
                self.bar[self.turn] -= 1
            #Movement
            else:
                self.board[self.turn, move.src] -= 1
            #Bearing off
            if move.dst in [-1, 24]:
                self.beared_off[self.turn] += 1
            #Movement
            else:
                if self.board[self._other_player(), move.dst] == 1:
                    self.bar[self._other_player()] += 1
                    self.board[self._other_player(), move.dst] = 0
                self.board[self.turn, move.dst] += 1
                
            # Remove the used die.
            # Logic: remove the first occurrence of move.n
            if move.n in self._piece_moves_left:
                self._piece_moves_left.remove(move.n)
            else:
                # Fallback? Should not happen if move is legal.
                pass
                
            if len(self._piece_moves_left) == 0:
                self._goto_next_turn()
            else:
                self.generate_movement_moves()
                if len(self.legal_moves) == 0:
                    self._goto_next_turn()

    cdef void _goto_next_turn(self):
        self.check_for_winner()
        if not self.game_ended():
            self.turn = BLACK if self.turn == WHITE else WHITE
            self.set_nature_turn(True)
            self._piece_moves_left = []
            self.generate_nature_moves()
            if self.turn == WHITE:
                self.turn_number += 1

    cpdef bint game_ended(self):
        return self.winner != NONE

    cpdef bint check_for_winner(self):
        if self.n_beared_off_pieces(self.turn) == 15:
            self.winner = self.turn

    cpdef signed char n_pieces_on_bar(self, int player):
        return self.bar[player]

    cpdef signed char n_pieces_on_board(self, int player):
        return np.sum(self.board[player]) + self.bar[player]

    cpdef signed char n_beared_off_pieces(self, int player):
        return self.beared_off[player]

    cpdef void generate_pre_game_2d6_moves(self):
        self.legal_moves = self.get_roll_2d6_moves()

    cpdef void generate_nature_moves(self):
        """Generates all possible rollings of a 2d6 as the possible legal moves"""
        self.legal_moves = self.get_roll_2d6_moves()

    cpdef list get_roll_2d6_moves(self):
        """Generates all possible rollings of a 2d6."""
        cdef int i, j
        cdef Move move
        cdef list moves = []
        for i in range(1, 7):
            for j in range(1, 7):
                move = Move()
                move.i = i
                move.j = j
                move.is_movement_move = False
                moves.append(move)
        return moves
