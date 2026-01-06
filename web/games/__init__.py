from web.games.backgammon import BackgammonGame
from web.games.pig import PigGame
from web.games.tictactoe import TicTacToeGame


GAMES = {
    BackgammonGame.name: BackgammonGame(),
    PigGame.name: PigGame(),
    TicTacToeGame.name: TicTacToeGame(),
}
