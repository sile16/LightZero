from zoo.board_games.backgammon.config.backgammon_alphazero_bot_mode_config import main_config, create_config
from lzero.entry import train_alphazero

if __name__ == '__main__':
    train_alphazero(main_config, create_config, seed=0, model_path=None)
