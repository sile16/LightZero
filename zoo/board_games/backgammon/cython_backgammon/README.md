# Cython Backgammon Engine

High-performance Cython-based backgammon game engine for LightZero.

## Building

```bash
cd zoo/board_games/backgammon/cython_backgammon
python setup.py build_ext --inplace
```

## Requirements

- Python >= 3.8
- Cython >= 3.0
- NumPy >= 1.20

## Components

- `state.pyx` - Core game state and move generation
- `mcts.pyx` - Monte Carlo Tree Search implementation
- `utils.pyx` - Utility functions

## Usage

This engine is used by `BackgammonEnv` in `zoo/board_games/backgammon/envs/backgammon_env.py`.
