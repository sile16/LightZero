# Backgammon Agent Goals

Goal: Build a state-of-the-art backgammon agent that performs well under limited CPU/GPU resources.
We will enhance Stochastic MuZero with modern improvements and validate each change in isolation.

## Strategy

- Stochastic MuZero baseline with chance labels (21 dice outcomes).
- Add improvements one at a time, measuring sample efficiency on 2048, then on backgammon short_game.
- Once short_game is strong, resume training on full backgammon.

## Techniques to Add (In Order)

1. ReZero-style reanalyze for efficiency (based on https://arxiv.org/html/2404.16364v1).
2. Gumbel for Stochastic MuZero:
   - top-k trick
   - Gumbel Max trick
   - sequential halving
3. Phased MCTS simulations:
   - start with 2 simulations per move
   - increase target budget over training once policy improves
4. Curriculum learning:
   - short_game backgammon first
   - transition to full game after stable performance
5. Value head upgrade (research item):
   - 4-value head predicting win / gammon / opponent-gammon / backgammon
   - combine with Match Equity Table (MET) for match-level value

## Phased MCTS Reference (Paper)

Reference: https://arxiv.org/html/2310.11305v3
Notes: uses progressive simulation and simulation budget scheduling.

## Progressive Simulation (Paper-Aligned)

We implement the paper's Algorithm 1 (progressive simulation budget allocation).
Config knobs (in policy.progressive_simulation):
- total_iterations (I): total training iterations in the schedule
- total_budget (B): total simulations across all iterations
- n_min (N_min): minimum simulations per iteration
- n_max (N_max): maximum target simulations used early in schedule construction

Default behavior:
- total_budget defaults to total_iterations * baseline num_simulations
- n_max defaults to baseline num_simulations

## Validation Protocol

- 2048: detect higher learning efficiency per training step vs. baseline using wandb curves.
  - Primary: eval return at fixed steps (10M/50M/100M).
  - Secondary: AUC of eval return over steps; time-to-threshold.
  - Stability: variance across seeds.
- Backgammon short_game: reconfirm gains and stability (same metrics).
- Backgammon full game: final training and benchmarking (same metrics).

## TODO List

Status legend: [done], [in-progress], [blocked], [todo]

- [done] Stochastic MuZero backgammon env chance labels and config support.
- [done] Run Tic-Tac-Toe on stochastic MuZero (validate 2-player correct learning).
  - Finding: Stochastic MuZero struggles with deterministic games (chance_space_size=1).
  - Regular MuZero achieved perfect play (all draws) at 8,000 iterations.
  - Stochastic MuZero failed to learn at 567,000+ iterations on same game.
  - Recommendation: Use regular MuZero for deterministic 2-player games.
- [done] Run Pig on stochastic MuZero (validate multi-action per turn + stochastic info).
  - Finding: Required max_episode_steps=200 to prevent infinite games (avg was 1,738 steps).
  - Training progressed after config fix; dice chance labels (7) working correctly.
- [done] Run 2048 on Stochastic MuZero (validate stochastic environment learning).
  - Result: Training progressing well - reward mean ~6,500, max ~14,500 at 55k iterations.
  - Stochastic MuZero performs correctly on stochastic games (chance_space_size > 1).
- [done] Implement progressive simulation schedule (paper-aligned) for stochastic MuZero.
- [todo] Define ReZero reanalyze schedule and implement for stochastic MuZero.
- [todo] Validate ReZero change on 2048 (sample efficiency report).
- [todo] Reconfirm ReZero on backgammon short_game.
- [todo] Add Gumbel + top-k + sequential halving to stochastic MuZero.
- [todo] Validate Gumbel change on 2048.
- [todo] Reconfirm Gumbel on backgammon short_game.
- [todo] Validate progressive simulation schedule on 2048.
- [todo] Reconfirm progressive simulation on backgammon short_game.
- [todo] Curriculum learning: short_game to full game transition.
- [todo] Prototype 4-value head + MET evaluation (test on small game first).
- [todo] Backgammon-specific optimizations after core stability.

## Evaluator Improvements (Board Games)

The MuZero evaluator (`lzero/worker/muzero_evaluator.py`) was enhanced to provide separate
statistics for model-first vs model-second evaluation in board games:

- Tracks `model_first_reward_mean/std/count` and `model_second_reward_mean/std/count`
- For `env_type='board_games'`, half the eval envs have model go first, half have a random
  first move made (simulating bot-first) before the model's turn
- Logs separate stats at end of each evaluation cycle
- Note: True alternating first-player requires using env's `start_player_index` for accurate
  testing, as random first moves don't represent optimal opponent play

## Key Findings Summary

| Game | Algorithm | Iterations | Result |
|------|-----------|------------|--------|
| TicTacToe | Regular MuZero | 8,000 | Perfect play (all draws) |
| TicTacToe | Stochastic MuZero | 567,000+ | Failed to learn |
| 2048 | Stochastic MuZero | 55,000+ | Mean ~6,500, Max ~14,500 (learning) |
| Pig | Stochastic MuZero | In progress | Requires max_episode_steps fix |

**Takeaway**: Stochastic MuZero's chance encoder interferes with deterministic games.
Use regular MuZero for TicTacToe and similar deterministic 2-player games.

## Next Step

Define and implement ReZero-style reanalyze schedule for stochastic MuZero, then run 2048 A/B.
Lock seeds and baseline config references in LOGBOOK before starting A/B.
