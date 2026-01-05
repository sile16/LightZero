# Experiment Logbook

## 2048 Baseline

- Status: running
- Progress: ~2M frames/steps (target 100M)
- Summary: early learning is OK so far; continue to 100M before applying new techniques.
- Config: `zoo/game_2048/config/stochastic_muzero_2048_config.py`
- Command:
  ```
  cd /home/sile/github/LightZero
  source venv-lightzero/bin/activate
  PYTHONPATH=$PWD python zoo/game_2048/config/stochastic_muzero_2048_config.py
  ```
- Notes: keep baseline config unchanged for comparability; use wandb curves for eval return/AUC.
