# LightZero Web Arena

A minimal single-session web UI for playing board games against bots or model checkpoints. The server runs inference; the browser renders the UI.

## Start

```bash
python web/server.py
```

Open `http://localhost:8000` (override with `LIGHTZERO_WEB_PORT`).

## Games

- Tic Tac Toe: grid UI, human vs bot/model, bot vs bot, model vs model.
- Pig: score/turn UI with Roll/Hold buttons.
- Backgammon: summary UI with legal-action dropdown.

## Checkpoints

Checkpoints are discovered under `data_*` folders and filtered by game name in the path. Unsupported algorithms are shown but disabled in the UI.

## Extending

Add a new game module under `web/games/` with a `Game` class providing:
- `name`, `label`, `supported_policy_configs`
- `new_session(players, auto_play)` returning a session with `state()` and `apply_human_action()`

Then register it in `web/games/__init__.py`.
