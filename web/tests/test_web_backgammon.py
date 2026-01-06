from web.games.backgammon import BackgammonGame
from web.games.common import PlayerSpec


def test_backgammon_human_move():
    game = BackgammonGame()
    session = game.new_session(
        {
            1: PlayerSpec(player_type="human"),
            2: PlayerSpec(player_type="bot", bot_type="random"),
        },
        auto_play=False,
    )
    state = session.state()
    assert state["game"] == "backgammon"
    if state["legal_actions"]:
        action = state["legal_actions"][0]
        session.apply_human_action(action)
        next_state = session.state()
        assert next_state["last_action"] == action


def test_backgammon_supports_alphazero():
    game = BackgammonGame()
    assert "alphazero" in game.supported_policy_configs
