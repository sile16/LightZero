from web.games.pig import PigGame
from web.games.common import PlayerSpec


def test_pig_human_roll():
    game = PigGame()
    session = game.new_session(
        {
            1: PlayerSpec(player_type="human"),
            2: PlayerSpec(player_type="bot", bot_type="random"),
        },
        auto_play=False,
    )
    state = session.state()
    assert state["game"] == "pig"
    session.apply_human_action(0)
    next_state = session.state()
    assert next_state["last_action"] == 0
