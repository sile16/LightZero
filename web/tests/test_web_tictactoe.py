from web.games.tictactoe import PlayerSpec, TicTacToeGame


def test_tictactoe_human_move():
    game = TicTacToeGame()
    session = game.new_session(
        {
            1: PlayerSpec(player_type="human"),
            2: PlayerSpec(player_type="bot", bot_type="random"),
        },
        auto_play=False,
    )
    state = session.state()
    assert state["game"] == "tictactoe"
    assert state["legal_actions"]
    action = state["legal_actions"][0]
    session.apply_human_action(action)
    next_state = session.state()
    assert next_state["last_action"] == action


def test_tictactoe_supports_alphazero():
    game = TicTacToeGame()
    assert "alphazero" in game.supported_policy_configs
