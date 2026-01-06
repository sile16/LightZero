from web.games.common import PlayerSpec


def test_player_spec_num_simulations():
    spec = PlayerSpec(player_type="model", algo="muzero", num_simulations=50)
    assert spec.num_simulations == 50
