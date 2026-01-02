import numpy as np


class BackgammonRandomBot:
    """Random bot for backgammon that selects uniformly from legal actions."""

    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        """
        Select a random legal action.

        Action encoding: action = source * 2 + die_slot
        - source: 0-23 (board points) or 24 (bar)
        - die_slot: 0 (first die) or 1 (second die)
        - Total action space: 50 actions
        """
        # Get legal actions from action mask
        action_mask = obs['action_mask']
        legal_actions = [i for i, x in enumerate(action_mask) if x == 1]

        if len(legal_actions) == 0:
            # Should not happen - env auto-handles no-move situations
            return 0

        # Pick a random legal action
        return legal_actions[np.random.randint(len(legal_actions))]
