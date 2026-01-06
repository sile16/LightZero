from easydict import EasyDict

# ==============================================================
# MuZero TicTacToe with Multi-Bot Evaluation
# ==============================================================
collector_env_num = 8
n_episode = 16            # More diverse data per cycle
evaluator_env_num = 4     # 1 env per matchup - less parallelism = less race conditions
num_simulations = 50      # Better MCTS targets
update_per_collect = 25   # Better data/training ratio
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.
# ==============================================================

tictactoe_muzero_config = dict(
    exp_name='data_mz/tictactoe_muzero_eval_matchups_seed0',
    env=dict(
        board_size=3,
        battle_mode='self_play_mode',
        bot_action_type='v0',
        channel_last=False,
        # Eval matchups - 4 matchups (2 bots x 2 roles)
        eval_matchups=[
            dict(bot='random', action_type='random', agent_role='first'),
            dict(bot='random', action_type='random', agent_role='second'),
            dict(bot='heuristic', action_type='v0', agent_role='first'),
            dict(bot='heuristic', action_type='v0', agent_role='second'),
        ],
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=40,  # 10 episodes per matchup
        manager=dict(shared_memory=False),
        # Simulation env settings
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        scale=True,
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 3, 3),
            action_space_size=9,
            image_channel=3,
            num_res_blocks=1,
            num_channels=16,
            reward_head_hidden_channels=[8],
            value_head_hidden_channels=[8],
            policy_head_hidden_channels=[8],
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
        ),
        use_wandb=True,
        wandb_project='LightZero - TicTacToe',
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=9,  # Full TicTacToe game length
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        # Board game specific settings
        td_steps=9,  # Full game length for value targets
        num_unroll_steps=3,
        discount_factor=1,  # Episodic game
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

tictactoe_muzero_config = EasyDict(tictactoe_muzero_config)
main_config = tictactoe_muzero_config

tictactoe_muzero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
tictactoe_muzero_create_config = EasyDict(tictactoe_muzero_create_config)
create_config = tictactoe_muzero_create_config

if __name__ == '__main__':
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
