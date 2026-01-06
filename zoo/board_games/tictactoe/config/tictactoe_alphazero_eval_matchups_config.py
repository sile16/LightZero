from easydict import EasyDict

# ==============================================================
# AlphaZero TicTacToe with Multi-Bot Evaluation
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 8  # 2 envs per matchup (4 matchups)
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
mcts_ctree = False  # Use Python MCTS tree (ctree not built)
# ==============================================================

tictactoe_alphazero_config = dict(
    exp_name='data_az/tictactoe_alphazero_eval_matchups_seed0',
    env=dict(
        board_size=3,
        battle_mode='self_play_mode',
        bot_action_type='v0',
        channel_last=False,
        # Eval matchups - 2 bots x 2 roles = 4 matchups (skip slow alpha-beta)
        eval_matchups=[
            dict(bot='random', action_type='random', agent_role='first'),
            dict(bot='random', action_type='random', agent_role='second'),
            dict(bot='heuristic', action_type='v0', agent_role='first'),
            dict(bot='heuristic', action_type='v0', agent_role='second'),
        ],
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=80,  # 20 episodes per matchup
        manager=dict(shared_memory=False),
        # Simulation env settings
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        scale=True,
        alphazero_mcts_ctree=mcts_ctree,
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        simulation_env_id='tictactoe',
        simulation_env_config_type='self_play',
        model=dict(
            observation_shape=(3, 3, 3),
            action_space_size=9,
            num_res_blocks=1,
            num_channels=16,
            value_head_hidden_channels=[8],
            policy_head_hidden_channels=[8],
        ),
        use_wandb=True,
        cuda=True,
        board_size=3,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        learning_rate=0.003,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(2e3),
        mcts=dict(num_simulations=num_simulations),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

tictactoe_alphazero_config = EasyDict(tictactoe_alphazero_config)
main_config = tictactoe_alphazero_config

tictactoe_alphazero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
tictactoe_alphazero_create_config = EasyDict(tictactoe_alphazero_create_config)
create_config = tictactoe_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
