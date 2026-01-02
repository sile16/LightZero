from easydict import EasyDict

# ==============================================================
# Begin of the implementation
# ==============================================================
# Env Config
backgammon_alphazero_config = dict(
    exp_name='backgammon_alphazero_bot_mode',
    env=dict(
        env_id='backgammon',
        battle_mode='self_play_mode',
        obs_type='features',  # 'minimal' (40, 1, 25), 'standard' (47, 1, 25), or 'features' (52, 1, 25)
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(52, 1, 25),  # Updated for features obs_type
            action_space_size=50,  # 25 sources × 2 dice slots
            image_channel=52,
            # Backgammon has 50 actions (source × die_slot encoding).
            # We can use a smaller categorical output.
            categorical_distribution=False,
            # Standard AlphaZero ResNet configuration adapted for smaller width
            # (52, 1, 25) is small, so we don't need deep downsampling.
            downsample=dict(
                is_downsample=False,
            ),
            # Use a simple ResNet block structure
            num_res_blocks=2,
            num_channels=64,
            value_head_channels=16,
            policy_head_channels=16,
            fc_value_layers=[32],
            fc_policy_layers=[32],
        ),
        mcts=dict(
            num_simulations=100, # Start with 100 for speed
            max_env_step=None, 
        ),
        cuda=True,
        board_size=None, # Not needed for custom model shape
        update_per_collect=50,
        batch_size=256,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=8,
        eval_freq=1000,
        mcts_env_name='backgammon',
        mcts_env_config=dict(
            env_id='backgammon',
            battle_mode='self_play_mode',
            obs_type='features',
        ),
    ),
)

backgammon_alphazero_config = EasyDict(backgammon_alphazero_config)
main_config = backgammon_alphazero_config

backgammon_alphazero_create_config = dict(
    env=dict(
        type='backgammon',
        import_names=['zoo.board_games.backgammon.envs.backgammon_env'],
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
    ),
)
backgammon_alphazero_create_config = EasyDict(backgammon_alphazero_create_config)
create_config = backgammon_alphazero_create_config
