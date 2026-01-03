from easydict import EasyDict

# ==============================================================
# Begin of the implementation
# ==============================================================
# Env Config
backgammon_stochastic_muzero_config = dict(
    exp_name='backgammon_stochastic_muzero_bot_mode',
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
            chance_space_size=21,  # 21 unordered dice outcomes
            image_channel=52,
            model_type='conv',
            num_res_blocks=2,
            num_channels=64,
            reward_head_channels=16,
            value_head_channels=16,
            policy_head_channels=16,
            reward_head_hidden_channels=[32],
            value_head_hidden_channels=[32],
            policy_head_hidden_channels=[32],
            reward_support_range=(-3., 4., 1.),
            value_support_range=(-3., 4., 1.),
            self_supervised_learning_loss=False,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        use_ture_chance_label_in_chance_encoder=True,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        battle_mode='self_play_mode',
        game_segment_length=200,
        update_per_collect=50,
        batch_size=256,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=100,
        reanalyze_ratio=0.,
        # NOTE: In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=200,
        num_unroll_steps=5,
        # NOTE: In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=8,
        eval_freq=1000,
        replay_buffer_size=int(1e5),
        collector_env_num=8,
        evaluator_env_num=5,
    ),
)

backgammon_stochastic_muzero_config = EasyDict(backgammon_stochastic_muzero_config)
main_config = backgammon_stochastic_muzero_config

backgammon_stochastic_muzero_create_config = dict(
    env=dict(
        type='backgammon',
        import_names=['zoo.board_games.backgammon.envs.backgammon_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
)
backgammon_stochastic_muzero_create_config = EasyDict(backgammon_stochastic_muzero_create_config)
create_config = backgammon_stochastic_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path)
