from easydict import EasyDict

# ==============================================================
# Pig Dice Game with Stochastic MuZero
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 10
num_simulations = 25
update_per_collect = 20  # Reduced for slower collection with long games
batch_size = 32  # Reduced for long games (600+ moves avg)
max_env_step = int(2e5)
reanalyze_ratio = 0.
# Stochastic MuZero settings
# Pig has 7 chance outcomes (0 = no roll, 1-6 = die faces)
chance_space_size = 7
use_ture_chance_label_in_chance_encoder = True
# ==============================================================

pig_stochastic_muzero_config = dict(
    exp_name=f'data_stochastic_mz/pig_stochastic_muzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0',
    env=dict(
        env_id='Pig',
        target_score=100,
        battle_mode='self_play_mode',
        bot_action_type='hold_at_20',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,  # Match env_num to avoid subprocess reset issues
        manager=dict(shared_memory=False),
        chance_space_size=chance_space_size,
    ),
    policy=dict(
        model=dict(
            observation_shape=5,  # [my_score, opp_score, turn_score, am_player_1, can_hold]
            action_space_size=2,  # roll or hold
            chance_space_size=chance_space_size,  # 0 = no roll, 1-6 = die faces
            model_type='mlp',
            latent_state_dim=64,
            num_res_blocks=1,
            num_channels=64,
            reward_head_hidden_channels=[32],
            value_head_hidden_channels=[32],
            policy_head_hidden_channels=[32],
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
            norm_type='BN',
            self_supervised_learning_loss=True,
        ),
        model_path=None,
        use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,
        use_wandb=True,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=400,  # Pig games can be 300+ moves
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        td_steps=100,  # Longer horizon for Pig games
        num_unroll_steps=10,  # More unroll steps for longer games
        discount_factor=1,  # Episodic game
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
pig_stochastic_muzero_config = EasyDict(pig_stochastic_muzero_config)
main_config = pig_stochastic_muzero_config

pig_stochastic_muzero_create_config = dict(
    env=dict(
        type='pig',
        import_names=['zoo.board_games.pig.envs.pig_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
)
pig_stochastic_muzero_create_config = EasyDict(pig_stochastic_muzero_create_config)
create_config = pig_stochastic_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
