from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 10
num_simulations = 25
update_per_collect = 100  # More updates per collect for short games
batch_size = 64  # Reduced for short TicTacToe games (~5-9 moves each)
max_env_step = int(2e5)
reanalyze_ratio = 0.
# Stochastic MuZero settings for deterministic TicTacToe
# chance_space_size=1 means no stochastic events (deterministic game)
chance_space_size = 1
use_ture_chance_label_in_chance_encoder = True
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

tictactoe_stochastic_muzero_config = dict(
    exp_name=f'data_stochastic_mz/tictactoe_stochastic_muzero_bot-mode_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0',
    env=dict(
        battle_mode='self_play_mode',
        eval_bots=[
            dict(name='random', action_type='random'),
            dict(name='heuristic', action_type='v0'),
            dict(name='perfect', action_type='alpha_beta_pruning'),
        ],
        eval_matchups=[
            dict(bot='random', action_type='random', agent_role='first'),
            dict(bot='random', action_type='random', agent_role='second'),
            dict(bot='heuristic', action_type='v0', agent_role='first'),
            dict(bot='heuristic', action_type='v0', agent_role='second'),
            dict(bot='perfect', action_type='alpha_beta_pruning', agent_role='first'),
            dict(bot='perfect', action_type='alpha_beta_pruning', agent_role='second'),
        ],
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=100,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=(3, 3, 3),
            action_space_size=9,
            chance_space_size=chance_space_size,  # 1 for deterministic game
            image_channel=3,
            # We use the small size model for tictactoe.
            num_res_blocks=1,
            num_channels=16,
            reward_head_hidden_channels=[8],
            value_head_hidden_channels=[8],
            policy_head_hidden_channels=[8],
            reward_support_range=(-10., 11., 1.),
            value_support_range=(-10., 11., 1.),
            norm_type='BN',
            # Required for stochastic muzero chance encoder
            self_supervised_learning_loss=True,
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        use_ture_chance_label_in_chance_encoder=use_ture_chance_label_in_chance_encoder,
        use_wandb=True,
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=5,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,  # self-supervised learning loss weight
        # NOTE: In board_games, we set large td_steps to make sure the value target is the final outcome.
        td_steps=9,
        num_unroll_steps=3,
        # NOTE: In board_games, we set discount_factor=1.
        discount_factor=1,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e4),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
tictactoe_stochastic_muzero_config = EasyDict(tictactoe_stochastic_muzero_config)
main_config = tictactoe_stochastic_muzero_config

tictactoe_stochastic_muzero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
)
tictactoe_stochastic_muzero_create_config = EasyDict(tictactoe_stochastic_muzero_create_config)
create_config = tictactoe_stochastic_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)
