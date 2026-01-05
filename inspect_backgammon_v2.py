import gym
import gym_backgammon

try:
    env = gym.make('backgammon-v0')
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Try to see how actions look like
    # Usually backgammon has a dynamic action space (depends on dice)
    # Most gym envs force a static action space.
    
    print(f"Env Class: {type(env)}")
    
    # Check if it has a way to get legal actions
    if hasattr(env, 'get_valid_actions'):
        print("Has get_valid_actions")
    
    env.close()
except Exception as e:
    print(e)
