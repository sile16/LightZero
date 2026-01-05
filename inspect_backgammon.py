import gym
import gym_backgammon

env = gym.make('gym_backgammon:Backgammon-v0')
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")
try:
    print(f"Action Sample: {env.action_space.sample()}")
except:
    pass
env.close()
