import gym
import gym_backgammon
from gym import envs

print("Registered Envs containing 'Backgammon':")
for env in envs.registry:
    if 'Backgammon' in env:
        print(env)

# Try direct instantiation if possible to bypass registry issues if any
try:
    env = gym.make('gym_backgammon:Backgammon-v0')
    print("Success with 'gym_backgammon:Backgammon-v0'")
except Exception as e:
    print(f"Failed 'gym_backgammon:Backgammon-v0': {e}")

try:
    env = gym.make('Backgammon-v0')
    print("Success with 'Backgammon-v0'")
except Exception as e:
    print(f"Failed 'Backgammon-v0': {e}")
