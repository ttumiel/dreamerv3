import gym
import procgen

from embodied.envs import from_gym


def Procgen(task):
    return from_gym.FromGym(gym.make(f"procgen:procgen-{task}-v0"))
