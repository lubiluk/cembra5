import gym
import numpy as np
import panda_gym
import torch
import torch.nn as nn
from gym.envs.registration import make
from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation

from evolve_cma import Model
from wrappers import DoneOnSuccessWrapper


def make_env():
    return FlattenObservation(
        FilterObservation(
            DoneOnSuccessWrapper(gym.make("PandaReachDense-v2", render=True), reward_offset=0),
            filter_keys=["observation", "desired_goal"],
        )
    )


if __name__ == "__main__":
    with torch.no_grad():
        env = make_env()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0]
        model = Model(
            input_dim=obs_dim,
            hidden_sizes=[32, 32],
            out_dim=act_dim,
            activation=nn.ReLU,
            act_limit=act_limit,
        )
        model.load_state_dict(torch.load("data/best_cma.pth"))

        for _ in range(100):
            obs = env.reset()
            done = False
            ret = 0
            while not done:
                action = model.forward(torch.from_numpy(obs)).numpy()
                obs, rew, done, _ = env.step(action)
                ret += rew
                env.render()
            print(ret)

