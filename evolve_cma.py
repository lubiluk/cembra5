import gym
from gym.envs.registration import make
import panda_gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cma


from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation


def make_env():
    return FlattenObservation(
        FilterObservation(
            DoneOnSuccessWrapper(gym.make("PandaReachDense-v1"), reward_offset=0),
            filter_keys=["observation", "desired_goal"],
        )
    )


class Model(nn.Module):
    def __init__(self, input_dim, hidden_sizes, out_dim, activation, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32, True),
            nn.ReLU(),
            nn.Linear(32, 32, True),
            nn.ReLU(),
            nn.Linear(32, 3, bias=True),
        )

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.act_limit

    def genotype(self):
        params = []
        for layer in self.net:
            if not isinstance(layer, nn.Linear):
                continue

            print(layer.bias.data)



if __name__ == "__main__":
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    model = Model(obs_dim, [32, 32], act_dim, nn.ReLU, act_limit)
    model.genotype()
