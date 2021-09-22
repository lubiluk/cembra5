import os

import gym
import panda_gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from wrappers import DoneOnSuccessWrapper
from train_sac import wrap, save_path


if __name__ == "__main__":
    from algos.test_policy import load_policy_and_env, run_policy
    env = wrap(gym.make("PandaReachDense-v1", render=True))
    _, get_action = load_policy_and_env(save_path + '/best')
    run_policy(env, get_action)
