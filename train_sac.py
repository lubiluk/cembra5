import os
import subprocess
import sys

import gym
import panda_gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from algos import SAC
from algos.common import replay_buffer
from algos.sac import core
from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

save_path = "./data/reach_su"
exp_name = "reach_su"


def wrap(env):
    return FlattenObservation(
            FilterObservation(
                DoneOnSuccessWrapper(env, reward_offset=0),
                filter_keys=["observation", "desired_goal"],
            )
        )

if __name__ == "__main__":
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')

    os.makedirs(save_path, exist_ok=True)

    env = wrap(gym.make("PandaReachDense-v2"))

    ac_kwargs = dict(
        hidden_sizes=[10, 10], activation=nn.ReLU
    )
    rb_kwargs = dict(size=100_000)

    logger_kwargs = dict(output_dir=save_path, exp_name=exp_name)

    model = SAC(
        env=env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=ac_kwargs,
        replay_buffer=replay_buffer.ReplayBuffer,
        rb_kwargs=rb_kwargs,
        max_ep_len=50,
        batch_size=256,
        gamma=0.95,
        lr=0.0003,
        update_after=512,
        update_every=512,
        logger_kwargs=logger_kwargs,
        use_gpu_buffer=True,
    )

    model.train(steps_per_epoch=1024, epochs=5000)

    from algos.test_policy import load_policy_and_env, run_policy

    _, get_action = load_policy_and_env(save_path)

    run_policy(env, get_action)
