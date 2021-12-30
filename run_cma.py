import os
import pickle

import gym
import numpy as np
import panda_gym
import ray
import torch
import torch.nn as nn
from gym.envs.registration import make
from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation

from evolve_cma import Evaluator
from wrappers import DoneOnSuccessWrapper


def make_env():
    return FlattenObservation(
        FilterObservation(
            DoneOnSuccessWrapper(
                gym.make("PandaReachDense-v2", render=True), reward_offset=0
            ),
            filter_keys=["observation", "desired_goal"],
        )
    )


if __name__ == "__main__":
    ray.init(address=os.environ.get("ip_head"))

    with open("data/best_cma_5300.pkl", "rb") as f:
        genome = pickle.load(f)

    with torch.no_grad():
        eval = Evaluator.remote(render=True)
        fit = ray.get(eval.evaluate.remote(genome))
        print(fit)
