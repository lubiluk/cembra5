import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cma
import sys
import subprocess
import ray
from ray.tune.registry import register_env

from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation


class Model(nn.Module):
    def __init__(self, input_dim, hidden_sizes, out_dim, activation, act_limit):
        super().__init__()
        self.act_limit = act_limit

        layers = [nn.Linear(input_dim, hidden_sizes[0], True), activation()]

        for i in range(len(hidden_sizes) - 1):
            layers += [
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], True),
                activation(),
            ]

        layers += [nn.Linear(hidden_sizes[-1], out_dim, bias=True)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.act_limit

    def genotype(self, g=None):
        if g is not None:
            params = torch.clone(torch.from_numpy(g))

            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                w_shape = layer.weight.data.shape
                w_shape_flat = w_shape[0] * w_shape[1]
                b_shape = layer.bias.data.shape
                b_shape_flat = b_shape[0]

                w = params[:w_shape_flat]
                params = params[w_shape_flat:]
                layer.weight.data = w.reshape(w_shape).float()
                b = params[:b_shape_flat]
                params = params[b_shape_flat:]
                layer.bias.data = b.float()
        else:
            params = torch.tensor([])
            for layer in self.net:
                if not isinstance(layer, nn.Linear):
                    continue

                w = layer.weight.data.flatten()
                b = layer.bias.data.flatten()
                f = torch.cat([w, b])
                params = torch.cat([params, f])

            return params


@ray.remote
class Evaluator:
    def __init__(self) -> None:
        import gym
        from gym.envs.registration import make
        import panda_gym

        def make_env():
            return FlattenObservation(
                FilterObservation(
                    DoneOnSuccessWrapper(gym.make("PandaReachDense-v1"), reward_offset=0),
                    filter_keys=["observation", "desired_goal"],
                )
            )

        self.env = make_env()
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        act_limit = self.env.action_space.high[0]
        self.model = Model(
            input_dim=obs_dim,
            hidden_sizes=[32,32],
            out_dim=act_dim,
            activation=nn.ReLU,
            act_limit=act_limit)

    def evaluate(self, genotype):
        with torch.no_grad():
            self.model.genotype(genotype)
            rets = []
            for _ in range(10):
                obs = self.env.reset()
                done = False
                ret = 0
                while not done:
                    action = self.model.forward(torch.from_numpy(obs)).numpy()
                    obs, rew, done, _ = self.env.step(action)
                    ret += rew
                rets.append(ret)

            return -(sum(rets) / len(rets))

    def genome_shape(self):
        return self.model.genotype().shape


if __name__ == "__main__":
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')

    with torch.no_grad():
        eval = Evaluator.remote()
        genome_shape = ray.get(eval.genome_shape.remote())
        genome = torch.zeros(genome_shape)
        del eval

        es = cma.CMAEvolutionStrategy(genome.numpy(), 0.5)

        eval_pool = [Evaluator.remote() for _ in range(4)]

        while not es.stop():
            genotype = es.ask()
            fitness_remotes = []

            for i, g in enumerate(genotype):
                fr = eval_pool[i % len(eval_pool)].evaluate.remote(g)
                fitness_remotes.append(fr)

            fitness = ray.get(fitness_remotes)
            es.tell(genotype, fitness)
            es.logger.add()
            es.disp()

        es.result_pretty()
