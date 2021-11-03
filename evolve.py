import torch
import torch.nn as nn
import gym
import panda_gym
from wrappers import DoneOnSuccessWrapper
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation

# No grad needed at all
torch.autograd.set_grad_enabled(False)

evaluation_runs = 10
population_size = 100

def make_model():
    return nn.Sequential(
        nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3)
    )


def phenotype(genome):
    model = make_model()
    linear_layers = [l for l in model if isinstance(l, nn.Linear)]
    layer_genome_lenghts = [
        l.weight.flatten().shape[0] + l.bias.shape[0] for l in linear_layers
    ]
    genome_length = sum(layer_genome_lenghts)

    assert len(genome) == genome_length

    layer_genomes = genome.split(layer_genome_lenghts)

    for i in range(len(linear_layers)):
        layer = linear_layers[i]
        layer_genome = layer_genomes[i]
        weights, biases = layer_genome.split(layer.weight.flatten().shape[0])
        layer.weight = nn.Parameter(weights.reshape(layer.weight.shape))
        layer.bias = nn.Parameter(biases)

    return model


def make_genome(n = 1):
    model = make_model()
    linear_layers = [l for l in model if isinstance(l, nn.Linear)]
    layer_genome_lenghts = [
        l.weight.flatten().shape[0] + l.bias.shape[0] for l in linear_layers
    ]
    genome_length = sum(layer_genome_lenghts)

    return torch.rand((n, genome_length))


def wrap(env):
    return FlattenObservation(
        FilterObservation(
            DoneOnSuccessWrapper(env, reward_offset=0),
            filter_keys=["observation", "desired_goal"],
        )
    )


def evaluate(genome, env, render=False):
    model = phenotype(genome)
    fitness = 0

    for _ in range(evaluation_runs):
        obs = env.reset()
        obs = torch.from_numpy(obs)
        done = False

        while not done:
            action = model(obs).cpu().numpy()
            obs, rew, done, _ = env.step(action)
            obs = torch.from_numpy(obs)
            fitness += rew
            if render:
                env.render()
                time.sleep(1/240)

    return fitness / evaluation_runs


def mutate(genome):
    copy = genome.clone()
    mutation_prob = 0.02
    mutation_mag = 0.02
    indxs = torch.rand_like(copy) < mutation_prob
    copy[indxs] += (
        copy[indxs] * (torch.rand_like(copy[indxs]) * 2 - 1) * mutation_mag
    )

    return copy


def crossover(genome1, genome2):
    copy1  = genome1.clone()
    copy2  = genome2.clone()
    probs = torch.rand_like(genome1) < 0.5
    copy1[probs < 0.5] = 0
    copy2[probs >= 0.5] = 0

    return copy1 + copy2

import time

def main():
    env = wrap(gym.make("PandaReachDense-v1"))
    # viz_env = wrap(gym.make("PandaReachDense-v1", render=True))

    population = make_genome(population_size)
    fitness = torch.zeros(population_size)

    for generation in range(1000):
        for i in range(len(population)):
            fitness[i] = evaluate(population[i], env)

        _, topind = torch.topk(fitness, 10)
        pairind = torch.combinations(topind, r=2)
        offspring = torch.stack([crossover(g1, g2) for g1, g2 in population[pairind]])
        _, topind = torch.topk(fitness, 55)
        mutants = torch.stack([mutate(g) for g in population[topind]])
        population = torch.cat((offspring, mutants), 0)

        # Show the best
        topfit, topind = torch.topk(fitness, 1)
        print("Best fitness: {}".format(topfit.squeeze()))
        # evaluate(population[topind.squeeze()], viz_env, render=True)

    env.close()
    # viz_env.close()

if __name__ == "__main__":
    main()
