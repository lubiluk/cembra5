import torch
import gym
import panda_gym
from neat.nn.feed_forward import FeedForwardNetwork
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.filter_observation import FilterObservation
from wrappers import DoneOnSuccessWrapper

def wrap(env):
    return FlattenObservation(
            FilterObservation(
                DoneOnSuccessWrapper(env, reward_offset=0),
                filter_keys=["observation", "desired_goal"],
            )
        )

class PoleBalanceConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 9
    NUM_OUTPUTS = 3
    USE_BIAS = True

    ACTIVATION = "relu"
    SCALE_ACTIVATION = 1.0

    FITNESS_THRESHOLD = 195.0

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 1_000_000
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80


    def fitness_fn(self, genome):
        # OpenAI Gym
        env = wrap(gym.make("PandaReach-v1", render=False, reward_type="dense"))
        act_limit = env.action_space.high[0]

        fitness = 200
        phenotype = FeedForwardNetwork(genome, self)

        for i in range(10):
            done = False
            observation = env.reset()

            while not done:
                input = torch.Tensor([observation]).to(self.DEVICE)

                pred = phenotype(input)

                # scaling
                pi_action = torch.tanh(pred)
                pi_action = act_limit * pi_action
                pi_action = pi_action.detach().cpu().numpy().squeeze()

                observation, reward, done, info = env.step(pi_action)
                fitness += reward

        env.close()

        return fitness


import logging

import gym
import torch

import neat.population as pop
from visualize import draw_net
from neat.nn.feed_forward import FeedForwardNetwork

logger = logging.getLogger(__name__)

logger.info(PoleBalanceConfig.DEVICE)
neat = pop.Population(PoleBalanceConfig)
solution, generation = neat.run()

if solution is not None:
    logger.info("Found a Solution")
    draw_net(
        solution,
        view=True,
        filename="./images/reach-solution",
        show_disabled=True,
    )

    # OpenAI Gym
    env = wrap(gym.make("PandaReach-v1", render=True, reward_type="dense"))
    done = False
    observation = env.reset()

    phenotype = FeedForwardNetwork(solution, PoleBalanceConfig)

    torch.save(phenotype, "data/reach_neat")

    while not done:
        env.render()
        input = torch.Tensor([observation]).to(PoleBalanceConfig.DEVICE)

        pred = phenotype(input).detach().cpu().numpy().squeeze()
        observation, reward, done, info = env.step(pred)

        if done:
            observation = env.reset()

    env.close()