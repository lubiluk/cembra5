from __future__ import print_function

import os
import pickle

import gym

import neat

# load the winner
with open("data/winner-feedforward", "rb") as f:
    c = pickle.load(f)

print("Loaded genome:")
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config-feedforward")
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

net = neat.nn.FeedForwardNetwork.create(c, config)
env = gym.make("CartPole-v0")

for runs in range(10):
        fitness = 0.0
        done = False
        obs = env.reset()
        env.render()

        while not done:
            prob = net.activate(obs)[0]
            act = 0 if prob < 0.5 else 1
            obs, rew, done, info = env.step(act)
            env.render()
            fitness += rew

        print(fitness)
        
