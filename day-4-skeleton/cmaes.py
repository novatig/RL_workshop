#!/usr/bin/env python

# Check the following tutorial:
# https://github.com/cselab/korali/blob/master/tutorials/basic/1.optimization/run-cmaes.py


from common import getNumNetworkParams     # Check what it does!
from common import overwriteNetworkParams  # Check what it does!
from common import Environment

import argparse
import korali
import numpy as np
import torch
import torch.nn as nn

class DiscretePolicy:
    """Discrete policy class: when the action is a choice between different options."""

    def __init__(self, env):
        # NOTE: Use a smaller network than before!
        self.network = nn.Sequential(
                nn.Linear(env.stateSize, 16),
                nn.Softsign(),
                # nn.Linear(16, 16),
                # nn.Softsign(),
                nn.Linear(16, env.actionSize),
                nn.Softmax(dim=-1))  # Note the softmax at the end!

    def getAction(self, state):
        # Evaluate the probability of choosing each option.
        probabilities = self.network(torch.tensor(state))

        # Convert the probabilities to numpy.
        probabilitiesNumpy = probabilities.data.cpu().numpy()

        # Select an action based on the options probabilities.
        action = np.random.choice(len(probabilitiesNumpy), p=probabilitiesNumpy)

        return action, probabilities[action]


class CMAESKorali:
    """CMA-ES-based training algorithm using Korali."""
    def __init__(self, policy, populationSize, sigma):
        self.populationSize = populationSize
        self.sigma = sigma

    def trainPolicy(self, policy, env, numIterations):
        """Evaluate `numIterations` updating iterations on the given policy for
        the given environment."""

        # Define the objective.
        def objective(p):
            """Update the policy with given parameters, evalute and return the reward."""
            weights = p["Parameters"]

            # TODO: Update the network weights with values given by Korali (see overwriteNetworkParams).
            #       Run env.numEpisodesPerEvaluation episodes and compute the mean total reward.
            #
            #
            #
            #
            meanTotalReward = 0


            p["Evaluation"] = meanTotalReward  # Korali v1.0.1.
            # p["F(x)"] = meanTotalReward        # Korali master branch (?)


        # Create Korali and problem objects.
        k = korali.Engine()
        e = korali.Experiment()

        # Configure the problem.
        e["Problem"]["Type"] = "Evaluation/Direct/Basic"  # Korali v1.0.1.
        # e["Problem"]["Type"] = "Optimization/Stochastic"  # Korali master branch (?)
        e["Problem"]["Objective Function"] = objective

        # TODO: Define the problem variables (see getNumNetworkParams).
        #
        #       Don't forget to set all parameter values: initial mean, variance, lower/upper bound!
        #
        #
        #
        #

        # TODO: Set up CMA-ES.
        #
        #
        #
        #

        # Run Korali.
        k.run(e)


def main():
    parser = argparse.ArgumentParser(description="RL exercise.")
    ADD = parser.add_argument
    ADD('-e', '--environment', default='CartPole-v1', help="Name of the OpenAI gym environment to train on.")
    ADD('-m', '--numEpisodesPerEval', type=int, default=100, help="Number of episodes per policy update iteration.")
    ADD('-n', '--numIterations', type=int, default=1000, help="Number of policy updates.")
    ADD('-r', '--renderEvery', type=int, default=0, help="Render every nth episode. 0 to disable.")
    ADD('-z', '--sigma', type=float, default=0.2, help="Standard deviation of policy for continuous actions. Exploration noise.")
    ADD('-p', '--populationSize', type=int, default=64, help="Population size.")
    args = parser.parse_args()

    # Create the environment, the policy and the training algorithm.
    env = Environment(args.environment, args.numEpisodesPerEval, args.renderEvery)
    policy = DiscretePolicy(env)
    algo = CMAESKorali(policy, populationSize=args.populationSize, sigma=args.sigma)

    # Train the policy.
    algo.trainPolicy(policy, env, args.numIterations)


if __name__ == '__main__':
    main()
