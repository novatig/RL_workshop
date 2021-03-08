#!/usr/bin/env python

from common import getNumNetworkParams     # Check what it does!
from common import overwriteNetworkParams  # Check what it does!
from common import Environment

import argparse
import numpy as np
import torch
import torch.nn as nn


class DiscretePolicy:
    """Discrete policy class: when the action is a choice between different options."""

    def __init__(self, env):
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


class EvolutionStrategies:
    def __init__(self, policy, populationSize, sigma, learnRate):
        self.populationSize = populationSize
        self.sigma = sigma
        self.learnRate = learnRate
        self.numParams = getNumNetworkParams(policy.network)
        self.meanWeights = torch.zeros((self.numParams))
        print("Number of policy parameters =", self.numParams)

    def _trainPolicy(self, policy, env):
        # Generate a population of parameters and evaluate each:
        noisePopulation = torch.randn((self.populationSize, self.numParams))
        expectedRewards = np.zeros(self.populationSize)

        for p in range(self.populationSize):
            # Perturb weights:
            weights = self.meanWeights + self.sigma * noisePopulation[p, :]

            # Write weights into policy:
            overwriteNetworkParams(policy.network, weights)

            # Evaluate policy:
            for ep in range(env.numEpisodesPerEvaluation):
                _, _, _, rewards = env.performOneEpisode(policy)
                expectedRewards[p] += np.sum(rewards)
            expectedRewards[p] /= env.numEpisodesPerEvaluation

        # Standardize rewards to reduce sensitivity to arbitrary learning rate:
        meanR, stdevR = np.mean(expectedRewards), np.std(expectedRewards)
        scaledRewards = (expectedRewards - meanR) / stdevR

        # Update weights : coefficient * J(w)
        updateCoef = self.learnRate / self.sigma / self.populationSize
        updateStep = torch.tensor(updateCoef * scaledRewards)
        self.meanWeights += torch.matmul(updateStep, noisePopulation)

        # Update sigma (optional):
        dSigma = 0
        for p in range(self.populationSize):
            dSigma += updateStep[p] * (torch.sum(noisePopulation[p,:]**2) - 1)
        self.sigma += float(dSigma / self.numParams)

        print('Average R: {0:6.2f} : Max R {1:6.2f} | sigma: {2:6.4f}' \
          .format(np.mean(expectedRewards), np.max(expectedRewards), self.sigma))

    def trainPolicy(self, policy, env, numIterations):
        for k in range(numIterations):
            self._trainPolicy(policy, env)


def main():
    parser = argparse.ArgumentParser(description="RL exercise.")
    ADD = parser.add_argument
    ADD('-e', '--environment', default='CartPole-v1', help="Name of the OpenAI gym environment to train on.")
    ADD('-m', '--numEpisodesPerEval', type=int, default=100, help="Number of episodes per policy update iteration.")
    ADD('-n', '--numIterations', type=int, default=1000, help="Number of policy updates.")
    ADD('-r', '--renderEvery', type=int, default=0, help="Render every nth episode. 0 to disable.")
    ADD('-l', '--learningRate', type=float, default=0.1, help="Learning rate of policy update.")
    ADD('-z', '--sigma', type=float, default=0.2, help="Standard deviation of policy for continuous actions. Exploration noise.")
    ADD('-p', '--populationSize', type=int, default=64, help="Population size.")
    args = parser.parse_args()

    # Create the environment, the policy and the training algorithm.
    env = Environment(args.environment, args.numEpisodesPerEval, args.renderEvery)
    policy = DiscretePolicy(env)
    algo = EvolutionStrategies(policy, populationSize=args.populationSize,
                               sigma=args.sigma, learnRate=args.learningRate)

    # Train the policy.
    algo.trainPolicy(policy, env, args.numIterations)


if __name__ == '__main__':
    main()
