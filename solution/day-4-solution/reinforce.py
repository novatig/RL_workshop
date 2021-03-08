#!/usr/bin/env python

from common import Environment, computeDiscountedRewards

import argparse
import numpy as np
import torch
import torch.nn as nn


class DiscretePolicy:
    """Discrete policy class: when the action is a choice between different options."""

    def __init__(self, env):
        self.network = nn.Sequential(
                nn.Linear(env.stateSize, 32),
                nn.Softsign(),
                # nn.Linear(32, 32),
                # nn.Softsign(),
                nn.Linear(32, env.actionSize),
                nn.Softmax(dim=-1))  # Note the softmax at the end!

    def getAction(self, state):
        # Evaluate the probability of choosing each option.
        probabilities = self.network(torch.tensor(state))

        # Convert the probabilities to numpy.
        probabilitiesNumpy = probabilities.data.cpu().numpy()

        # Select an action based on the options probabilities.
        action = np.random.choice(len(probabilitiesNumpy), p=probabilitiesNumpy)

        return action, probabilities[action]


class ReinforceAlgorithm:
    """Implementation of the REINFORCE algorithm."""

    def __init__(self, policy, gamma, learnRate):
        self.b = 0  # Mean of discounted rewards.
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(policy.network.parameters(), lr=learnRate)

    def trainPolicy(self, policy, env, numIterations):
        for k in range(numIterations):
            self._trainPolicy(policy, env)

    def _trainPolicy(self, policy, env):
        batchPolicyEvaluations = []
        batchDiscountedRewards = []
        sumRewards = np.zeros(env.numEpisodesPerEvaluation)
        for ep in range(env.numEpisodesPerEvaluation):
            # Evaluate the policy on one environment episode.
            states, actions, policyEvaluations, rewards = env.performOneEpisode(policy)

            # Compute the discounted rewards.
            discountedRewards = computeDiscountedRewards(rewards, self.gamma)

            # Update the lists.
            batchPolicyEvaluations.extend(policyEvaluations)
            batchDiscountedRewards.extend(discountedRewards)

            # Update mean reward.
            sumRewards[ep] = np.sum(rewards)

        rewardsTorch = torch.tensor(batchDiscountedRewards)

        # Compute advantages A = rewards - mean( discountedRewards(i-1) ).
        advantagesTorch = rewardsTorch - self.b

        # Calculate actions log likelihood.
        logLikelihood = torch.log(torch.stack(batchPolicyEvaluations))

        # Multiply selected log likelihoods by A and return.
        loss = -(advantagesTorch * logLikelihood).mean()

        # Update the parameters of the policy (using the pointers inside of optimizer).
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the mean discounted rewards.
        self.b = np.mean(batchDiscountedRewards)

        averageReward = np.mean(sumRewards)

        print("Average R: {:6.2f} | Average discounted R: {:6.2f}"
              .format(averageReward, self.b))


def main():
    parser = argparse.ArgumentParser(description="RL exercise.")
    ADD = parser.add_argument
    ADD('-e', '--environment', default='CartPole-v1', help="Name of the OpenAI gym environment to train on.")
    ADD('-m', '--numEpisodesPerEval', type=int, default=100, help="Number of episodes per policy update iteration.")
    ADD('-n', '--numIterations', type=int, default=1000, help="Number of policy updates.")
    ADD('-l', '--learningRate', type=float, default=0.1, help="Learning rate of policy update.")
    ADD('-g', '--gamma', type=float, default=0.99, help="Rewards discount factor.")
    ADD('-r', '--renderEvery', type=int, default=0, help="Render every nth episode. 0 to disable.")
    args = parser.parse_args()

    # Create the environment, the policy and the training algorithm.
    env = Environment(args.environment, args.numEpisodesPerEval, args.renderEvery)
    policy = DiscretePolicy(env)
    algo = ReinforceAlgorithm(policy, gamma=args.gamma, learnRate=args.learningRate)

    # Train the policy.
    algo.trainPolicy(policy, env, args.numIterations)


if __name__ == '__main__':
    main()
