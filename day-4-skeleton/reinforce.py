#!/usr/bin/env python

from common import Environment, computeDiscountedRewards

import argparse
import numpy as np
import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)

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

        # TODO: Gather all required information from all episodes.
        N    = 0
        loss = 0

        bold   = self.b
        self.b = 0
        
        for ep in range(env.numEpisodesPerEvaluation):
            # Evaluate the policy on one environment episode.
            _, _, policyEvaluations, rewards = env.performOneEpisode(policy)

            for i in range(len(rewards)):
                g = 1
                q = 0
                for j in range(len(rewards[i:])-1):
                    q += np.power(self.gamma, j) * rewards[i+j]
                    
                self.b += q
                loss    = loss - (q-bold)*torch.log(policyEvaluations[i])
                   
            N += len(rewards)
            
            
        # TODO: Compute the loss.
        loss = loss/N
        
        # Update the parameters of the policy (using the pointers inside of optimizer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss.detach()
        # Update the mean discounted rewards.
        self.b = self.b/N

        averageReward = self.b  # TODO: Compute mean reward per episode.

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
