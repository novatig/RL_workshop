#!/usr/bin/env python

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

class Environment:
    """OpenAI Gym environment wrapper."""

    def __init__(self, environmentName, numEpisodesPerEvaluation, renderEvery):
        self.env = gym.make(environmentName).unwrapped
        self.numEpisodesPerEvaluation = numEpisodesPerEvaluation

        self.stateSize = self.env.observation_space.shape[0]  # State space size.
        self.actionSize = self.env.action_space.n             # Number of available actions.

        self.renderEvery = renderEvery

        self._episodeCounter = 0

    def performOneEpisode(self, policy):
        """Run one episode of the simulation.

        Start from an initial state, perform actions according to the given
        policy until the simulation is complete.

        Returns:
            states: List of states at the beginning of time steps.
            actions: Actions selected by the policy.
            networkEvaluations: Qvalues returned by policy (a list of torch objects!).
            rewards: List of rewards.
        """

        states  = []
        actions = []
        rewards = []
        networkEvaluations = []

        done  = False
        state = self.env.reset()
        while not done:
            # TODO: Get action from the policy, update the environment,
            
            
            states.append(state)
            action, qvalue = policy.getAction(state)
            state, reward, done, info = self.env.step(action)
            actions.append(action)
            rewards.append(reward)
            networkEvaluations.append(qvalue)

            if self.renderEvery > 0 and self._episodeCounter % self.renderEvery == 0:
                self.env.render()

        self._episodeCounter += 1
        return states, actions, networkEvaluations, rewards


class DiscreteQFunctionPolicy:
    """Discrete Q function: gets state as input and outputs one Q for each action option."""

    def __init__(self, env, hiddenLayers=[32, 32]):
        layers = []
        # TODO: Replace a linear one-layer network with a better one: linear,
        #       softsign, linear, softsign, ..., linear
        print(hiddenLayers)
        layers.append(torch.nn.Linear(env.stateSize, hiddenLayers[0]))
        layers.append(torch.nn.Sigmoid())
        layers.append(torch.nn.Linear(hiddenLayers[0], hiddenLayers[1]))
        layers.append(torch.nn.Sigmoid())
        layers.append(torch.nn.Linear(hiddenLayers[1], env.actionSize))

        self.network = torch.nn.Sequential(*layers)

        # Initialize weights.
        for layer in self.network[::2]:
            torch.nn.init.xavier_uniform_(layer.weight)

    def getAction(self, state):
        """Compute Qvalues for the given state and sample an action."""
        # TODO: Evaluate the network to compute Qvalues.
        Qvalues = self.network(torch.from_numpy(state))

        # TODO: Sample from a softmax distribution with respect to Qvalues.
        sm      = torch.nn.Softmax(dim=0)
        entropy = sm(Qvalues)
        
        rands = np.random.multinomial(1, entropy.detach().numpy())
        a     = np.where(rands == 1)
        action = a[0].item()
        return action, Qvalues


def computeDiscountedRewards(rewards, gamma=0.99):
    """Compute discounted rewards for every simulation step.

    >>> computeDiscountedRewards([2000, 200, 20], 0.5)
    [2105.  210.   20.]
    """
    T = len(rewards)
    R = np.zeros(T)
    R[-1] = rewards[-1]
    for t in range(T-2, -1, -1):
        R[t] = gamma * R[t + 1] + rewards[t]
    return R


class SARSAAlgorithm:
    def __init__(self, Qnet, gamma=0.99, learnRate=0.001):
        self.gamma     = gamma
        self.optimizer = torch.optim.Adam(Qnet.network.parameters(), lr=learnRate)

    def trainPolicy(self, Qnet, env, numIterations):
        """Train the network for `numIterations` iterations."""
        for k in range(numIterations):
            self._trainPolicy(Qnet, env)

    def _trainPolicy(self, Qnet, env):
        """Run the simulation for `numEpisodesPerEvaluation` and update the network (once)."""
        N    = 0
        loss = 0
        sumRewards = np.zeros(env.numEpisodesPerEvaluation)
        
        for ep in range(env.numEpisodesPerEvaluation):
            # Evaluate the policy on one environment episode.
            states, actions, Qvalues, rewards = env.performOneEpisode(Qnet)
            # TODO: Compute the loss and total rewards.
            #
            

            sumRewards[ep] = np.sum(rewards)
            for i in range(len(states)-1):
                tdvalue = rewards[i] + self.gamma*Qvalues[i+1][actions[i+1]].detach()
                loss   += (Qvalues[i][actions[i]] - tdvalue)**2
            
            N += len(states)
            
        loss = 0.5/N*loss

        # Update the parameters of the policy.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(loss)
        print("Average R: {:6.2f}".format(np.mean(sumRewards)))


def main():
    parser = argparse.ArgumentParser(description="RL exercise.")
    ADD = parser.add_argument
    ADD('-e', '--environment', default='CartPole-v1', help="Name of the OpenAI gym environment to train on.")
    ADD('-H', '--hiddenLayers', nargs='+', type=int, default=[32,32], help="Size of the policy's hidden layers.")
    ADD('-m', '--numEpisodesPerEval', type=int, default=100, help="Number of episodes per policy update iteration.")
    ADD('-n', '--numIterations', type=int, default=1000, help="Number of policy updates.")
    ADD('-l', '--learningRate', type=float, default=0.1, help="Learning rate of policy update.")
    ADD('-g', '--gamma', type=float, default=0.99, help="Rewards discount factor.")
    ADD('-r', '--renderEvery', type=int, default=0, help="Render every nth episode. 0 to disable.")
    args = parser.parse_args()

    # Create the environment, the policy and the training algorithm.
    env = Environment(args.environment, args.numEpisodesPerEval, args.renderEvery)
    policy = DiscreteQFunctionPolicy(env, args.hiddenLayers)
    algo = SARSAAlgorithm(policy, gamma=args.gamma, learnRate=args.learningRate)

    # Train the policy.
    algo.trainPolicy(policy, env, args.numIterations)


if __name__ == '__main__':
    main()
