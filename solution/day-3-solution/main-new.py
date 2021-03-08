#!/usr/bin/env python

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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

        # Reset the environment to start a new episode.
        state = self.env.reset()

        done = False
        while not done:
            # Choose an action using the current policy.
            action, networkEvaluation = policy.getAction(state)

            states.append(state)   # Store the state before the time step.
            actions.append(action) # Store the action.
            networkEvaluations.append(networkEvaluation) # Store NN outputs.

            # Advance the simulation and overwrite state.
            state, reward, done, info = self.env.step(action)

            rewards.append(reward) # Store the reward after the time step.

            if self.renderEvery > 0 and self._episodeCounter % self.renderEvery == 0:
                self.env.render()

        self._episodeCounter += 1
        return states, actions, networkEvaluations, rewards


class DiscreteQFunctionPolicy:
    """Discrete Q function: gets state as input and outputs one Q for each action option."""
    def __init__(self, env):
        # Construct a network.
        self.network = nn.Sequential(
                nn.Linear(env.stateSize, 32),
                nn.Softsign(),
                nn.Linear(32, 32),
                nn.Softsign(),
                nn.Linear(32, env.actionSize))

        # Initialize weights.
        for layer in self.network[::2]:
            nn.init.xavier_uniform_(layer.weight)

    def getAction(self, state):
        """Compute Qvalues for the given state and sample an action."""
        # Evaluate the probability of choosing each option.
        Qvalues = self.network(torch.tensor(state))

        # Convert the Q values to softmax policy.
        probabilities = nn.functional.softmax(Qvalues, dim=0)

        # Select an action based on the options probabilities.
        probabilitiesNumpy = probabilities.data.cpu().numpy()
        action = np.random.choice(len(probabilitiesNumpy), p=probabilitiesNumpy)

        return action, Qvalues



class SARSAAlgorithm:
    def __init__(self, Qnet, gamma=0.99, learnRate=0.1):
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(Qnet.network.parameters(), lr=learnRate)

    def trainPolicy(self, Qnet, env, numIterations):
        """Train the network for `numIterations` iterations."""
        for k in range(numIterations):
            self._trainPolicy(Qnet, env)

    def _trainPolicy(self, Qnet, env):
        """Run the simulation for `numEpisodesPerEvaluation` and update the network (once)."""
        loss = 0
        sumRewards = np.zeros(env.numEpisodesPerEvaluation)

        for ep in range(env.numEpisodesPerEvaluation):
            # Evaluate the policy on one environment episode.
            states, actions, Qvalues, rewards = env.performOneEpisode(Qnet)
            T = len(states)

            actionsTorch = torch.tensor(actions).type(torch.LongTensor)
            Qvalues = torch.stack(Qvalues) # Turn list into a torch tensor.

            # Select actions Qvalues depending on which action was chosen.
            ## selectedQvalues = torch.gather(Qvalues, 1, actionsTorch.unsqueeze(1)).squeeze()
            selectedQvalues = torch.stack(
                    [Qvalues[t, actions[t]] for t in range(T)])

            # SARSA, temporal difference learning.
            ## targets[:-1] += self.gamma * selectedQvalues[1:].detach()
            targets = torch.tensor(rewards)
            for t in range(T - 1):
                targets[t] += self.gamma * selectedQvalues[t + 1].detach()

            ## loss += nn.functional.mse_loss(selectedQvalues, targets)
            s = 0
            for t in range(T):
                s += (selectedQvalues[t] - targets[t]) ** 2
            loss += s / T

            # Update mean reward.
            sumRewards[ep] = np.sum(rewards)

        # Update the parameters of the policy.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("Average R: {:6.2f}".format(np.mean(sumRewards)))


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
    policy = DiscreteQFunctionPolicy(env)
    algo = SARSAAlgorithm(policy, gamma=args.gamma, learnRate=args.learningRate)

    # Train the policy.
    algo.trainPolicy(policy, env, args.numIterations)


if __name__ == '__main__':
    main()
