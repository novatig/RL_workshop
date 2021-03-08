# NO NEED TO CHANGE ANYTHING HERE!!

import gym
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


def computeDiscountedRewards(rewards, gamma):
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


def product(array):
    """Compute the product of all numbers in the given array."""
    result = 1
    for x in array:
        result *= x
    return result


def getNumNetworkParams(network):
    """Compute the number of parameters of a given neural network."""
    return sum(product(params.size()) for params in network.parameters())


def overwriteNetworkParams(network, packedParams):
    """Overwrite parameters of a given neural networks.

    Arguments:
        - network - A neural network object (torch.nn.Sequential).
        - packedParams - A flat list of parameter values. Its length must
                         match the total number of parameters in the network.
    """
    assert isinstance(network, torch.nn.Sequential), \
           "Expected a neural network, got `{}` instead.".format(network)

    offset = 0
    for params in network.parameters():
        if isinstance(params.data, torch.Tensor):
            size = product(params.size())
            if offset + size > len(packedParams):
                raise ValueError("Too few parameters given.")
            data = packedParams[offset : offset + size]
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            offset += size

            # Overwrite the torch parameters with parameters from Korali.
            params.data[:] = torch.reshape(data, params.size())
        else:
            raise TypeError("Unknown layer parameters data type `{}`.".format(type(data)))
