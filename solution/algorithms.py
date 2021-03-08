from abc import abstractmethod, ABC
import numpy as np
import torch
import policies

def product(array):
    """Compute the product of all numbers in the given array."""
    result = 1
    for x in array:
        result *= x
    return result


def getNumNetworkParams(layers):
    """Compute the number of parameters of a given neural network."""
    return sum(product(params.size()) for params in layers.parameters())


def overwriteNetworkParams(layers, packedParams):
    """Overwrite parameters of a given neural networks.

    Arguments:
        - layers - A neural network object (torch.nn.Sequential).
        - packedParams - A flat list of parameter values. Its length must
                         match the total number of parameters in the network.
    """
    assert isinstance(layers, torch.nn.Sequential), \
           "Expected a neural network, got `{}` instead.".format(layers)

    offset = 0
    for params in layers.parameters():
        if isinstance(params.data, torch.Tensor):
            size = product(params.size())
            if offset + size > len(packedParams):
                raise ValueError("Too few parameters given.")
            data = packedParams[offset : offset + size]
            data = torch.tensor(data).type(policies.torchFloatType)
            offset += size

            # Overwrite the torch parameters with parameters from Korali.
            params.data[:] = torch.reshape(data, params.size())
        else:
            raise TypeError("Unknown layer parameters data type `{}`.".format(type(data)))


def computeDiscountedRewards(rewards, gamma=0.99):
    """Compute discounted rewards for every simulation step."""
    T = len(rewards)
    R = np.zeros(T)
    R[-1] = rewards[-1]
    for t in range(T-2, -1, -1):
        R[t] = gamma * R[t + 1] + rewards[t]
    return R


class BaseAlgorithm(ABC):
    """Base algorithm class."""

    @abstractmethod
    def trainPolicy(self, policy, env, numIterations):
        """Train / update / improve the current policy using some strategy."""
        pass
