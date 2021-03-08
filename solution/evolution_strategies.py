from algorithms import BaseAlgorithm
from algorithms import getNumNetworkParams, overwriteNetworkParams
import torch
import policies
import numpy as np

class EvolutionStrategies(BaseAlgorithm):
    def __init__(self, policy, populationSize=32, sigma=0.1, learnRate=0.01):
        self.populationSize = populationSize
        self.sigma = sigma
        self.learnRate = learnRate
        self.numParams = getNumNetworkParams(policy.layers)
        self.meanWeights = torch.zeros((self.numParams))
        print("Number of policy parameters =", self.numParams)

    def _trainPolicy(self, policy, env):
        #
        # Generate a population of parameters and evaluate each:
        noisePopulation = torch.randn((self.populationSize, self.numParams))
        allReturns = np.zeros([self.populationSize, env.numEpisodesPerEvaluation])
        #
        for p in range(self.populationSize):
            #
            # Perturb weights:
            weights = self.meanWeights + self.sigma * noisePopulation[p,:]
            #
            # Write weights into policy:
            overwriteNetworkParams(policy.layers, weights)
            #
            # Evaluate policy:
            for ep in range(env.numEpisodesPerEvaluation):
                _, _, _, rewards = env.performOneEpisode(policy)
                allReturns[p][ep] = np.sum(rewards)
        #
        # Standardize rewards to reduce sensitivity to arbitrary learning rate:
        expectedRewards = np.mean(allReturns, axis=1)
        meanR, stdevR = np.mean(expectedRewards), np.std(expectedRewards)
        scaledRewards = (expectedRewards - meanR)/ stdevR
        #
        # Update weights : coefficient * J(w)
        updateCoef = self.learnRate / self.sigma / self.populationSize
        updateStep = torch.tensor(updateCoef*scaledRewards).type(policies.torchFloatType)
        self.meanWeights += torch.matmul(updateStep, noisePopulation)
        #
        # Update sigma (optional) :
        dSigma = torch.matmul(updateStep, torch.sum(noisePopulation**2, dim=1) - 1)
        self.sigma += float(dSigma / self.numParams)
        #
        file_object = open('ES.txt', 'a')
        file_object.write('%e ' % np.max(expectedRewards))
        file_object.close()
        print('Average R: {0:6.2f} : Max R {1:6.2f} | sigma: {2:6.4f}' \
          .format(np.mean(meanR), np.max(expectedRewards), self.sigma))

    def trainPolicy(self, policy, env, numIterations):
        for k in range(numIterations):
            self._trainPolicy(policy, env)
        file_object = open('ES.txt', 'a')
        file_object.write('\n')
        file_object.close()
