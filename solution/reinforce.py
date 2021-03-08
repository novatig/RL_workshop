from algorithms import BaseAlgorithm, computeDiscountedRewards
import numpy as np
import torch
import policies

class Reinforce(BaseAlgorithm):
    """Implementation of the REINFORCE algorithm."""

    def __init__(self, policy, gamma=0.99, learnrate=0.01):
        self.b = 0 # mean of discounted rewards
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(policy.layers.parameters(), lr=learnrate)

    def _trainPolicy(self, policy, env):
        batchActions           = []
        batchPolicyEvaluations = []
        batchDiscountedRewards = []
        sumRewards = np.zeros(env.numEpisodesPerEvaluation)
        for ep in range(env.numEpisodesPerEvaluation): # Each new episode starts here
            #
            # Ealuate the policy on one environment episode
            states, actions, policyEvaluations, rewards = env.performOneEpisode(policy)
            #
            # Compute the discounted rewards
            discountedRewards = computeDiscountedRewards(rewards, self.gamma)
            #
            # Update the lists
            batchActions.extend(actions)
            batchPolicyEvaluations.extend(policyEvaluations)
            batchDiscountedRewards.extend(discountedRewards)
            #
            # Update mean reward
            sumRewards[ep] = np.sum(rewards)
        #
        rewardsTorch = torch.tensor(batchDiscountedRewards).type(policies.torchFloatType)
        #
        # Compute advantages A = rewards - mean( discountedRewards(i-1) )
        advantagesTorch = rewardsTorch - self.b
        #
        # Loss function definition depends on the policy type
        loss = policy.computePolicyLoss(advantagesTorch, batchPolicyEvaluations, batchActions)
        #
        # Update the parameters of the policy (using the pointers inside of optimizer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #
        # Update the mean discounted rewards
        self.b = np.mean(batchDiscountedRewards)
        #
        file_object = open('PG.txt', 'a')
        file_object.write('%e ' % np.mean(sumRewards))
        file_object.close()
        print("Average R: {:6.2f} | Average discounted R: {:6.2f}"
              .format(np.mean(sumRewards), self.b))

    def trainPolicy(self, policy, env, numIterations):
        for k in range(numIterations):
            self._trainPolicy(policy, env)
        file_object = open('PG.txt', 'a')
        file_object.write('\n')
        file_object.close()
