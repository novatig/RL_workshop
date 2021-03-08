from algorithms import BaseAlgorithm, computeDiscountedRewards
import numpy as np
import torch
import policies

#-------------------------------------------------------------------------------
# Implementation of the reinforce algorithm
#
class ValueIteration(BaseAlgorithm):

    def __init__(self, Qnet, gamma=0.99, learnrate=0.01, estimator='Qlearning'):
        assert estimator in ['MonteCarlo', 'SARSA', 'Qlearning', 'MinVar']
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(Qnet.layers.parameters(), lr=learnrate)
        self.estimator = estimator

    def _trainPolicy(self, Qnet, env):
        #
        loss = 0
        sumRewards = np.zeros(env.numEpisodesPerEvaluation)
        sumExpectRew = np.zeros(env.numEpisodesPerEvaluation)
        sumDiscountRew = np.zeros(env.numEpisodesPerEvaluation)
        #
        for ep in range(env.numEpisodesPerEvaluation): # Each new episode starts here
            #
            # Ealuate the policy on one environment episode
            states, actions, Qvalues, rewards = env.performOneEpisode(Qnet)
            # Compute the discounted rewards
            discountRew = computeDiscountedRewards(rewards, self.gamma)
            #
            actionsTorch = torch.tensor(actions).type(policies.torchIntType)
            Qvalues = torch.stack(Qvalues) # turn list into a torch tensor
            # Select actions Qvalues depending on which action was chosen:
            selectedQvalues = torch.gather(Qvalues, 1, actionsTorch.unsqueeze(1)).squeeze()
            #
            if self.estimator == 'MonteCarlo':
                targets = torch.tensor(discountRew)
            elif self.estimator == 'SARSA': # Temporal Difference Learning
                targets = torch.tensor(rewards)
                targets[:-1] += self.gamma * selectedQvalues[1:].detach()
            elif self.estimator == 'Qlearning': # Temporal Difference Learning
                targets = torch.tensor(rewards)
                maxQvals, _ = Qvalues.max(dim=1)
                targets[:-1] += self.gamma * maxQvals[1:].detach()
            elif self.estimator == 'MinVar': # Temporal Difference Learning
                targets = torch.tensor(rewards)
                Qnext = Qvalues[1:,:].detach()
                probabilities = torch.nn.functional.softmax(Qnext, dim=1)
                targets[:-1] += self.gamma * torch.sum(Qnext*probabilities, dim=1)
            #
            loss += torch.nn.functional.mse_loss(selectedQvalues, targets)
            #
            # Update mean reward
            sumRewards[ep] = np.sum(rewards)
            sumExpectRew[ep] = np.mean((selectedQvalues.data).cpu().numpy())
            sumDiscountRew[ep] = np.mean(discountRew)
        #
        # Update the parameters of the policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #
        file_object = open(self.estimator+'.txt', 'a')
        file_object.write('%e ' % np.mean(sumRewards))
        file_object.close()
        print('Average R: {0:6.2f} {1:6.2f} {2:6.2f} '.format(
          np.mean(sumRewards), np.mean(sumExpectRew), np.mean(sumDiscountRew)) )

    def trainPolicy(self, Qnet, env, numIterations):
        for k in range(numIterations):
            self._trainPolicy(Qnet, env)
        file_object = open(self.estimator+'.txt', 'a')
        file_object.write('\n')
        file_object.close()

