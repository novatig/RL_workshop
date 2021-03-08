import gym
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, environmentName, numEpisodesPerEvaluation, renderEvery):
        self.numEpisodesPerEvaluation = numEpisodesPerEvaluation
        self.env = gym.make(environmentName).unwrapped
        #
        # Get the size of the state from the environment
        self.stateSize = self.env.observation_space.shape[0]
        self.renderEvery = renderEvery
        self.totalEpCounter = 0
        #
        # Get the kind and number of actions from the environment
        dtype = self.env.action_space.dtype
        if dtype == 'int32' or dtype == 'int64':
            # Discrete action space
            self.actionType = 'discrete'
            # number of options for the discrete action:
            self.actionSize = self.env.action_space.n
        elif dtype == 'float32' or dtype == 'float64':
            # Continuous action space
            self.actionType = 'continuous'
            # number of components of the action vector:
            self.actionSize = len(self.env.action_space.sample())

    def performOneEpisode(self, policy):
        # Initialize empty lists to store experiences
        states  = []
        actions = []
        rewards = []
        networkEvaluations = []
        #
        # Reset the environment to start a new episode
        state = self.env.reset()
        #
        t = 0
        while True: # Episode loop
            #
            # Choose an action using the current policy
            action, networkEvaluation = policy.getAction(state)
            #
            states.append(state)   # store the state before the time step
            actions.append(action) # store the action
            networkEvaluations.append(networkEvaluation) # store NN outputs
            #
            # Advance the simulation and overwrite state
            state, reward, doneEpisode, info = self.env.step(action)
            t += 1
            #
            if self.renderEvery > 0 and self.totalEpCounter % self.renderEvery == 0:
                self.env.render()
            #
            rewards.append(reward) # store the reward after the time step
            #
            if doneEpisode or t > 500: # The episode has ended
                break
        #
        self.totalEpCounter += 1
        return states, actions, networkEvaluations, rewards
