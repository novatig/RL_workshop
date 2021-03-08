import numpy as np
import torch

useCuda = False
if useCuda:
    torchIntType = torch.cuda.LongTensor
    #torchFloatType = torch.cuda.FloatTensor # single precision
    torchFloatType = torch.cuda.DoubleTensor # double precision
else:
    torchIntType = torch.LongTensor
    #torchFloatType = torch.FloatTensor # single precision
    torchFloatType = torch.DoubleTensor # double precision
torch.set_default_tensor_type(torchFloatType)



class BaseNetwork():
    """Base network class with declaration of members and common code."""

    def __init__(self, stateDim, hiddenLayers):
        self.numNeurons = [stateDim] + hiddenLayers
        self.layersList = []
        # Internal layers with softsign activation function
        for i in range(len(self.numNeurons)-1):
            inputSize, outputSize = self.numNeurons[i], self.numNeurons[i+1]
            self.layersList.append(torch.nn.Linear(inputSize, outputSize))
            self.layersList.append(torch.nn.Softsign())

    def initNetwork(self):
        self.layers = torch.nn.Sequential(* self.layersList)
        #
        # Initialize weights
        for layer in self.layers[::2]:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, state):
        #
        # convert state to torch variable and compute NN's output:
        return self.layers( torch.tensor(state) )


class DiscreteQfunction(BaseNetwork):
    """Discrete Q function: gets state as input and outputs one Q for each action option."""

    def __init__(self, env, hiddenLayers = [32, 32]):
        super().__init__(env.stateSize, hiddenLayers)

        # Output layer with Softmax activation function
        self.layersList.append(torch.nn.Linear(self.numNeurons[-1], env.actionSize))
        self.initNetwork()
        self.layers[-1].bias.data.uniform_(10, 10)

    def getAction(self, state):
        #
        # Evaluate the probability of choosing each option
        Qvalues = self.forward(state)
        #
        # Convert the Q values to softmax policy:
        probabilities = torch.nn.functional.softmax(Qvalues**3, dim=0)
        #
        # Select an action based on the options probabilities
        probabilitiesNumpy = (probabilities.data).cpu().numpy()
        action = np.random.choice(len(probabilitiesNumpy), p=probabilitiesNumpy)
        #
        return action, Qvalues


class DiscretePolicy(BaseNetwork):
    """Discrete policy class: when the action is a choice between different options."""

    def __init__(self, env, hiddenLayers = [32, 32]):
        super().__init__(env.stateSize, hiddenLayers)

        # Output layer with Softmax activation function
        self.layersList.append(torch.nn.Linear(self.numNeurons[-1], env.actionSize))
        self.layersList.append(torch.nn.Softmax(dim=-1))
        self.initNetwork()

    def getAction(self, state):
        #
        # Evaluate the probability of choosing each option
        probabilities = self.forward(state)
        #
        # Convert the probabilities to numpy:
        probabilitiesNumpy = (probabilities.data).cpu().numpy()
        #
        # Select an action based on the options probabilities
        action = np.random.choice(len(probabilitiesNumpy), p=probabilitiesNumpy)
        #
        return action, probabilities[action]

    def computePolicyLoss(self, advantagesTorch, probabilitiesTorch, batchActions):
        #
        # Actions are used as indices, must be of integer type
        actionsTorch = torch.tensor(batchActions).type(torchIntType)
        #
        # Calculate actions log likelihood
        logLikelihood = torch.log(torch.stack(probabilitiesTorch))
        #
        # Multiply selected log likelihoods by A and return
        return - (advantagesTorch * logLikelihood).mean()


class GaussianPolicy(BaseNetwork):
    """Gaussian policy class: when the action(s) is(are) continuous."""

    def __init__(self, env, hiddenLayers = [32, 32], stdev = 0.2):
        self.actionDim = env.actionSize

        super().__init__(env.stateSize, hiddenLayers)

        # Select upper and lower scale for actions. Should be a vector, but
        # OpenAI gym never has action values that change per action-component.
        # Therefore keep it scalar which makes computePolicyLoss simpler.
        highVal, lowVal = env.env.action_space.high[0], env.env.action_space.low[0]
        self.stdev = stdev * (highVal - lowVal)/2

        # Output linear vector (mean) with same size as action dimensionality:
        self.layersList.extend([torch.nn.Linear(self.numNeurons[-1], self.actionDim)])
        self.initNetwork()

    def getAction(self, state):
        #
        # Evaluate the policy
        mean = self.forward(state)
        #
        # Convert to numpy
        mean_numpy = (mean.data).cpu().numpy()
        action = mean_numpy + self.stdev * np.random.randn(self.actionDim)
        return action, mean

    def computePolicyLoss(self, advantagesTorch, meansTorch, batchActions):
        #
        # Convert the actions to torch tensor
        actionTorch = torch.tensor(batchActions).type(torchFloatType)
        #
        # Calculate loss: - advantage(action) * log probability of action
        diffAct = (actionTorch - torch.stack(meansTorch)) / self.stdev
        return 0.5 * (advantagesTorch * torch.sum(diffAct**2, dim=1)).mean()
