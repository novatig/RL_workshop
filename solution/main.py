#!/usr/bin/env python3

import argparse
import numpy as np
from environment import Environment
import policies
import algorithms

if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description = "RL exercise.")
    parser.add_argument('-a', '--algorithm', default='Reinforce',
        choices=('Reinforce', 'PG', 'CMAES', 'ES', 'SARSA', 'MinVar', 'MonteCarlo', 'Qlearning'),
        help="Name of the algorithm used for training.")
    parser.add_argument('-e', '--environment', default='CartPole-v1',
        help="Name of the OpenAI gym environment to train on.")
    parser.add_argument('-H', '--hiddenLayers', nargs='+', type=int, default=[32],
        help="Size of the policy's hidden layers.")
    parser.add_argument('-z', '--explorationNoise', type=float, default=0.2,
        help="Standard deviation of policy for continuous actions.")
    parser.add_argument('-m', '--numEpisodesPerEval', type=int, default=100,
        help="Number of episodes per policy update iteration.")
    parser.add_argument('-n', '--numIterations', type=int, default=100,
        help="Number of policy updates.")
    parser.add_argument('-l', '--learningRate', type=float, default=0.1,
        help="Learning rate of policy update.")
    parser.add_argument('-g', '--gamma', type=float, default=0.99,
        help="Rewards discount factor.")
    parser.add_argument('-p', '--populationSize', type=int, default=64,
        help="Population size for CMAES.")
    parser.add_argument('-r', '--renderEvery', type=int, default=0,
        help="Render every nth episode. 0 to disable.")
    args = parser.parse_args()


    # Initialize the environment
    env = Environment(args.environment, args.numEpisodesPerEval, args.renderEvery)

    if args.algorithm == 'SARSA' or args.algorithm == 'Qlearning' or args.algorithm == 'MonteCarlo' or args.algorithm == 'MinVar':
        from value_iteration import ValueIteration
        policy = policies.DiscreteQfunction(env, args.hiddenLayers)
        algo = ValueIteration(policy,
                              gamma = args.gamma,
                              learnrate = args.learningRate,
                              estimator = args.algorithm)
    else : # policy based methods
        # Initialize the policy
        if env.actionType == 'discrete':
            policy = policies.DiscretePolicy(env,
                                             args.hiddenLayers)
        elif env.actionType == 'continuous':
            policy = policies.GaussianPolicy(env,
                                             args.hiddenLayers,
                                             args.explorationNoise)
        else:
            raise Exception("Unreachable.")

        # Select a training algorithm.
        if args.algorithm == 'Reinforce' or args.algorithm == 'PG':
            from reinforce import Reinforce
            algo = Reinforce(policy,
                             gamma = args.gamma,
                             learnrate = args.learningRate)
        elif args.algorithm == 'CMAES':
            from cmaes_korali import CMAESKorali
            algo = CMAESKorali(policy,
                               populationSize = args.populationSize,
                               sigma = args.explorationNoise)
        elif args.algorithm == 'ES':
            from evolution_strategies import EvolutionStrategies
            algo = EvolutionStrategies(policy,
                                       populationSize = args.populationSize,
                                       sigma = args.explorationNoise,
                                       learnRate = args.learningRate)

    # Train / update / improve the policy using the selected algorithm.
    algo.trainPolicy(policy, env, args.numIterations)
