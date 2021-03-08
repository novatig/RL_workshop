import matplotlib.pyplot as plt, numpy as np
nItersPerRun = 50
colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']

def getAverageRewards(run):
  data = np.fromfile(run + '.txt', sep=' ')
  nRuns = data.size // nItersPerRun
  # cut out non-complete training runs:
  data = data[:nRuns*nItersPerRun]
  data = data.reshape(nRuns, nItersPerRun)
  return data

def getAverageRewards2(run):
  data = np.fromfile(run + '.txt', sep=' ')
  return data

algos = ['SARSA', 'MonteCarlo', 'Qlearning']
for i in range(len(algos)):
  algo = algos[i]
  data = getAverageRewards(algo)
  meanR = np.mean(data, axis=0)
  stdevR = np.std(data, axis=0)
  #rTop, rBot = meanR - stdevR, meanR + stdevR
  rBot = np.percentile(data, 20, axis=0)
  rTop = np.percentile(data, 80, axis=0)
  X = np.arange(data.shape[1])
  plt.fill_between(X, rBot, rTop, facecolor=colors[i], alpha=.5)
  plt.plot(X, meanR, color=colors[i], label=algos[i])

algos = ['Reinforce', 'Evolutionary Strategies', 'CMAES']
for i in range(len(algos)):
  algo = algos[i]
  data = getAverageRewards2(algo)
  X = np.arange(data.size)
  plt.plot(X, data, color=colors[i+3], label=algos[i])

plt.xlabel('iteration')
plt.ylabel('average returns')
plt.legend()
plt.show()