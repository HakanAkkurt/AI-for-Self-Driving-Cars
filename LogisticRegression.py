import numpy as np
import matplotlib.pyplot as plt

nPts = 100
np.random.seed(0)
topRegion = np.array([np.random.normal(10, 2, nPts), np.random.normal(12, 2, nPts)]).T
bottomRegion = np.array([np.random.normal(5, 2, nPts), np.random.normal(6, 2, nPts)]).T
_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(topRegion[:, 0], topRegion[:, 1], color='r')
ax.scatter(bottomRegion[:, 0], topRegion[:, 1], color='b')
plt.show()