import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    ln = plt.plot(x1, x2, "-")
    plt.pause(0.0001)
    ln[0].remove()

def sigmoid(score):
    return 1 / (1 + np.exp(-score))

def calculateError(lineParameters, points, y):
    m = points.shape[0]
    p = sigmoid(points * lineParameters)
    crossEntropy = -(1 / m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return crossEntropy

def gradientDescent(lineParameters, points, y, alpha):
    m = points.shape[0]
    for i in range(2000):
        p = sigmoid(points * lineParameters)
        gradient = (points.T * (p - y)) * (alpha / m)
        lineParameters = lineParameters - gradient
        w1 = lineParameters.item(0)
        w2 = lineParameters.item(1)
        b = lineParameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + x1 * (-w1 / w2)
        draw(x1, x2)
        # print(calculateError(lineParameters, points, y))

nPts = 100
np.random.seed(0)
bias = np.ones(nPts)
topRegion = np.array([np.random.normal(10, 2, nPts), np.random.normal(12, 2, nPts), bias]).T
bottomRegion = np.array([np.random.normal(5, 2, nPts), np.random.normal(6, 2, nPts), bias]).T
allPoints = np.vstack((topRegion, bottomRegion))

lineParameters = np.matrix([np.zeros(3)]).T

linearCombination = allPoints * lineParameters
probabilities = sigmoid(linearCombination)
y = np.array([np.zeros(nPts), np.ones(nPts)]).reshape(nPts * 2, 1)

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(topRegion[:, 0], topRegion[:, 1], color="r")
ax.scatter(bottomRegion[:, 0], bottomRegion[:, 1], color="b")
gradientDescent(lineParameters, allPoints, y, 0.06)

plt.show()
