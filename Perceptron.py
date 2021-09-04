import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

nPts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, nPts),
               np.random.normal(12, 2, nPts)]).T
Xb = np.array([np.random.normal(8, 2, nPts),
               np.random.normal(6, 2, nPts)]).T
 
X = np.vstack((Xa, Xb))
Y = np.matrix(np.append(np.zeros(nPts), np.ones(nPts))).T

model = Sequential()
model.add(Dense(units = 1, input_shape = (2,), activation='sigmoid'))
adam = Adam(learning_rate = 0.1)
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
h = model.fit(x = X, y = Y, verbose = 1, batch_size = 50, epochs = 500, shuffle = 'true')

# plt.plot(h.history['loss'])
# plt.title('loss')
# plt.xlabel('epoch')
# plt.legend(['loss'])
# plt.show()

def plotDecisionBoundry(X, Y, model):
    xSpan = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
    ySpan = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1)
    xx, yy = np.meshgrid(xSpan, ySpan)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    predFunc = model.predict(grid)
    z = predFunc.reshape(xx.shape)
    plt.contourf(xx, yy, z)

plotDecisionBoundry(X, Y, model)
plt.scatter(X[:nPts,0], X[:nPts,1])
plt.scatter(X[nPts:,0], X[nPts:,1])

x = 7.5
y = 5

point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
plt.show()
print("Prediction is: ", prediction)