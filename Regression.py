import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(0)
points = 500
X = np.linspace(-3, 3, points)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)

model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=1))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1))

adam = Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=adam)
model.fit(X, y, epochs=50)

predictions = model.predict(X)
plt.scatter(X, y)
plt.plot(X, predictions, 'ro')
plt.show()