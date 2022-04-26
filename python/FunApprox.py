# example of fitting a neural net on x vs x^2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from numpy import asarray
from matplotlib import pyplot
import numpy as np

# define the dataset
x1 = asarray([i for i in range(-10,11)])
x2 = asarray([i for i in range(-10,11)])

# create a non-linear function
X1 = np.array([])
X2 = np.array([])
Y = np.array([])
for i in x1:
	for j in x2:
		y = asarray([i**2 -j**2])
		X1 = np.append(X1, i)
		X2 = np.append(X2, j)
		Y = np.append(Y, y)

# prepare data
Y = np.transpose(Y)
X1 = X1.reshape((len(X1), 1))
X2 = X2.reshape((len(X2), 1))
X = np.append(X1,X2,1)

# design the neural network model
model = Sequential()
model.add(Dense(10, input_dim=2, activation = LeakyReLU(alpha=0.01), kernel_initializer='he_uniform'))
model.add(Dense(50, activation = LeakyReLU(alpha=0.01), kernel_initializer='he_uniform'))
model.add(Dense(50, activation = LeakyReLU(alpha=0.01), kernel_initializer='he_uniform'))
model.add(Dense(1))

model.compile(loss='mse', optimizer= Adam(learning_rate=1e-3))
model.fit(X, Y, epochs=100, batch_size=1, verbose = 1)
Yhat = model.predict(X)

fig = pyplot.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X1, X2, Y, label='Actual')
ax.scatter(X1, X2, Yhat, label='Predicted')
pyplot.title('Input (x) versus Output (y)')
pyplot.xlabel('Input Variable (x1)')
pyplot.ylabel('Input Variable (x2)')
pyplot.ylabel('Output Variable (y)')


pyplot.legend()
pyplot.show()