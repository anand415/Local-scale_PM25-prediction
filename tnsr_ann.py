# Import the kera modules
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor. Since the input only has one column
inputs = Input(shape=(4,))

# a layer instance is callable on a tensor, and returns a tensor
# To the first layer we are feeding inputs
x = Dense(32, activation='relu')(inputs)
# To the next layer we are feeding the result of previous call here it is h
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# Predictions are the result of the neural network. Notice that the predictions are also having one column.
predictions = Dense(1)(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
# Here the loss function is mse - Mean Squared Error because it is a regression problem.
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mse'])