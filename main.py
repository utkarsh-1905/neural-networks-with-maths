import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import os

from utils.activation import Activation_ReLU
from utils.loss import Activation_Softmax_Loss_CategoricalCrossentropy
from utils.optimizers import Optimizer_Adam
from utils.layer import Layer_Dense

nnfs.init()

# optimizer = Optimizer_SGD(decay=1e-4, momentum=0.8)
# optimizer = Optimizer_Adagrad(decay=1e-5)
optimizer = Optimizer_Adam(decay=5e-7, learning_rate=0.05)

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 50)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(50, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

acc_list = []
loss_list = []
lr_list = []

for i in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    # printing info

    acc_list.append(accuracy)
    loss_list.append(loss)
    lr_list.append(optimizer.current_lr)
    if not i % 100:
        print(f"Epoch: {i}\t"+f"Accuracy: {accuracy*100:.2f}%\t" +
              f"Loss: {loss:.3f}\t"+f"lr: {optimizer.current_lr:.6f}")

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Optimizer
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# plt.plot(acc_list, 'g', linewidth=0.2)
# plt.plot(loss_list, 'r', linewidth=0.2)
# plt.plot(lr_list, 'b', linewidth=0.1)
# plt.show()
# os._exit(0)

# Validation

x_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(x_test)
activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
