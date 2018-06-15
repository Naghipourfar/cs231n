import numpy as np

W1 = np.zeros([4, 3])
W2 = np.zeros([4, 4])
W3 = np.zeros([1, 4])
b1 = np.zeros([4, 1])
b2 = np.zeros([4, 1])
b3 = np.zeros([1, 1])

# forward-pass of a 3-layer neural network:
f = lambda x: 1.0 / (1.0 + np.exp(-x))  # activation function (use sigmoid)
x = np.random.randn(3, 1)  # random input vector of three numbers (3x1)
hidden_layer_1 = f(np.dot(W1, x) + b1)  # calculate first hidden layer activations (4x1)
hidden_layer_2 = f(np.dot(W2, hidden_layer_1) + b2)  # calculate second hidden layer activations (4x1)
output_layer = np.dot(W3, hidden_layer_2) + b3  # output neuron (1x1)
print(x)
print(f(x))
