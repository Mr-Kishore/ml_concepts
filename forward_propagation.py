import numpy as np

# Define the activation function (Sigmoid in this case)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the forward propagation function
def forward_propagation(X, weights, biases):
    # Input Layer (Layer 1)
    Z1 = np.dot(weights['W1'], X) + biases['b1']
    A1 = sigmoid(Z1)
    
    # Hidden Layer (Layer 2)
    Z2 = np.dot(weights['W2'], A1) + biases['b2']
    A2 = sigmoid(Z2)
    
    # Output Layer (Layer 3)
    Z3 = np.dot(weights['W3'], A2) + biases['b3']
    A3 = sigmoid(Z3)
    
    return A3

# Sample weights and biases for a 3-layer network
weights = {
    'W1': np.array([[0.2, 0.4], [0.3, 0.5]]), # 2x2 matrix for Layer 1 weights
    'W2': np.array([[0.1, 0.6], [0.7, 0.8]]), # 2x2 matrix for Layer 2 weights
    'W3': np.array([[0.5, 0.9]])              # 1x2 matrix for Layer 3 weights
}

biases = {
    'b1': np.array([[0.1], [0.2]]),  # 2x1 matrix for Layer 1 biases
    'b2': np.array([[0.3], [0.4]]),  # 2x1 matrix for Layer 2 biases
    'b3': np.array([[0.5]])          # 1x1 matrix for Layer 3 bias
}

# Input features (2 features, 1 example)
X = np.array([[0.5], [0.9]])

# Perform forward propagation
output = forward_propagation(X, weights, biases)
print("Network Output:", output)
