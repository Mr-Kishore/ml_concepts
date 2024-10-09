import random
import math

# Helper Functions for Matrix Operations
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_multiply(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def matrix_add(A, B):
    if len(A) == len(B) and len(A[0]) == len(B[0]):
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    else:
        raise ValueError("Matrix dimensions must agree for addition.")

def scalar_multiply(matrix, scalar):
    return [[matrix[i][j] * scalar for j in range(len(matrix[0]))] for i in range(len(matrix))]

# Activation Functions
def ReLU(Z):
    return [[max(0, z) for z in row] for row in Z]

def softmax(Z):
    exp_Z = [[math.exp(z) for z in row] for row in Z]
    sum_exp = [sum(row) for row in exp_Z]
    return [[exp_Z[i][j] / sum_exp[i] for j in range(len(Z[0]))] for i in range(len(Z))]

# ReLU Derivative
def ReLU_deriv(Z):
    return [[1 if z > 0 else 0 for z in row] for row in Z]

# One-Hot Encoding
def one_hot(Y, num_classes):
    one_hot_Y = [[0] * num_classes for _ in range(len(Y))]
    for i, val in enumerate(Y):
        one_hot_Y[i][val] = 1
    return one_hot_Y

# Initialize Parameters
def init_params(input_size, hidden_size, output_size):
    W1 = [[random.random() - 0.5 for _ in range(input_size)] for _ in range(hidden_size)]
    b1 = [[random.random() - 0.5] for _ in range(hidden_size)]
    W2 = [[random.random() - 0.5 for _ in range(hidden_size)] for _ in range(output_size)]
    b2 = [[random.random() - 0.5] for _ in range(output_size)]
    return W1, b1, W2, b2

# Forward Propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = matrix_add(matrix_multiply(W1, X), b1)
    A1 = ReLU(Z1)
    Z2 = matrix_add(matrix_multiply(W2, A1), b2)
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Backward Propagation
def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = len(Y[0])
    dZ2 = [[A2[i][j] - Y[i][j] for j in range(m)] for i in range(len(A2))]
    dW2 = scalar_multiply(matrix_multiply(dZ2, transpose(A1)), 1/m)
    db2 = scalar_multiply([[sum(row)] for row in dZ2], 1/m)
    dZ1 = matrix_multiply(transpose(W2), dZ2)
    dZ1 = [[dZ1[i][j] * ReLU_deriv(Z1)[i][j] for j in range(m)] for i in range(len(dZ1))]
    dW1 = scalar_multiply(matrix_multiply(dZ1, transpose(X)), 1/m)
    db1 = scalar_multiply([[sum(row)] for row in dZ1], 1/m)
    return dW1, db1, dW2, db2

# Update Parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = matrix_add(W1, scalar_multiply(dW1, -alpha))
    b1 = matrix_add(b1, scalar_multiply(db1, -alpha))
    W2 = matrix_add(W2, scalar_multiply(dW2, -alpha))
    b2 = matrix_add(b2, scalar_multiply(db2, -alpha))
    return W1, b1, W2, b2

# Prediction and Accuracy
def get_predictions(A2):
    return [row.index(max(row)) for row in A2]

def get_accuracy(predictions, Y):
    return sum([1 if predictions[i] == Y[i] else 0 for i in range(len(Y))]) / len(Y)

# Gradient Descent
def gradient_descent(X, Y, alpha, iterations, input_size, hidden_size, output_size):
    W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print(f"Iteration {i} - Accuracy: {get_accuracy(predictions, [row.index(1) for row in Y])}")
    return W1, b1, W2, b2

# Example parameters
input_size = 784
hidden_size = 10
output_size = 10
alpha = 0.13
iterations = 100

# Dummy input data (28x28 images flattened)
X_train = [[random.random() for _ in range(input_size)] for _ in range(1000)]
Y_train = one_hot([random.randint(0, 9) for _ in range(1000)], 10)

# Run the model
W1, b1, W2, b2 = gradient_descent(transpose(X_train), transpose(Y_train), alpha, iterations, input_size, hidden_size, output_size)
