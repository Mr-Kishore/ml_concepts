import math

#defining sigmoid function

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# defining derivative

def sigmoid_derivative(x):
    return x * (1 - x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Test the function with some inputs
inputs = [-3, -1, 0, 1, 3]
print("Sigmoid Function Outputs:")
for x in inputs:
    print(f"sigmoid({x}) = {sigmoid(x)}")