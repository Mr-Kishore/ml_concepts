import math

# Implementing Tanh function
def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

# Test inputs to see the Tanh function in action
inputs = [-2, -1, 0, 1, 2]

print("Tanh Function Outputs for the given inputs:")
for x in inputs:
    output = tanh(x)
    print(f"Tanh({x}) = {output:.4f}")