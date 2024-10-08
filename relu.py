# Implementing  ReLU function
def relu(x):
    return max(0, x)

# Test inputs
inputs = [-5, -2, 0, 2, 5, 10, -10]

print("ReLU Function Outputs for the given inputs:")
for x in inputs:
    output = relu(x)
    print(f"ReLU({x}) = {output}")
