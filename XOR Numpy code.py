import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define XOR training data
# One node for bias value
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

print(f"X : {X.shape}")
print(f"Y : {y.shape}")

m = np.maximum(*y.shape)
print (f"m : {m}")

# Define hyperparameters
input_neurons = 2
hidden_neurons = 3
output_neurons = 1
learning_rate = 0.1
epochs = 10000

# Cost graph
Cost = []

# Initialize weights randomly
hidden_weights = np.random.uniform(size=(input_neurons+1, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons+1, output_neurons))

print(f"hidden_weights : {hidden_weights.shape}  \n {hidden_weights} \n")
print(f"hidden_weights Transpose : {hidden_weights.T.shape}  \n {hidden_weights.T} \n")

print(f"output_weights : {output_weights.shape}  \n {output_weights} \n")
print(f"output_weights Transpose : {output_weights.T.shape}  \n {output_weights.T} \n")


# Train the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)
    hidden_layer_output = np.concatenate((hidden_layer_output, np.ones((len(X), 1))), axis=1)
    
    output_layer_input = np.dot(hidden_layer_output, output_weights)
    output_layer_output = sigmoid(output_layer_input)
    
    # Calculate error
    output_error = y - output_layer_output
    
    # Cost

    loss =  1/X.shape[1] * np.sum(-1 * (y * np.log(output_layer_output) + (1-y) * np.log(1-output_layer_output)))
    Cost.append(loss)
    
    # Backpropagation
    output_delta = output_error * sigmoid_derivative(output_layer_output)
    hidden_error = np.dot(output_delta, output_weights[:-1].T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output[:,:-1])
    
    # Update weights
    output_weights += learning_rate * np.dot(hidden_layer_output.T, output_delta)
    hidden_weights += learning_rate * np.dot(X.T, hidden_delta)
    

print(f"Final Hidden Weight = {hidden_weights}" )
print(f"Final Output Weight = {output_weights}" )

print(f"\n\n Cost shape and costMatrix : {len(Cost)}\n\n")
plt.plot(Cost, color='blue' )
plt.ylabel("Cost")
plt.xlabel("epochs")
plt.show()
print(f"\n\n\n")


# Test the neural network 
# Forward propagation steps
test_inputs = np.arange(-0.8, 1.2, 0.05)
test_outputs = []
for i in test_inputs:
    for j in test_inputs:
        test_input = np.array([[i, j, 1]])
        hidden_layer_input = np.dot(test_input, hidden_weights)
        hidden_layer_output = sigmoid(hidden_layer_input)
        hidden_layer_output = np.concatenate((hidden_layer_output, np.ones((1, 1))), axis=1)
        output_layer_input = np.dot(hidden_layer_output, output_weights)
        # y^ Sigmoid to shrinl value in between 0 and 1
        output_layer_output = sigmoid(output_layer_input)
        test_outputs.append(output_layer_output[0])
        
# Plot the output in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-0.8, 1.2, 0.05)
y = np.arange(-0.8, 1.2, 0.05)
X, Y = np.meshgrid(x, y)
Z = np.array(test_outputs).reshape(X.shape)
print(f"Z : {Z}")

ax.plot_surface(X, Y, Z,  cmap='coolwarm')

ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')
print(f"Axes 3D graph \n\n")
plt.show()