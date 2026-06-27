import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

np.random.seed(42)

weights = np.random.rand(2, 1)
bias = np.zeros((1, 1))

hidden_weights = np.random.rand(2, 2)
hidden_bias = np.zeros((1, 2))

# bigger learning rate for faster convergence
# but can lead to overshooting the correct solution if too high
learning_rate = 1.0

epochs = 50000

for epoch in range(epochs):
    #-------forward pass-------
    hidden_z = X @ hidden_weights + hidden_bias
    hidden_predictions = sigmoid(hidden_z)
    
    z = hidden_predictions @ weights + bias
    predictions = sigmoid(z)
    
    #------loss-------
    error = predictions - y
    loss = np.mean(error ** 2)
    
    #------backward pass: gradients-------
    d_loss_predictions = 2 * error / len(y)
    d_predictions_z = sigmoid_derivative(z)
    
    d_loss_z = d_loss_predictions * d_predictions_z
    
    d_hidden_predictions = d_loss_z @ weights.T
    d_hidden_z = d_hidden_predictions * sigmoid_derivative(hidden_z)
    
    d_hidden_weights = X.T @ d_hidden_z
    d_hidden_bias = np.sum(d_hidden_z, axis=0, keepdims=True)
       
    d_weights = hidden_predictions.T @ d_loss_z
    d_bias = np.sum(d_loss_z)
    
    #-------parameters update-------
    hidden_weights -= learning_rate * d_hidden_weights
    hidden_bias -= learning_rate * d_hidden_bias
    
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
hidden_predictions = sigmoid(X @ hidden_weights + hidden_bias)
predictions = sigmoid(hidden_predictions @ weights + bias)

print(f"\nTrained weights: {weights.flatten()}")
print(f"Trained bias: {bias[0]}")
print (f"\nPredictions: \n{predictions}")

final_predictions = (predictions > 0.5).astype(int)
print(f"\nFinal Predictions: {final_predictions.flatten()}")