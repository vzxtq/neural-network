import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [0],
    [0],
    [1]
])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

np.random.seed(42)

weights = np.random.rand(2, 1)
bias = np.random.rand(1)

learning_rate = 0.9  

# bigger learning rate for faster convergence
# but can lead to overshooting the correct solution if too high
epochs = 10000

for epoch in range(epochs):
    z = X @ weights + bias
    predictions = sigmoid(z)
    
    error = predictions - y
    loss = np.mean(error ** 2)
    
    d_loss_predictions = 2 * error / len(y)
    d_predictions_z = sigmoid_derivative(z)
    
    d_loss_z = d_loss_predictions * d_predictions_z
    
    d_weights = X.T @ d_loss_z
    d_bias = np.sum(d_loss_z)
    
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
print(f"\nTrained weights: {weights.flatten()}")
print(f"Trained bias: {bias[0]}")
print (f"\nPredictions: {sigmoid(X @ weights + bias)}")

final_predictions = (sigmoid(X @ weights + bias) > 0.5).astype(int)
print(f"\nFinal Predictions: {final_predictions.flatten()}")