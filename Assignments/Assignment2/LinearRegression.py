import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------------
# Given dataset from Problem-1:
#   X1  X2   y
#   2   1    5
#   4   3    7
#   6   5    8
X1 = np.array([2, 4, 6], dtype=float)
X2 = np.array([1, 3, 5], dtype=float)
y  = np.array([5, 7, 8], dtype=float)

# Number of samples
N = len(y)

# X to include a column of ones (for θ0).
# Shape: X -> (N, 3), columns = [1, X1, X2]
X_0 = np.ones_like(X1)
X = np.column_stack((X_0, X1, X2))

# -------------------------------------------------------
# -------------------------------------------------------
alpha = 0.01          # learning rate
iterations = 20       # number of iterations (varies)
theta = np.array([1.5, 0.3, 0.7], dtype=float)  # initial weights: [θ0, θ1, θ2]

# -------------------------------------------------------
# Gradient Descent Loop
# -------------------------------------------------------
for i in range(iterations):
    #  Compute predictions y_pred
    y_pred = X.dot(theta) 

    # Compute error vector
    errors = y_pred - y    

    # Compute gradient of MSE
    #     grad = (2/N) * X^T * (errors)
    grad = (2 / N) * X.T.dot(errors)

    # Update weights
    theta = theta - alpha * grad

    #  Optionally print loss every few iterations
    if (i+1) % 5 == 0:
        mse = np.mean(errors**2)
        print(f"Iteration {i+1}: MSE = {mse:.4f}, theta = {theta}")

# Final parameters after gradient descent
print("\nFinal weights (theta):", theta)

# -------------------------------------------------------
# Visualization
# -------------------------------------------------------
# a 3D plot as we have two features:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original data points
ax.scatter(X1, X2, y, color='red', label='Data Points')

# meshgrid to show the regression plane
x1_range = np.linspace(min(X1), max(X1), 20)
x2_range = np.linspace(min(X2), max(X2), 20)
X1g, X2g = np.meshgrid(x1_range, x2_range)

# Predicted z-values (y-values) over the grid
Z = theta[0] + theta[1]*X1g + theta[2]*X2g

# Plot the regression plane
ax.plot_surface(X1g, X2g, Z, alpha=0.5, color='blue', edgecolor='none')

# Labels and legend
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title('Linear Regression with Two Features')
plt.legend()
plt.show()
