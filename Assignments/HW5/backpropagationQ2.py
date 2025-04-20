import numpy as np

# 1) Define the training set and targets
X_train = np.array([
    [1, 1, 0],   # d1 = 0
    [0, 1, 0],   # d2 = 0
    [0, 1, 1],   # d3 = 1
], dtype=float)
y_train = np.array([0, 0, 1], dtype=float)

# 2) Define the testing set (we'll predict these after training)
X_test = np.array([
    [1, 0, 0],   # Testing Data‑1
    [0, 0, 1],   # Testing Data‑2
], dtype=float)

# 3) Hyperparameters & initial weights
eta = 0.01
W = np.array([0.3, 0.1, 0.3], dtype=float)   # [W1, W2, W3]

def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

print("=== Backpropagation Demo (1 epoch) ===")
print("Initial weights:", W, "\n")

# 4) One epoch of SGD (pure online updates, no bias)
for i, (x, d) in enumerate(zip(X_train, y_train), start=1):
    # forward
    u = np.dot(W, x)
    y = sigmoid(u)
    # error + delta
    error = d - y
    delta = error * y * (1 - y)
    # weight update
    dW = eta * delta * x
    # print every step
    print(f"Sample {i}: x = {x}, target d = {d}")
    print(f"  u = W·x = {u:.4f}")
    print(f"  y = sigmoid(u) = {y:.4f}")
    print(f"  error = d – y = {error:.4f}")
    print(f"  δ = error * y*(1–y) = {delta:.4f}")
    print(f"  ΔW = η * δ * x = {dW}")
    # apply update
    W += dW
    print(f"  updated W = {W}\n")

print("Final weights after 1 epoch:", W, "\n")

# 5) Compute & print test outputs
print("=== Testing ===")
for i, xt in enumerate(X_test, start=1):
    ut = np.dot(W, xt)
    yt = sigmoid(ut)
    print(f"Test {i}: x = {xt}")
    print(f"  u = W·x = {ut:.4f}")
    print(f"  y = sigmoid(u) = {yt:.4f}\n")
