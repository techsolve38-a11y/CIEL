"""
The Multi-Layer Perceptron & Backpropagation — Track B, item 6
===================================================================

Goal: build a network with a HIDDEN LAYER, and train it with proper
backpropagation — the chain rule applied through more than one layer.

Why we need more than one neuron/layer: a single neuron computes
sigmoid(w1*x1 + w2*x2 + b) — geometrically, this can only draw ONE
STRAIGHT LINE (or straight hyperplane) to separate classes. Some
patterns are not linearly separable, no matter how you set the weights.
The classic example is XOR: four points arranged so that no single line
divides the two classes correctly.

    (0,1) = class 1        (1,1) = class 0
    (0,0) = class 0        (1,0) = class 1

Try to draw ONE straight line separating {class0} from {class1} above —
you can't. A hidden layer fixes this by letting the network first bend
space (each hidden neuron draws its own line), then the output layer
draws a line THROUGH THE BENT SPACE, which lets the combined effect be a
curve in the original space.
"""

import numpy as np

np.random.seed(42)

# --- Step 1: XOR-pattern data, with a little noise to make it realistic ----
n_per_cluster = 50


def make_cluster(center, n=n_per_cluster, spread=0.15):
    return np.random.randn(n, 2) * spread + np.array(center)


X = np.vstack([
    make_cluster([0, 0]),   # class 0
    make_cluster([1, 1]),   # class 0  (same class as (0,0) — this is the XOR pattern)
    make_cluster([0, 1]),   # class 1
    make_cluster([1, 0]),   # class 1
])
y = np.concatenate([
    np.zeros(n_per_cluster), np.zeros(n_per_cluster),
    np.ones(n_per_cluster), np.ones(n_per_cluster),
])

print(f"X shape: {X.shape}, y shape: {y.shape}")
print("This is the XOR pattern: (0,0) and (1,1) are class 0; "
      "(0,1) and (1,0) are class 1.")
print("No single straight line can separate these two classes.")


# --- Step 2: PROVE a single neuron fails here (reusing last session's code) -
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def single_neuron_forward(X, w, b):
    return sigmoid(X @ w + b)


def binary_cross_entropy(a, y, eps=1e-9):
    a = np.clip(a, eps, 1 - eps)
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))


w = np.random.randn(2) * 0.1
b = 0.0
learning_rate = 0.5
for step in range(2000):
    a = single_neuron_forward(X, w, b)
    error = a - y
    grad_w = (1 / len(y)) * (X.T @ error)
    grad_b = (1 / len(y)) * np.sum(error)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

final_preds = (single_neuron_forward(X, w, b) >= 0.5).astype(int)
final_accuracy = np.mean(final_preds == y)
print(f"\nSingle neuron, fully trained (2000 steps): accuracy = {final_accuracy:.3f}")
print("This should plateau far below 1.0 — no amount of training fixes it, "
      "because the problem isn't training, it's that ONE straight line "
      "structurally cannot separate this pattern.")


# --- Step 3: The fix — a hidden layer ---------------------------------------
# Architecture: 2 inputs -> 4 hidden neurons -> 1 output neuron
#
#   Layer 1 (hidden): z1 = X @ W1 + b1     shape: (n, 4)
#                      a1 = sigmoid(z1)     shape: (n, 4)  <- 4 "bent" features
#   Layer 2 (output):  z2 = a1 @ W2 + b2    shape: (n, 1)
#                      a2 = sigmoid(z2)     shape: (n, 1)  <- final prediction
#
# Each of the 4 hidden neurons learns its OWN straight-line boundary in the
# original 2D space. The output neuron then draws a straight line THROUGH
# THE OUTPUTS of those 4 neurons — which, translated back into the original
# space, can look like a curve, because the hidden layer already bent space
# before the output layer got to it.
n_hidden = 4
n_inputs = 2

# Small random initialization — NOT zero. If all weights start at zero,
# every hidden neuron computes the exact same thing and gets the exact
# same gradient update forever, so they never differentiate from each
# other. Random initialization breaks that symmetry.
W1 = np.random.randn(n_inputs, n_hidden) * 0.5
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, 1) * 0.5
b2 = np.zeros(1)


def mlp_forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2   # returning intermediate values — backprop needs them


z1, a1, z2, a2 = mlp_forward(X, W1, b1, W2, b2)
print(f"\nShapes check — z1: {z1.shape}, a1: {a1.shape}, z2: {z2.shape}, a2: {a2.shape}")
print(f"Untrained predictions (should hover near 0.5): mean={a2.mean():.3f}")


# --- Step 4: Backpropagation ------------------------------------------------
# This is the SAME chain rule from the single neuron, applied TWICE —
# once at the output layer, once more at the hidden layer. "Back"-prop
# because the error signal flows backward, layer by layer, from output
# toward input.
#
# OUTPUT LAYER (identical to the single neuron case):
#   dL/dz2 = a2 - y                              <- clean, as before
#   dL/dW2 = a1.T @ dz2 / n                       <- note: a1, not X —
#                                                    W2's inputs are the
#                                                    HIDDEN layer's outputs
#   dL/db2 = mean(dz2)
#
# HIDDEN LAYER (the new part):
#   To know how much each HIDDEN neuron contributed to the final error,
#   we need to push the error signal backward through W2:
#     dL/da1 = dz2 @ W2.T                         <- "how much did each
#                                                     hidden neuron's
#                                                     output affect the
#                                                     final loss?"
#   Then through that hidden neuron's OWN sigmoid activation:
#     dL/dz1 = dL/da1 * a1 * (1 - a1)             <- a1*(1-a1) is the
#                                                     derivative of sigmoid
#                                                     itself, evaluated at
#                                                     this neuron's output
#   Then finally to the hidden layer's weights:
#     dL/dW1 = X.T @ dz1 / n
#     dL/db1 = mean(dz1, axis=0)
def backward(X, y, z1, a1, z2, a2, W2):
    n = len(y)
    y = y.reshape(-1, 1)   # match a2's shape (n, 1)

    dz2 = a2 - y                          # (n, 1) — output layer error signal
    dW2 = a1.T @ dz2 / n                  # (4, 1)
    db2 = np.mean(dz2, axis=0)            # (1,)

    da1 = dz2 @ W2.T                      # (n, 4) — push error back through W2
    dz1 = da1 * a1 * (1 - a1)             # (n, 4) — push through hidden sigmoid
    dW1 = X.T @ dz1 / n                   # (2, 4)
    db1 = np.mean(dz1, axis=0)            # (4,)

    return dW1, db1, dW2, db2


# --- Step 5: Train -----------------------------------------------------------
W1 = np.random.randn(n_inputs, n_hidden) * 0.5
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, 1) * 0.5
b2 = np.zeros(1)

learning_rate = 1.0
n_steps = 3000

print("\n--- Training MLP ---")
for step in range(n_steps):
    z1, a1, z2, a2 = mlp_forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(X, y, z1, a1, z2, a2, W2)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if step % 300 == 0 or step == n_steps - 1:
        loss = binary_cross_entropy(a2.flatten(), y)
        preds = (a2.flatten() >= 0.5).astype(int)
        accuracy = np.mean(preds == y)
        print(f"step {step:4d} | loss={loss:.4f} | accuracy={accuracy:.3f}")
