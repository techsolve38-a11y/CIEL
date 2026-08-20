"""
The Single Neuron — Track B, item 5
=======================================

Goal: build a neuron with an activation function, and train it to do
binary classification (decide which of two classes a point belongs to)
— not just predict a number, but make a decision.

Why this matters for CIEL-0: every transformer layer is built from
neurons like this one, just with many inputs instead of one or two, and
stacked in the thousands. The math here doesn't change at scale — only
the size does.

A neuron = weighted sum of inputs + bias, then an ACTIVATION FUNCTION:
    z = w1*x1 + w2*x2 + b        <- the "weighted sum" (same as before,
                                     just with 2 inputs now instead of 1)
    a = activation(z)             <- the new part

Without the activation step, stacking neurons is mathematically pointless:
a linear function of a linear function is still just linear. The
activation is what lets a network represent curves, decision boundaries,
anything non-straight-line.
"""

import numpy as np

np.random.seed(42)

# --- Step 1: Generate 2D data belonging to two classes ---------------------
# Class 0: points clustered around (-2, -2)
# Class 1: points clustered around (2, 2)
n_per_class = 100

class0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-2, -2])
class1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([2, 2])

X = np.vstack([class0, class1])              # shape (200, 2) — 200 points, 2 features each
y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])  # 0 or 1 label per point

print(f"X shape: {X.shape}  (200 points, 2 features: x1, x2)")
print(f"y shape: {y.shape}  (200 labels: 0 or 1)")
print(f"First 3 points, class 0: {X[:3]}")
print(f"First 3 points, class 1: {X[100:103]}")


# --- Step 2: The neuron — weighted sum + sigmoid activation -----------------
def sigmoid(z):
    """Squashes any real number into the range (0, 1). We interpret the
    output as 'the neuron's estimated probability that this point is
    class 1'. As z -> +inf, sigmoid(z) -> 1. As z -> -inf, sigmoid(z) -> 0.
    At z=0, sigmoid(z)=0.5 (maximally uncertain)."""
    return 1 / (1 + np.exp(-z))


def neuron_forward(X, w, b):
    """X has shape (n_points, 2). w has shape (2,) — one weight per
    input feature. This computes, for every point at once (vectorized):
        z_i = w1*x1_i + w2*x2_i + b
        a_i = sigmoid(z_i)
    """
    z = X @ w + b        # (n_points, 2) @ (2,) -> (n_points,) — a weighted sum per point
    a = sigmoid(z)
    return a


# Sanity check with random weights: predictions should be scattered
# around 0.5, since a random line through 2D space isn't going to
# separate our two clusters correctly yet.
w_init = np.random.randn(2) * 0.1
b_init = 0.0
preds = neuron_forward(X, w_init, b_init)
print(f"\nPredictions with random weights (should hover near 0.5): "
      f"min={preds.min():.3f}, max={preds.max():.3f}, mean={preds.mean():.3f}")


# --- Step 3: Binary cross-entropy loss --------------------------------------
# Why not reuse MSE from last time? MSE punishes a confident WRONG answer
# (predicting 0.99 when the true label is 0) only a little worse than a
# mildly wrong one (predicting 0.6 when the true label is 0) — the penalty
# grows quadratically, which is too forgiving for classification.
#
# Cross-entropy punishes confident wrong answers MUCH more harshly — it
# grows toward infinity as a confident prediction approaches "completely
# wrong." That's the right incentive: a model that's 99% sure and wrong
# should be penalized far more than one that was uncertain and wrong.
#
# Formula, per point:  loss_i = -[ y_i * log(a_i) + (1-y_i) * log(1-a_i) ]
#   - if y_i = 1: loss_i = -log(a_i)      -> punishes low a_i (predicting
#                                             "probably class 0" when truth is 1)
#   - if y_i = 0: loss_i = -log(1-a_i)    -> punishes high a_i
def binary_cross_entropy(a, y, eps=1e-9):
    a = np.clip(a, eps, 1 - eps)   # avoid log(0), which is -infinity
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))


bad_loss = binary_cross_entropy(preds, y)
print(f"Loss with random (bad) weights: {bad_loss:.3f}  (should be near -log(0.5)={-np.log(0.5):.3f})")


# --- Step 4: Gradients ------------------------------------------------------
# This is where sigmoid + cross-entropy earns its keep as a PAIR. If you
# work through the chain rule for d(Loss)/d(z) — where z is the
# pre-activation weighted sum — the sigmoid's own derivative and the
# cross-entropy's own derivative CANCEL most of their complexity, leaving:
#
#     d(Loss)/d(z_i) = a_i - y_i
#
# That's it. "Prediction minus truth" — the exact same shape of error
# signal as in the linear regression gradient last time. This is not a
# coincidence; it's *why* sigmoid+cross-entropy is the standard pairing
# for binary classification instead of, say, sigmoid+MSE.
#
# From there, same chain rule as before to get to the actual weights:
#     d(Loss)/d(w_j) = (1/n) * sum( (a_i - y_i) * x_ij )
#     d(Loss)/d(b)   = (1/n) * sum( a_i - y_i )
def compute_gradients(X, y, w, b):
    n = len(y)
    a = neuron_forward(X, w, b)
    error = a - y                          # shape (n,) — this is (a_i - y_i) for every point
    grad_w = (1 / n) * (X.T @ error)        # shape (2,) — one gradient per weight
    grad_b = (1 / n) * np.sum(error)
    return grad_w, grad_b


# --- Step 5: Train ----------------------------------------------------------
w = np.random.randn(2) * 0.1
b = 0.0
learning_rate = 0.5
n_steps = 1000

print("\n--- Training ---")
for step in range(n_steps):
    grad_w, grad_b = compute_gradients(X, y, w, b)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    if step % 100 == 0 or step == n_steps - 1:
        loss = binary_cross_entropy(neuron_forward(X, w, b), y)
        preds_binary = (neuron_forward(X, w, b) >= 0.5).astype(int)
        accuracy = np.mean(preds_binary == y)
        print(f"step {step:4d} | loss={loss:.4f} | accuracy={accuracy:.3f}")

print(f"\nFinal weights: w={w}, b={b:.4f}")
