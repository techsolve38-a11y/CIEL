"""
Gradient Descent From Scratch — Track A checkpoint
=====================================================

Goal: given noisy (x, y) points that roughly follow a line
y = true_w * x + true_b, RECOVER w and b using nothing but numpy —
no sklearn, no autograd. Every gradient is one we derive and write
by hand.

Why this matters for CIEL-0 later: a neural network training step is
the exact same four-part loop you'll see below (guess -> measure error
-> compute gradient -> nudge parameters), just with millions of
parameters instead of two.
"""

import numpy as np

# Fix the random seed so results are reproducible while we're learning.
np.random.seed(42)

# --- Step 1: Generate data with a KNOWN underlying relationship -----------
# We pretend we don't know these, and our job is to recover them from data.
true_w = 3.2
true_b = -1.5

n_points = 200
x = np.random.uniform(-5, 5, size=n_points)          # random x values
noise = np.random.normal(0, 1.5, size=n_points)        # gaussian noise
y = true_w * x + true_b + noise                          # noisy observations

print(f"True relationship: y = {true_w} * x + {true_b}")
print(f"First 5 data points (x, y):")
for i in range(5):
    print(f"  ({x[i]:.3f}, {y[i]:.3f})")


# --- Step 2: The model — our (initially wrong) guess -----------------------
def predict(x, w, b):
    """The model's guess: a straight line. This is the entire hypothesis
    space — we're only allowed to represent linear relationships."""
    return w * x + b


# --- Step 3: The loss — a single number measuring how wrong we are --------
def mse_loss(y_pred, y_true):
    """Mean Squared Error. Squaring does two jobs at once:
    1) makes errors positive regardless of direction (too-high vs too-low
       both count as 'bad'),
    2) punishes large errors disproportionately more than small ones,
       which is exactly the behavior we want during training."""
    return np.mean((y_pred - y_true) ** 2)


# Sanity check: start with a deliberately bad guess and confirm the loss
# is large; a good guess should have small loss.
w_bad, b_bad = 0.0, 0.0
bad_loss = mse_loss(predict(x, w_bad, b_bad), y)
print(f"\nLoss with a bad guess (w=0, b=0): {bad_loss:.3f}")

good_loss = mse_loss(predict(x, true_w, true_b), y)
print(f"Loss with the TRUE parameters (w={true_w}, b={true_b}): {good_loss:.3f}")


# --- Step 4: Gradients — derived by hand via the chain rule -----------------
# Loss(w, b) = (1/n) * sum( (w*x_i + b - y_i)^2 )
#
# Let e_i = (w*x_i + b - y_i)   <- the "error" (prediction minus truth) for point i
# Loss = (1/n) * sum(e_i^2)
#
# d(Loss)/d(w):
#   By the chain rule: d(e_i^2)/d(w) = 2*e_i * d(e_i)/d(w)
#   and d(e_i)/d(w) = x_i   (because e_i = w*x_i + b - y_i, and only the
#                             w*x_i term depends on w)
#   => d(Loss)/d(w) = (1/n) * sum( 2*e_i * x_i ) = (2/n) * sum(e_i * x_i)
#
# d(Loss)/d(b):
#   d(e_i)/d(b) = 1   (b appears alone, unscaled, in e_i)
#   => d(Loss)/d(b) = (2/n) * sum(e_i)
def compute_gradients(x, y, w, b):
    n = len(x)
    y_pred = predict(x, w, b)
    error = y_pred - y                    # this is e_i for every point at once (vectorized)
    grad_w = (2 / n) * np.sum(error * x)
    grad_b = (2 / n) * np.sum(error)
    return grad_w, grad_b


# Sanity check: at the bad guess (w=0, b=0), the gradient should point
# toward increasing w (since true_w=3.2 > 0), i.e. grad_w should be negative
# (gradient descent moves OPPOSITE the gradient, so negative grad_w -> w increases).
gw, gb = compute_gradients(x, y, 0.0, 0.0)
print(f"\nGradients at (w=0, b=0): grad_w={gw:.3f}, grad_b={gb:.3f}")
print("(grad_w should be negative here, since increasing w reduces loss toward true_w=3.2)")


# --- Step 5: The training loop — gradient DESCENT ---------------------------
# The gradient points in the direction of STEEPEST INCREASE of the loss.
# We want to decrease the loss, so we step in the OPPOSITE direction.
# `learning_rate` controls how big a step we take each time:
#   - too large -> we overshoot and can diverge (loss explodes)
#   - too small -> we converge correctly but very slowly
w, b = 0.0, 0.0            # start from a deliberately bad guess
learning_rate = 0.01
n_steps = 500

print("\n--- Training ---")
for step in range(n_steps):
    grad_w, grad_b = compute_gradients(x, y, w, b)
    w -= learning_rate * grad_w   # move OPPOSITE the gradient
    b -= learning_rate * grad_b
    if step % 50 == 0 or step == n_steps - 1:
        loss = mse_loss(predict(x, w, b), y)
        print(f"step {step:4d} | w={w:7.4f} | b={b:7.4f} | loss={loss:7.4f}")

print(f"\nRecovered:  w={w:.4f}, b={b:.4f}")
print(f"True:       w={true_w}, b={true_b}")
