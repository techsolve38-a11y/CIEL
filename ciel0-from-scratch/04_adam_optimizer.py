"""
The Adam Optimizer — Track B, item 7
========================================

Goal: replace plain gradient descent's single update rule
    theta -= learning_rate * grad
with Adam, which fixes three specific weaknesses of plain GD:

1. NO MEMORY: plain GD only looks at the current gradient. If gradients
   are noisy, updates zig-zag. Adam keeps a running average of past
   gradients (momentum) so it moves in a consistently smoothed direction.

2. ONE-SIZE-FITS-ALL STEP SIZE: plain GD takes the same size step for
   every parameter. But some parameters may need small careful steps
   (steep, sensitive directions) while others could take large steps
   (flat, insensitive directions). Adam tracks each parameter's own
   gradient history and scales its step size individually.

3. FRAGILE LEARNING RATE CHOICE: you saw learning_rate=0.5 diverge to
   nan while 0.1 converged cleanly, on a TWO-parameter problem. At the
   scale of a real network (millions of parameters), hand-tuning one
   global learning rate that works for every parameter simultaneously
   becomes close to impossible. Adam's per-parameter adaptive scaling
   makes it far more forgiving of an imperfect learning rate choice.

We reuse the exact same XOR data and MLP architecture from last session,
changing ONLY the update rule, so the comparison isolates one variable.
"""

import numpy as np

np.random.seed(42)

# --- Setup: identical to 03_mlp_backprop.py -----------------------------
n_per_cluster = 50


def make_cluster(center, n=n_per_cluster, spread=0.15):
    return np.random.randn(n, 2) * spread + np.array(center)


X = np.vstack([
    make_cluster([0, 0]), make_cluster([1, 1]),
    make_cluster([0, 1]), make_cluster([1, 0]),
])
y = np.concatenate([
    np.zeros(n_per_cluster), np.zeros(n_per_cluster),
    np.ones(n_per_cluster), np.ones(n_per_cluster),
])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(a, y, eps=1e-9):
    a = np.clip(a, eps, 1 - eps)
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))


def mlp_forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def backward(X, y, z1, a1, z2, a2, W2):
    n = len(y)
    y = y.reshape(-1, 1)
    dz2 = a2 - y
    dW2 = a1.T @ dz2 / n
    db2 = np.mean(dz2, axis=0)
    da1 = dz2 @ W2.T
    dz1 = da1 * a1 * (1 - a1)
    dW1 = X.T @ dz1 / n
    db1 = np.mean(dz1, axis=0)
    return dW1, db1, dW2, db2


n_inputs, n_hidden = 2, 4
print("Setup complete — reusing XOR data and MLP architecture from last session.")


# --- The Adam optimizer, built from its two component ideas ----------------
#
# IDEA 1 — MOMENTUM (the "m" in Adam):
#   Instead of stepping directly by the current gradient, keep a running,
#   exponentially-weighted average of past gradients:
#       m = beta1 * m + (1 - beta1) * grad
#   With beta1=0.9 (the standard default), this means "today's direction
#   is 90% yesterday's accumulated direction, 10% today's new gradient."
#   This smooths out noisy, jittery gradients and builds up speed in a
#   consistent direction — like a ball rolling downhill gaining momentum,
#   which is exactly the physical metaphor it's named for.
#
# IDEA 2 — ADAPTIVE SCALING (the "v" in Adam):
#   Separately, keep a running average of the SQUARED gradient:
#       v = beta2 * v + (1 - beta2) * grad**2
#   This tracks how LARGE this parameter's gradients typically are,
#   regardless of their direction (squaring removes the sign). Dividing
#   the update by sqrt(v) means: parameters with consistently large
#   gradients get their steps shrunk, parameters with small/rare
#   gradients get their steps relatively amplified. Every parameter
#   effectively gets its own personalized learning rate.
#
# BIAS CORRECTION (the part that looks like unnecessary complexity but isn't):
#   m and v both START at zero. Early in training, that zero-initialization
#   biases them toward being too small — the running average hasn't had
#   time to "fill up" yet. Dividing by (1 - beta1**t) and (1 - beta2**t),
#   where t is the current step number, corrects exactly for this early
#   bias. As t grows large, beta1**t and beta2**t shrink toward 0, so this
#   correction fades out — it only matters in the first several steps.
#
# PUTTING IT TOGETHER, per parameter, per step:
#   m = beta1*m + (1-beta1)*grad
#   v = beta2*v + (1-beta2)*grad**2
#   m_hat = m / (1 - beta1**t)
#   v_hat = v / (1 - beta2**t)
#   theta -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
class AdamOptimizer:
    """One Adam optimizer instance per parameter tensor (W1, b1, W2, b2 each
    get their own — they must not share momentum/scaling state, since
    they're unrelated quantities)."""

    def __init__(self, shape, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, param, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# --- Head-to-head: plain GD vs Adam, identical starting weights ------------
def init_weights():
    np.random.seed(7)   # same seed for both runs -> identical starting point
    W1 = np.random.randn(n_inputs, n_hidden) * 0.5
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_hidden, 1) * 0.5
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def train_plain_gd(n_steps=1000, learning_rate=1.0):
    W1, b1, W2, b2 = init_weights()
    history = []
    for step in range(n_steps):
        z1, a1, z2, a2 = mlp_forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, z1, a1, z2, a2, W2)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        if step % 50 == 0:
            loss = binary_cross_entropy(a2.flatten(), y)
            history.append((step, loss))
    return history


def train_adam(n_steps=1000, learning_rate=0.05):
    W1, b1, W2, b2 = init_weights()
    opt_W1 = AdamOptimizer(W1.shape, learning_rate)
    opt_b1 = AdamOptimizer(b1.shape, learning_rate)
    opt_W2 = AdamOptimizer(W2.shape, learning_rate)
    opt_b2 = AdamOptimizer(b2.shape, learning_rate)
    history = []
    for step in range(n_steps):
        z1, a1, z2, a2 = mlp_forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, z1, a1, z2, a2, W2)
        W1 = opt_W1.step(W1, dW1)
        b1 = opt_b1.step(b1, db1)
        W2 = opt_W2.step(W2, dW2)
        b2 = opt_b2.step(b2, db2)
        if step % 50 == 0:
            loss = binary_cross_entropy(a2.flatten(), y)
            history.append((step, loss))
    return history


print("\n--- Plain Gradient Descent (lr=1.0) ---")
gd_history = train_plain_gd()
for step, loss in gd_history[:6]:
    print(f"step {step:4d} | loss={loss:.4f}")

print("\n--- Adam (lr=0.05) ---")
adam_history = train_adam()
for step, loss in adam_history[:6]:
    print(f"step {step:4d} | loss={loss:.4f}")

print("\n--- Comparison at step 250 ---")
gd_at_250 = [l for s, l in gd_history if s == 250][0]
adam_at_250 = [l for s, l in adam_history if s == 250][0]
print(f"Plain GD loss at step 250:  {gd_at_250:.4f}")
print(f"Adam loss at step 250:      {adam_at_250:.4f}")

print(f"\nAdam reached in 250 steps roughly what plain GD needed ~2000+ steps for "
      f"in your last session — same math, same architecture, just a smarter "
      f"update rule. This is why virtually every modern network, including "
      f"real transformers, trains with Adam (or a close variant, AdamW) "
      f"instead of plain gradient descent.")