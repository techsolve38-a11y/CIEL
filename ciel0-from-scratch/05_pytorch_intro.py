"""
Moving to PyTorch — Track B, item 9
========================================

Goal: rebuild the EXACT SAME problem (XOR, 2-input -> 4-hidden -> 1-output,
sigmoid, binary cross-entropy, Adam) in PyTorch, and confirm it converges
to the same place. This is the payoff for building everything by hand
first — you'll be able to see precisely what PyTorch is automating,
because you already built each piece yourself:

    YOUR CODE (03/04)                    PYTORCH EQUIVALENT
    ------------------                   -------------------
    W1, b1, W2, b2 as raw arrays    ->   nn.Linear layers hold these
    manual forward() function       ->   model.forward() / model(x)
    manual backward() with          ->   loss.backward() — AUTOGRAD
      hand-derived chain rule            walks the computation graph
                                          backward automatically
    binary_cross_entropy()          ->   nn.BCELoss()
    AdamOptimizer class              ->   torch.optim.Adam

Nothing here is conceptually new. It's the same math, automated.
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(7)
np.random.seed(42)

# --- Same XOR data as before, now as PyTorch tensors ------------------------
n_per_cluster = 50


def make_cluster(center, n=n_per_cluster, spread=0.15):
    return np.random.randn(n, 2) * spread + np.array(center)


X_np = np.vstack([
    make_cluster([0, 0]), make_cluster([1, 1]),
    make_cluster([0, 1]), make_cluster([1, 0]),
]).astype(np.float32)
y_np = np.concatenate([
    np.zeros(n_per_cluster), np.zeros(n_per_cluster),
    np.ones(n_per_cluster), np.ones(n_per_cluster),
]).astype(np.float32)

# Tensors are PyTorch's version of numpy arrays — same idea, but they
# additionally track the operations performed on them, which is what
# makes autograd possible.
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np).reshape(-1, 1)   # PyTorch wants shape (n, 1) to match model output

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X dtype: {X.dtype}  (PyTorch defaults to float32, not float64 like numpy — "
      f"faster, and precise enough for neural nets)")


# --- The model — replaces your manual W1/b1/W2/b2 + mlp_forward() ----------
# nn.Linear(in_features, out_features) creates a weight matrix + bias
# vector internally (exactly what you built by hand as W1/b1), and
# nn.Sequential chains layers together so forward() is just "run the
# input through each layer in order."
model = nn.Sequential(
    nn.Linear(2, 4),   # equivalent to your W1 (shape 2x4) + b1
    nn.Sigmoid(),      # equivalent to your sigmoid(z1)
    nn.Linear(4, 1),   # equivalent to your W2 (shape 4x1) + b2
    nn.Sigmoid(),      # equivalent to your sigmoid(z2)
)

# Inspect what got created — confirm the shapes match what you built by hand.
for name, param in model.named_parameters():
    print(f"{name}: shape {tuple(param.shape)}")


# --- Loss and optimizer — replaces binary_cross_entropy() + AdamOptimizer --
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)   # same lr as your hand-rolled Adam

# --- Training loop — compare this to train_adam() in 04_adam_optimizer.py --
# Notice what's GONE compared to your version: no backward() function you
# wrote yourself, no manual dW1/db1/dW2/db2, no manual parameter update
# lines. Three lines do all of that:
#   loss.backward()       <- autograd computes EVERY gradient automatically,
#                             by walking the computation graph backward —
#                             this is doing exactly what your hand-derived
#                             chain rule did, for an arbitrarily deep network,
#                             without you deriving anything by hand
#   optimizer.step()      <- applies the Adam update rule you built by hand
#   optimizer.zero_grad() <- PyTorch accumulates gradients by default, so
#                             this clears them before the next step (a
#                             common bug for beginners is forgetting this)
print("\n--- Training (PyTorch) ---")
for step in range(300):
    predictions = model(X)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 30 == 0 or step == 299:
        with torch.no_grad():
            preds_binary = (predictions >= 0.5).float()
            accuracy = (preds_binary == y).float().mean().item()
        print(f"step {step:4d} | loss={loss.item():.4f} | accuracy={accuracy:.3f}")

print(f"\nCompare this loss curve to your hand-rolled Adam run in "
      f"04_adam_optimizer.py (lr=0.05) — it should converge on a very "
      f"similar trajectory, since it's mathematically the same algorithm.")