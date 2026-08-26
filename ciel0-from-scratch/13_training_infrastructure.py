"""
Training Infrastructure — Track D, item 18
================================================

Three pieces of unglamorous engineering that matter enormously once
training runs take hours instead of minutes:

1. GRADIENT ACCUMULATION — train with an effectively larger batch size
   than fits in memory at once.
2. MIXED PRECISION — use lower-precision numbers for most computation to
   save memory and (on GPU) increase speed.
3. PROPER CHECKPOINTING — save enough state to resume training EXACTLY,
   not just the model weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)

# --- Piece 1: Gradient Accumulation -----------------------------------------
# Problem: you want to train with batch_size=64 (larger batches generally
# give smoother, more reliable gradient estimates), but your GPU/CPU
# memory can only fit batch_size=16 at once.
#
# Solution: run 4 separate forward+backward passes of size 16 each,
# WITHOUT stepping the optimizer between them (gradients accumulate —
# add up — in each parameter's .grad, since PyTorch does this by default
# unless you zero_grad()). After 4 accumulated passes, step the optimizer
# ONCE. This should be mathematically equivalent to one batch of 64.
#
# We verify this claim directly, rather than just asserting it.
model_a = nn.Linear(4, 1)
model_b = nn.Linear(4, 1)
model_b.load_state_dict(model_a.state_dict())   # IDENTICAL starting weights

torch.manual_seed(0)
full_batch_x = torch.randn(64, 4)
full_batch_y = torch.randn(64, 1)

# --- Approach A: one big batch of 64 ---
optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
optimizer_a.zero_grad()
pred_a = model_a(full_batch_x)
loss_a = F.mse_loss(pred_a, full_batch_y)
loss_a.backward()
grad_a = model_a.weight.grad.clone()
optimizer_a.step()

# --- Approach B: 4 accumulated mini-batches of 16 each ---
optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
optimizer_b.zero_grad()
accumulation_steps = 4
mini_batch_size = 64 // accumulation_steps
for i in range(accumulation_steps):
    start, end = i * mini_batch_size, (i + 1) * mini_batch_size
    pred_b = model_b(full_batch_x[start:end])
    # CRITICAL: divide the loss by accumulation_steps. Each mini-batch's
    # loss is currently an average over only 16 samples; without dividing,
    # summing 4 such losses' gradients would effectively be a 4x too-large
    # gradient relative to the true 64-sample average. This is the single
    # most common bug when implementing gradient accumulation.
    loss_b = F.mse_loss(pred_b, full_batch_y[start:end]) / accumulation_steps
    loss_b.backward()   # accumulates into .grad — no zero_grad() between iterations
grad_b = model_b.weight.grad.clone()
optimizer_b.step()

print(f"Gradient from one batch of 64:        {grad_a.flatten()}")
print(f"Gradient from 4 accumulated batches:  {grad_b.flatten()}")
print(f"Match within floating point tolerance: {torch.allclose(grad_a, grad_b, atol=1e-5)}")

weights_match = torch.allclose(model_a.weight, model_b.weight, atol=1e-5)
print(f"Resulting weights after optimizer step also match: {weights_match}")


# --- Piece 2: Mixed Precision -----------------------------------------------
# Every number you've used so far is float32 (32 bits per number). Mixed
# precision runs MOST computation in a lower-precision format (float16 or
# bfloat16, 16 bits) while keeping a few numerically sensitive operations
# (like the final loss accumulation) in float32. Benefits at real scale:
#   - Roughly HALF the memory for activations and gradients
#   - On GPUs with dedicated hardware for it (tensor cores), often
#     2-3x faster matrix multiplication
#
# HONEST CAVEAT: we're on CPU, single-threaded, with no GPU tensor cores.
# The code below runs correctly and demonstrates the MECHANISM, but you
# will NOT see a speedup here — mixed precision's performance benefit is
# specifically a GPU hardware story. Understanding the mechanism now means
# you'll know exactly what to enable the day you train on a GPU.
print("\n--- Mixed precision (mechanism demo — no speedup expected on CPU) ---")

small_model = nn.Linear(10, 10)
x = torch.randn(4, 10)
y = torch.randn(4, 10)
optimizer = torch.optim.SGD(small_model.parameters(), lr=0.01)

# torch.autocast automatically runs eligible operations (mainly matrix
# multiplications) in a lower-precision dtype within its context, while
# keeping numerically sensitive ops (reductions, loss computation) in
# float32 automatically — this "automatic" choice-making is exactly what
# "autocast" refers to in the name.
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    pred = small_model(x)
    print(f"Prediction dtype INSIDE autocast block: {pred.dtype}")
    loss = F.mse_loss(pred, y)
    print(f"Loss dtype (kept higher precision automatically): {loss.dtype}")

loss.backward()
optimizer.step()
print("Training step completed successfully under autocast.")


# --- Piece 3: Proper Checkpointing ------------------------------------------
# 10_tiny_gpt.py's checkpoint only saved model_state and total_steps. This
# is INCOMPLETE, and here's the concrete proof of why: Adam maintains
# per-parameter momentum (m) and variance (v) running averages, built up
# over many steps (see 04_adam_optimizer.py). If you save only the model
# weights and later resume by creating a FRESH optimizer, that optimizer
# starts with m=0, v=0 — throwing away everything Adam had learned about
# each parameter's gradient behavior. Training doesn't crash, but it
# doesn't resume as if uninterrupted either — there's a real discontinuity.
torch.manual_seed(0)
model_full = nn.Linear(4, 4)
opt_full = torch.optim.Adam(model_full.parameters(), lr=0.1)

# Run several real steps so Adam's internal momentum/variance state
# becomes non-trivial (not just its zero initialization).
for _ in range(20):
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    loss = F.mse_loss(model_full(x), y)
    opt_full.zero_grad()
    loss.backward()
    opt_full.step()

# Grab Adam's actual internal state for inspection.
adam_state = opt_full.state[list(model_full.parameters())[0]]
print(f"\nAfter 20 steps, Adam's momentum buffer (m) for this parameter "
      f"(first few values): {adam_state['exp_avg'].flatten()[:4]}")
print("This is exactly the kind of state that gets SILENTLY LOST if you "
      "only checkpoint model weights.")

# --- The correct way: save model state AND optimizer state together -------
checkpoint = {
    "model_state": model_full.state_dict(),
    "optimizer_state": opt_full.state_dict(),   # <- the piece 10_tiny_gpt.py was missing
    "step": 20,
}
torch.save(checkpoint, "proper_checkpoint.pt")

# Simulate resuming in a fresh process: new model, new optimizer, load both.
resumed_model = nn.Linear(4, 4)
resumed_opt = torch.optim.Adam(resumed_model.parameters(), lr=0.1)
loaded = torch.load("proper_checkpoint.pt", weights_only=True)
resumed_model.load_state_dict(loaded["model_state"])
resumed_opt.load_state_dict(loaded["optimizer_state"])

resumed_adam_state = resumed_opt.state[list(resumed_model.parameters())[0]]
momentum_preserved = torch.allclose(adam_state["exp_avg"], resumed_adam_state["exp_avg"])
print(f"\nAfter proper resume, Adam's momentum buffer matches exactly: {momentum_preserved}")
print("A checkpoint that saves ONLY model weights would resume with this "
      "reset to zero instead — training would continue, but not identically "
      "to how it would have without the interruption.")