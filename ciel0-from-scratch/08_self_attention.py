"""
Self-Attention From Scratch — Track C, item 13
====================================================

The mechanism that replaced RNNs. Core idea: every token produces three
vectors from its embedding —
    Query (Q): "what am I looking for from other tokens?"
    Key   (K): "what do I offer, that other tokens might look for?"
    Value (V): "what information do I actually contribute, once attended to?"

A token's OUTPUT is a weighted average of every token's Value vector,
where the weight is how well THIS token's Query matches THAT token's Key.
Every token computes this simultaneously — no left-to-right dependency,
which is exactly what fixes the RNN's sequential bottleneck from last
session.

Formula (the famous one):
    Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V

We'll build every piece of that formula by hand, on numbers small enough
to verify with a calculator, before applying it to real sentences.
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# --- Step 1: A tiny, fully hand-inspectable example -------------------------
# 3 tokens, embedding dimension 4. Made-up numbers — the point here is to
# verify the MECHANICS work correctly, not to represent real words yet.
np.random.seed(0)
n_tokens = 3
d_model = 4     # embedding dimension
d_k = 4         # dimension of Q and K (must match, since we dot-product them)

X = np.random.randn(n_tokens, d_model)   # 3 token embeddings, stacked
print(f"Input embeddings X, shape {X.shape}:\n{X}")

# In a real network these projection matrices are LEARNED (just like every
# weight matrix you've trained so far). For this first pass, we'll use
# random ones too, purely to verify the mechanics — training comes next.
W_q = np.random.randn(d_model, d_k) * 0.5
W_k = np.random.randn(d_model, d_k) * 0.5
W_v = np.random.randn(d_model, d_model) * 0.5

Q = X @ W_q   # shape (3, 4) — each row is one token's Query vector
K = X @ W_k   # shape (3, 4) — each row is one token's Key vector
V = X @ W_v   # shape (3, 4) — each row is one token's Value vector

print(f"\nQ shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
print("Each of the 3 rows in Q/K/V corresponds to one of the 3 tokens.")


# --- Step 2: Raw attention scores — Q @ K^T ---------------------------------
# scores[i, j] = dot product of token i's Query with token j's Key
#              = "how much does token i want to attend to token j?"
# A HIGH dot product means the vectors point in a similar direction —
# i.e. token j offers exactly what token i's Query is looking for.
raw_scores = Q @ K.T   # shape (3, 3) — one row per token, one column per token
print(f"\nRaw attention scores (Q @ K^T), shape {raw_scores.shape}:\n{raw_scores}")
print("Read row i as: 'how much token i attends to each of the 3 tokens (including itself)'")


# --- Step 3: Scale by sqrt(d_k) -----------------------------------------------
# WHY scale? Dot products of high-dimensional vectors tend to grow large
# in magnitude just from having more dimensions to accumulate across —
# not because the tokens are more "relevant" to each other. Large raw
# scores pushed through softmax next would make it EXTREMELY peaked
# (near one-hot), which starves gradients during training — the model
# would stop being able to learn nuanced attention patterns. Dividing by
# sqrt(d_k) counteracts this growth and keeps scores in a well-behaved
# range regardless of dimension size.
scaled_scores = raw_scores / np.sqrt(d_k)
print(f"\nScaled scores (/ sqrt({d_k})={np.sqrt(d_k):.2f}):\n{scaled_scores}")


# --- Step 4: Softmax — turn each row into a probability distribution -------
# Applied ROW-WISE: each token's attention weights across all tokens
# (including itself) must sum to 1, since the output is a WEIGHTED
# AVERAGE of Value vectors — weights need to be genuine proportions.
def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)   # numerical stability trick —
                                                             # doesn't change the result,
                                                             # just prevents overflow in exp()
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


attention_weights = softmax(scaled_scores, axis=-1)
print(f"\nAttention weights (after softmax):\n{attention_weights}")
print(f"Row sums (should all be 1.0): {attention_weights.sum(axis=-1)}")


# --- Step 5: The final output — weighted sum of Value vectors --------------
output = attention_weights @ V
print(f"\nOutput (attention_weights @ V), shape {output.shape}:\n{output}")
print("\nEach output row is now a BLEND of all 3 tokens' Value vectors, "
      "weighted by relevance — token 0's output row is no longer just "
      "'token 0's own information,' it's 'token 0's information, "
      "informed by whichever other tokens its Query found relevant.'")


# --- Step 6: Package it as one function, matching the famous formula -------
def self_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


packaged_output, packaged_weights = self_attention(Q, K, V)
assert np.allclose(packaged_output, output), "Packaged version should match step-by-step version"
print("\nPackaged self_attention() function matches the step-by-step version exactly.")


# --- Step 7: Verify against PyTorch's real, built-in implementation --------
# This confirms what we built by hand isn't a simplified toy version —
# it's the same formula PyTorch itself uses internally.
import torch
import torch.nn.functional as F

Q_t, K_t, V_t = torch.tensor(Q), torch.tensor(K), torch.tensor(V)
torch_output = F.scaled_dot_product_attention(Q_t, K_t, V_t)
print(f"\nPyTorch's F.scaled_dot_product_attention output:\n{torch_output.numpy()}")
print(f"Matches our from-scratch version: "
      f"{np.allclose(torch_output.numpy(), output, atol=1e-5)}")