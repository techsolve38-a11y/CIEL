"""
The Full Transformer Block — Track C, item 14
====================================================

Assembling: positional encoding + multi-head attention + feedforward +
residual connections + layer normalization. This is the last conceptual
piece before item 15 (building an actual tiny GPT) — every real
transformer, including CIEL-0's eventual architecture, is a stack of
blocks built exactly this way.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)
np.set_printoptions(precision=3, suppress=True)


# --- Piece 1: Positional Encoding -------------------------------------------
# Problem: self-attention computes Q@K^T between EVERY pair of tokens
# regardless of their order. If you shuffled the tokens in a sentence,
# attention's raw math wouldn't structurally notice — "dog bites man" and
# "man bites dog" would look identical to attention alone, which is
# obviously wrong for language. We need to inject POSITION information
# directly into the input.
#
# The classic solution (from the original "Attention is All You Need"
# paper): add a unique, deterministic pattern of sine/cosine values to
# each position's embedding, using different frequencies per dimension.
# Nearby positions get similar patterns; distant positions get very
# different ones — and critically, it's a fixed formula, not learned,
# so it works even for sequence lengths longer than anything seen in
# training.
def positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]              # shape (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)   # even dimensions: sine
    pe[:, 1::2] = np.cos(position * div_term)   # odd dimensions: cosine
    return pe


seq_len, d_model = 6, 8
pe = positional_encoding(seq_len, d_model)
print(f"Positional encoding shape: {pe.shape}")
print(f"Position 0: {pe[0]}")
print(f"Position 1: {pe[1]}")
print(f"Position 5: {pe[5]}")
print("\nNotice position 0 and 1 are fairly close in value; position 5 is "
      "much more different from position 0 — the pattern encodes DISTANCE, "
      "which attention can then learn to make use of.")


# --- Piece 2: Multi-Head Attention -------------------------------------------
# Single-head attention (item 13) forces ALL relationship-finding through
# one set of Q/K/V projections. But language has many simultaneous kinds
# of relationships — e.g. one pattern might track "which noun does this
# adjective describe," another might track "which verb does this subject
# belong to." Multi-head attention runs SEVERAL smaller attention
# computations in parallel (each with its own learned Q/K/V projections,
# each operating on a smaller slice of the dimensions), then concatenates
# their outputs back together. Each head is free to specialize.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly across heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads   # dimension PER head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)   # final projection after concatenating heads

    def forward(self, x, causal_mask=False):
        batch, seq_len, d_model = x.shape

        Q = self.W_q(x)   # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split d_model into (n_heads, d_head) and move heads to their own
        # dimension so each head's attention is computed independently.
        Q = Q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # shapes now: (batch, n_heads, seq_len, d_head)

        attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=causal_mask)
        # attn_output shape: (batch, n_heads, seq_len, d_head)

        # Recombine heads back into (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_out(attn_output)


mha = MultiHeadAttention(d_model=8, n_heads=2)
x_test = torch.randn(1, seq_len, d_model)   # (batch=1, seq_len=6, d_model=8)
mha_out = mha(x_test)
print(f"\nMulti-head attention: input shape {x_test.shape} -> output shape {mha_out.shape}")
print("Input and output shape match — this is essential, since blocks get STACKED, "
      "so each block's output must be a valid input to the next.")


# --- Piece 3: Feedforward Network -------------------------------------------
# Applied INDEPENDENTLY to each token's vector (no cross-token interaction
# here — that's attention's job). Attention gathers information ACROSS
# tokens; the feedforward network then processes what each individual
# token ended up with. Standard design: project UP to a wider hidden
# dimension, apply a nonlinearity, project back down — giving the model
# extra representational capacity per token.
class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),   # a smoother variant of ReLU — GPT-style models use this by convention
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


# --- Piece 4: Residual connections + Layer Normalization --------------------
# RESIDUAL CONNECTION: output = x + sublayer(x), NOT just sublayer(x).
#   Why: in a network with many stacked layers, gradients have to flow
#   backward through every single one during training. Without a direct
#   path (the "+x" skip), gradients can shrink toward zero as they pass
#   through many layers (vanishing gradients) — a real problem you'll
#   recognize from Track B's gradient discussions, now at greater depth.
#   The "+x" gives gradients a direct shortcut back to earlier layers,
#   regardless of how deep the network is.
#
# LAYER NORMALIZATION: rescales each token's vector to have consistent
#   mean/variance BEFORE it's processed by the next sublayer. This keeps
#   activations in a stable numerical range as they flow through many
#   stacked blocks, which makes training dramatically more reliable —
#   without it, deep transformers are notoriously hard to train at all.
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask=False):
        # Pre-norm architecture (the modern convention, used by GPT-2 and
        # later): normalize BEFORE each sublayer, then add the residual.
        x = x + self.attention(self.norm1(x), causal_mask=causal_mask)
        x = x + self.feedforward(self.norm2(x))
        return x


# --- Assemble and verify the full block -------------------------------------
block = TransformerBlock(d_model=8, n_heads=2, d_ff=32)

# Real input: positional encoding ADDED to token embeddings, so the
# network receives both "what token is this" and "where is it."
token_embeddings = torch.randn(1, seq_len, d_model)
pos_encoding_t = torch.tensor(pe, dtype=torch.float32).unsqueeze(0)   # add batch dim
block_input = token_embeddings + pos_encoding_t

block_output = block(block_input, causal_mask=True)
print(f"\nFull transformer block: input shape {block_input.shape} -> "
      f"output shape {block_output.shape}")
print("Shape preserved end to end — this exact block can now be STACKED "
      "N times (a real GPT stacks dozens), and item 15 does exactly that.")

n_params = sum(p.numel() for p in block.parameters())
print(f"\nParameters in this one small block: {n_params:,}")


# --- Verify causal masking actually prevents looking at the future ---------
# For language GENERATION, token i must not be able to attend to tokens
# after it — otherwise the model could "cheat" during training by peeking
# at the answer it's supposed to predict. is_causal=True in
# scaled_dot_product_attention enforces this by masking out (setting to
# -infinity before softmax) any attention score where key position > query
# position, so softmax assigns it exactly zero weight.
#
# We verify this empirically: changing a LATER token's embedding should
# have ZERO effect on an EARLIER token's output, if causal masking is
# working correctly.
torch.manual_seed(0)
x_a = torch.randn(1, seq_len, d_model)
x_b = x_a.clone()
x_b[0, -1] += 100.0   # drastically change ONLY the LAST token's embedding

with torch.no_grad():
    out_a = block(x_a, causal_mask=True)
    out_b = block(x_b, causal_mask=True)

# Compare every position EXCEPT the last (which legitimately changed)
early_positions_changed = not torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-4)
print(f"\nChanged only the LAST token drastically. Did EARLIER tokens' "
      f"outputs change? {early_positions_changed}")
print("Should be False — earlier tokens must be completely unaffected by "
      "a later token, confirming causal masking is working correctly.")