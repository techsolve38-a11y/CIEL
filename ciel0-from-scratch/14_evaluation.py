"""
Evaluation, Done Properly — Track D, item 19
==================================================

Formalizes two things we've been doing loosely: reading loss numbers
meaningfully (perplexity), and measuring them reliably (averaged
held-out evaluation, not one noisy batch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# --- Reload the real trained model from the checkpoint ----------------------
with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [char_to_id[c] for c in s]


def decode(ids):
    return "".join(id_to_char[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, t, d = x.shape
        Q = self.W_q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.W_out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # temperature scaling — see below
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


block_size = 64
model = TinyGPT(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=512, block_size=block_size)
ckpt = torch.load("tiny_gpt_checkpoint.pt", weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()   # disables any training-only behavior (none in this model, but correct habit)
print(f"Loaded checkpoint at step {ckpt['total_steps']}.")


# --- Stable evaluation: average over MANY batches, not one ------------------
# During training we printed loss from a single batch each time — noisy,
# since one batch of 32 sequences is a small, high-variance sample. A real
# evaluation should average over many batches for a number you can
# actually trust and compare across checkpoints.
@torch.no_grad()
def evaluate(model, data, block_size, batch_size=32, n_batches=50):
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


train_loss = evaluate(model, train_data, block_size)
val_loss = evaluate(model, val_data, block_size)
print(f"\nStable train loss (avg over 50 batches): {train_loss:.4f}")
print(f"Stable val loss   (avg over 50 batches): {val_loss:.4f}")


# --- Perplexity: exp(loss) — an interpretable version of the same number --
# Cross-entropy loss is in "nats" (natural log units) — not very
# intuitive on its own. Perplexity = e^loss has a genuine interpretation:
# "the model is, on average, as uncertain as if choosing uniformly among
# this many options." Lower is better; a perfect model has perplexity 1
# (always certain); random guessing among 65 characters has perplexity 65.
train_perplexity = torch.exp(torch.tensor(train_loss)).item()
val_perplexity = torch.exp(torch.tensor(val_loss)).item()
print(f"\nTrain perplexity: {train_perplexity:.2f}  (vs. random-guess baseline: {vocab_size})")
print(f"Val perplexity:   {val_perplexity:.2f}")
print(f"\nInterpretation: this model is, on average, about as uncertain as choosing "
      f"among ~{val_perplexity:.1f} equally-likely next characters — a big improvement "
      f"over the {vocab_size}-way uncertainty it started with, but still far from a "
      f"confident, well-trained model (which would push this much closer to 1-3 for "
      f"character-level English).")


# --- Qualitative sampling: the effect of temperature ------------------------
# Recall generate() divides logits by `temperature` before softmax.
#   temperature < 1: SHARPENS the distribution — the model's already-favored
#     choices become even more dominant. At temperature -> 0, generation
#     becomes fully deterministic (always the single most likely character).
#   temperature = 1: use the model's raw learned probabilities, unmodified.
#   temperature > 1: FLATTENS the distribution — less-favored choices become
#     relatively more likely, increasing randomness/variety at the cost of
#     coherence.
def generate_text(model, prompt, max_new_tokens, temperature):
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature)
    return decode(out[0].tolist())


print("\n--- Same model, same prompt, different temperatures ---")
for temp in [0.3, 0.8, 1.5]:
    print(f"\ntemperature={temp}:")
    print(generate_text(model, "\nROMEO:", max_new_tokens=150, temperature=temp))