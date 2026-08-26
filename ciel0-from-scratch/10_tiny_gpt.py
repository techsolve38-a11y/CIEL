"""
Tiny GPT From Scratch — Track C, item 15
=============================================

THE flagship exercise of this entire curriculum. Every piece from items
10-14 (tokenization concepts, embeddings, self-attention, the full
transformer block) gets assembled here into an actual generative
language model, trained on real text (Shakespeare), that produces
genuinely novel text by the end.

We use CHARACTER-level tokenization here rather than the BPE tokenizer
from item 10 — simpler vocabulary (a few dozen characters vs a trained
subword vocab), which keeps the model small enough to train on CPU in
reasonable time while we focus on the architecture itself.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)   # Karpathy's traditional seed for this exact exercise

# --- Step 1: Load and tokenize the data -------------------------------------
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
train_data = data[:n]
val_data = data[n:]

print(f"Total characters: {len(text):,}")
print(f"Vocabulary size: {vocab_size} (unique characters)")
print(f"Train size: {len(train_data):,}, Val size: {len(val_data):,}")
print(f"Sample encode/decode round trip: {decode(encode('Hello there')) == 'Hello there'}")


# --- Step 2: Batch sampling --------------------------------------------------
# Pick random starting points in the data, take a `block_size`-length
# window as input, and the SAME window shifted one character later as the
# target. This is next-token prediction (same objective as item 11's toy
# version), just now over many positions in the sequence simultaneously —
# the causal mask from item 14 is what makes this valid: position i's
# prediction genuinely can't see position i's own target or anything later.
def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


block_size, batch_size = 64, 32
xb, yb = get_batch(train_data, block_size, batch_size)
print(f"\nBatch shapes — x: {xb.shape}, y: {yb.shape}")
print(f"Example: x[0][:10] = {xb[0][:10].tolist()}")
print(f"         y[0][:10] = {yb[0][:10].tolist()}  (same as x, shifted by 1)")
print(f"As text — input:  {decode(xb[0][:20].tolist())!r}")
print(f"          target: {decode(yb[0][:20].tolist())!r}")


# --- Step 3: The model — stack of transformer blocks from item 14 ----------
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
        self.position_embedding = nn.Embedding(block_size, d_model)   # LEARNED positions
        # (item 14 used fixed sine/cosine encoding; GPT-2 style models
        # instead LEARN a position embedding, same mechanism as token
        # embeddings — both are valid, this is the more common modern choice)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)   # projects back to vocab-sized logits

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_emb = self.token_embedding(idx)                              # (b, t, d_model)
        pos_emb = self.position_embedding(torch.arange(t, device=idx.device))  # (t, d_model)
        x = tok_emb + pos_emb                                             # broadcast add
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output_head(x)                                      # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # can't feed more than block_size context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]               # only care about the LAST position's prediction
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # sample, don't just argmax —
                                                                    # keeps generation varied
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = TinyGPT(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, block_size=block_size)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")


# --- Step 4: Generate from the UNTRAINED model — should be pure noise ------
def generate_text(model, prompt, max_new_tokens=200):
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens)
    return decode(out[0].tolist())


print("\n--- Untrained generation (expect gibberish) ---")
print(generate_text(model, "\n", max_new_tokens=200))


# --- Step 5: Real training, in checkpointed, RESUMABLE chunks ---------------
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

CHECKPOINT_PATH = "tiny_gpt_checkpoint.pt"
STEPS_THIS_RUN = 1000

start_step = 0
try:
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    start_step = ckpt["total_steps"]
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"\nResumed from checkpoint at step {start_step} (optimizer state restored too).")
    else:
        # Backward-compat: an OLD-format checkpoint (from before item 18's
        # fix) has no optimizer state to restore. Model weights still load
        # correctly — only Adam's momentum/variance restart from zero this
        # one time. From this save onward, the new format keeps it.
        print(f"\nResumed from checkpoint at step {start_step}. "
              f"(Old checkpoint format — no optimizer state found, so Adam's "
              f"momentum restarts fresh this one time. Future resumes will "
              f"carry it forward correctly.)")
except FileNotFoundError:
    print("\nNo checkpoint found — starting fresh.")

print(f"--- Training steps {start_step} to {start_step + STEPS_THIS_RUN} ---")
start = time.time()
for step in range(STEPS_THIS_RUN):
    xb, yb = get_batch(train_data, block_size, batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0 or step == STEPS_THIS_RUN - 1:
        with torch.no_grad():
            xv, yv = get_batch(val_data, block_size, batch_size)
            _, val_loss = model(xv, yv)
        elapsed = time.time() - start
        print(f"step {start_step + step:5d} | train loss={loss.item():.4f} | "
              f"val loss={val_loss.item():.4f} | {elapsed:.1f}s elapsed")

torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),   # FIX (see item 18)
    "total_steps": start_step + STEPS_THIS_RUN,
}, CHECKPOINT_PATH)
print(f"\nCheckpoint saved after {start_step + STEPS_THIS_RUN} total steps.")

print("\n--- Generation after training ---")
print(generate_text(model, "\n", max_new_tokens=300))