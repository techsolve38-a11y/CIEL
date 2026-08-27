"""
Live Interactive Demo — test CIEL-0's tiny prototype yourself
==================================================================

Run this and type things in. Three modes:
  1. A trained instruction (GREET / FAREWELL / INSULT) -> exact learned response
  2. An UNTRAINED instruction (e.g. THREATEN, or anything else) -> genuine test
     of whether the model generalized the INSTRUCTION FORMAT itself, even
     without having learned that specific category's content
  3. Free text (no recognized instruction) -> raw Shakespeare-style continuation,
     testing whether pretrained capability survived fine-tuning

Type 'quit' to exit.

HONEST EXPECTATIONS: this is an 818K-parameter, character-level model
trained briefly on CPU. It will not feel like a real assistant. Its value
is that it's REAL and RUNNING — everything happening underneath is
exactly what you built and verified piece by piece, not a simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
base_chars = sorted(set(text))

ckpt = torch.load("fine_tuned_checkpoint.pt", weights_only=True)
special_tokens = ckpt["special_tokens"]
base_vocab_size = ckpt["base_vocab_size"]
all_tokens = base_chars + special_tokens
vocab_size = len(all_tokens)
token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
id_to_token = {i: tok for i, tok in enumerate(all_tokens)}


def encode(s):
    ids, i = [], 0
    while i < len(s):
        matched = False
        for tok in special_tokens:
            if s[i:i + len(tok)] == tok:
                ids.append(token_to_id[tok])
                i += len(tok)
                matched = True
                break
        if not matched:
            ids.append(token_to_id[s[i]])
            i += 1
    return ids


def decode(ids):
    return "".join(id_to_token[i] for i in ids)


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

    def forward(self, idx):
        b, t = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, stop_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and next_id.item() == stop_token:
                break
        return idx


model = TinyGPT(vocab_size=vocab_size, d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
                 n_layers=ckpt["n_layers"], d_ff=ckpt["d_ff"], block_size=ckpt["block_size"])
model.load_state_dict(ckpt["model_state"])
model.eval()

print("=" * 60)
print("CIEL-0 prototype — live interactive demo")
print("=" * 60)
print("Instruction mode: type !WORD  (e.g. !GREET, !FAREWELL, !INSULT — trained)")
print("                   or !THREATEN, !WARN, etc. — untrained, genuine generalization test")
print("Continuation mode: type anything else (e.g. ROMEO:) for raw Shakespeare-style text")
print("Type 'quit' to exit.\n")

known_instructions = {"GREET", "FAREWELL", "INSULT"}

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    if user_input.startswith("!"):
        # Unambiguous instruction syntax — no guessing from casing, which
        # previously collided with Shakespeare's own all-caps character
        # names (e.g. "ROMEO:") and misclassified them as instructions.
        instruction = user_input[1:].strip().upper()
        is_known = instruction in known_instructions
        prompt = f"<INS>{instruction}<RES>"
        idx = torch.tensor([encode(prompt)], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=60, temperature=0.7,
                              stop_token=token_to_id["<END>"])
        response = decode(out[0].tolist())[len(prompt):].replace("<END>", "")
        tag = "(trained)" if is_known else "(UNTRAINED — testing generalization)"
        print(f"CIEL-0 {tag}: {response}\n")
    else:
        idx = torch.tensor([encode(user_input)], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=100, temperature=0.8)
        print(f"CIEL-0 (continuation): {decode(out[0].tolist())}\n")