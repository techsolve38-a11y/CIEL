"""
Instruction Tuning (SFT) — Track D, item 20
=================================================

Goal: turn a model that only continues text plausibly into one that
follows a specific instruction/response FORMAT. We teach a behavior
pretraining on raw Shakespeare could never have produced — proving the
effect is real, not just "the model got lucky continuing familiar text."

Two genuinely important real-world SFT mechanics get covered here:
  1. EXTENDING THE VOCABULARY — new special tokens (<INS>, <RES>, <END>)
     don't exist in the pretrained vocab, so the embedding table and
     output head must grow new rows, while preserving everything already
     learned for the original 65 characters.
  2. LOSS MASKING — during SFT, we only want to train the model to
     PREDICT THE RESPONSE, not to predict the instruction (which is
     given, not generated). We mask instruction-token loss out entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
base_chars = sorted(set(text))
base_vocab_size = len(base_chars)

# --- Extend the vocabulary with new special tokens --------------------------
special_tokens = ["<INS>", "<RES>", "<END>"]
all_tokens = base_chars + special_tokens   # each special token is now ONE id, not spelled out
vocab_size = len(all_tokens)
token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
id_to_token = {i: tok for i, tok in enumerate(all_tokens)}

print(f"Base vocab size: {base_vocab_size}")
print(f"Extended vocab size: {vocab_size} (+{len(special_tokens)} special tokens)")


def encode(s, specials_present=True):
    """Encode text that may contain our special token STRINGS embedded in
    it (e.g. '<INS>GREET<RES>...'). We scan for special tokens first,
    falling back to character-level encoding everywhere else."""
    ids = []
    i = 0
    while i < len(s):
        matched = False
        if specials_present:
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


# Sanity check the round trip with special tokens embedded.
test = "<INS>GREET<RES>Hark, good morrow!<END>"
assert decode(encode(test)) == test
print(f"Round-trip check with special tokens: OK")


# --- Model definition (identical architecture to 10_tiny_gpt.py) -----------
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

    def forward(self, idx, targets=None, ignore_index=-100):
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                    ignore_index=ignore_index)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, stop_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and next_id.item() == stop_token:
                break
        return idx


block_size = 64
d_model, n_heads, n_layers, d_ff = 128, 4, 4, 512

# --- Load pretrained weights, extending vocab-dependent layers only --------
pretrained = TinyGPT(vocab_size=base_vocab_size, d_model=d_model, n_heads=n_heads,
                      n_layers=n_layers, d_ff=d_ff, block_size=block_size)
ckpt = torch.load("tiny_gpt_checkpoint.pt", weights_only=True)
pretrained.load_state_dict(ckpt["model_state"])
print(f"\nLoaded pretrained checkpoint at step {ckpt['total_steps']}.")

model = TinyGPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                 n_layers=n_layers, d_ff=d_ff, block_size=block_size)

# Copy EVERYTHING except the two vocab-sized layers directly.
pretrained_state = pretrained.state_dict()
model_state = model.state_dict()
for key in pretrained_state:
    if key in ("token_embedding.weight", "output_head.weight", "output_head.bias"):
        continue   # handle these specially below — sizes differ
    model_state[key] = pretrained_state[key]

# For the two vocab-sized layers: copy the OLD rows into the corresponding
# rows of the NEW (larger) tensor, leaving the new model's own random
# initialization in place for the 3 new special-token rows.
model_state["token_embedding.weight"][:base_vocab_size] = pretrained_state["token_embedding.weight"]
model_state["output_head.weight"][:base_vocab_size] = pretrained_state["output_head.weight"]
model_state["output_head.bias"][:base_vocab_size] = pretrained_state["output_head.bias"]
model.load_state_dict(model_state)

print(f"Extended model created: vocab {base_vocab_size} -> {vocab_size}. "
      f"Pretrained knowledge preserved for original {base_vocab_size} characters; "
      f"3 new special-token rows start randomly initialized.")


# --- Verify pretrained knowledge actually survived the extension ----------
def generate_text(model, prompt, max_new_tokens, temperature=0.8):
    idx = torch.tensor([encode(prompt, specials_present=False)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature)
    return decode(out[0].tolist())


torch.manual_seed(42)
print("\n--- Confirming pretrained knowledge survived (should look like trained Shakespeare) ---")
print(generate_text(model, "\nROMEO:", max_new_tokens=100))


# --- Build a synthetic instruction dataset ----------------------------------
# Deliberately a behavior raw Shakespeare text couldn't teach: respond to
# an exact instruction KEYWORD with a specific CATEGORY of response. This
# isolates SFT's effect cleanly — any success here is attributable to the
# fine-tuning data, not residual pretraining patterns.
training_pairs = [
    ("GREET", "Hark, good morrow to thee, friend!"),
    ("GREET", "Well met! Good day unto you."),
    ("GREET", "Good morrow, gentle sir!"),
    ("FAREWELL", "Fare thee well, until we meet again."),
    ("FAREWELL", "Farewell, and may fortune smile upon thee."),
    ("FAREWELL", "Adieu, good friend, adieu!"),
    ("INSULT", "Thou art a knave and a coward!"),
    ("INSULT", "Away, thou lump of foul deformity!"),
    ("INSULT", "Thou craven, spiritless villain!"),
]

examples = [f"<INS>{ins}<RES>{resp}<END>" for ins, resp in training_pairs]
print(f"\n--- {len(examples)} training examples ---")
for ex in examples[:3]:
    print(f"  {ex!r}")


# --- Loss masking: only train on the RESPONSE, not the instruction ---------
# We don't want the model learning to PREDICT the instruction text (it's
# given, not generated) — only to predict what comes AFTER <RES>. We build
# targets where every position up to and including <RES> is masked out
# with ignore_index=-100, which F.cross_entropy skips entirely.
def build_training_example(example_str, res_token_id, ignore_index=-100):
    ids = encode(example_str)
    input_ids = ids[:-1]
    target_ids = ids[1:]
    res_position = ids.index(res_token_id)   # index of <RES> in the FULL sequence
    masked_targets = [
        (t if i >= res_position else ignore_index)
        for i, t in enumerate(target_ids)
    ]
    return input_ids, masked_targets


res_id = token_to_id["<RES>"]
built = [build_training_example(ex, res_id) for ex in examples]

# Verify masking is correct on one example before trusting it in training.
sample_input, sample_target = built[0]
print(f"\n--- Verifying loss masking on: {examples[0]!r} ---")
for i, (inp, tgt) in enumerate(zip(sample_input, sample_target)):
    marker = "IGNORED" if tgt == -100 else f"predict {id_to_token[tgt]!r}"
    if i < 8 or tgt != -100:
        print(f"  input={id_to_token[inp]!r:8} -> target: {marker}")


# --- Fine-tune -----------------------------------------------------------
# Small dataset, so we train for many EPOCHS (full passes over the data)
# rather than sampling random batches — and use a LOWER learning rate
# than pretraining, since we're making a small, targeted adjustment to an
# already-capable model, not learning from scratch. This is standard SFT
# practice: pretraining is the expensive, broad phase; fine-tuning is
# cheap and narrow.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"\n--- Fine-tuning for 200 epochs over {len(built)} examples ---")
for epoch in range(200):
    total_loss = 0.0
    for input_ids, target_ids in built:
        x = torch.tensor([input_ids])
        y = torch.tensor([target_ids])
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d} | avg loss = {total_loss/len(built):.4f}")


# --- Before/after comparison ------------------------------------------------
def generate_response(model, instruction, max_new_tokens=40, temperature=0.5):
    prompt = f"<INS>{instruction}<RES>"
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature,
                          stop_token=token_to_id["<END>"])
    full = decode(out[0].tolist())
    return full[len(prompt):]   # just the generated response portion


print("\n--- After fine-tuning: instruction -> response ---")
for instruction in ["GREET", "FAREWELL", "INSULT"]:
    response = generate_response(model, instruction)
    print(f"  <INS>{instruction}<RES> -> {response!r}")

print("\n--- Confirming pretrained Shakespeare capability wasn't destroyed ---")
print(generate_text(model, "\nROMEO:", max_new_tokens=100))