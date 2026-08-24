"""
Embeddings — Track C, item 11
==================================

Goal: understand embeddings not as a black box, but as exactly what they
are — a matrix of learnable numbers, one row per vocabulary token, looked
up by token ID, and trained by ordinary gradient descent like everything
else you've built. We'll prove they organize themselves meaningfully by
training a tiny next-token predictor and inspecting what the embeddings
learn.

Why not just feed raw token IDs into the network?
  A token ID is an arbitrary index. Token 23 isn't "more similar" to
  token 24 than to token 5 — there's no meaningful notion of distance or
  relationship in raw IDs. An embedding REPLACES the ID with a vector
  (e.g. 8 numbers) that IS learned to have meaningful geometry: tokens
  that behave similarly in the training data end up with similar
  vectors, purely as a side effect of gradient descent trying to
  minimize prediction error.
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(7)

# --- Reuse the BPE tokenizer from last session ------------------------------
corpus = """
the quick brown fox jumps over the lazy dog
the dog barks at the fox
the fox runs into the forest
a quick fox and a lazy dog become friends
"""


def get_pair_counts(words):
    counts = {}
    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(words, pair):
    merged_symbol = pair[0] + pair[1]
    new_words = []
    for word in words:
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_words.append(new_word)
    return new_words


def train_bpe(words, n_merges):
    words = [list(w) for w in words]
    merges = []
    for _ in range(n_merges):
        pair_counts = get_pair_counts(words)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        words = merge_pair(words, best_pair)
        merges.append(best_pair)
    return words, merges


words = [list(w) for w in corpus.split(" ") if w]
_, merges = train_bpe(words, 15)
vocab_list = sorted(set(corpus) | set(a + b for a, b in merges))
token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
id_to_token = {i: tok for tok, i in token_to_id.items()}
vocab_size = len(vocab_list)


def encode(text, merges, token_to_id):
    all_ids = []
    text_words = text.split(" ")
    for wi, w in enumerate(text_words):
        tokens = list(w)
        for a, b in merges:
            merged = a + b
            new_tokens, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        for tok in tokens:
            all_ids.append(token_to_id.get(tok, token_to_id.get(tok[0]) if tok else 0))
        if wi < len(text_words) - 1:
            all_ids.append(token_to_id[" "])
    return all_ids


print(f"Vocabulary size: {vocab_size}")

# --- The embedding table itself ---------------------------------------------
embedding_dim = 8
embedding_table = nn.Embedding(vocab_size, embedding_dim)

# This IS the whole idea: a (vocab_size x embedding_dim) matrix. Row i is
# token i's vector. Looking up a token is just indexing into this matrix
# — nothing more exotic than that.
print(f"\nEmbedding table shape: {tuple(embedding_table.weight.shape)}  "
      f"({vocab_size} tokens x {embedding_dim} dimensions each)")

the_id = token_to_id["the"]
fox_id = token_to_id["fox"]
print(f"\n'the' -> id {the_id} -> embedding (untrained, random): "
      f"{embedding_table.weight[the_id].detach().numpy().round(3)}")
print(f"'fox' -> id {fox_id} -> embedding (untrained, random): "
      f"{embedding_table.weight[fox_id].detach().numpy().round(3)}")
print("\nRight now these are just random numbers — no meaning yet. "
      "Training is what gives them structure.")


# --- A task that forces the embeddings to become meaningful -----------------
# Task: given token i, predict token i+1. This is the simplest possible
# language modeling objective — "what comes next" — and it's enough to
# force useful structure into the embeddings, because tokens that tend to
# PRECEDE similar next-tokens will get pushed toward similar vectors by
# gradient descent (this is the actual mechanism, not a metaphor).
full_text = corpus.replace("\n", " ").strip()
token_ids = encode(full_text, merges, token_to_id)
token_ids = [i for i in token_ids if i is not None]

inputs = torch.tensor(token_ids[:-1])
targets = torch.tensor(token_ids[1:])
print(f"\nTraining pairs: {len(inputs)} (token[i] -> token[i+1])")


class NextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = embedding_table          # SAME embedding table from above
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids):
        vectors = self.embedding(token_ids)        # (batch, embedding_dim)
        logits = self.output_layer(vectors)         # (batch, vocab_size) — raw scores per possible next token
        return logits


model = NextTokenPredictor(vocab_size, embedding_dim)
loss_fn = nn.CrossEntropyLoss()   # multi-class version of the binary cross-entropy from before
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print("\n--- Training next-token predictor ---")
for step in range(500):
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    optimizer.zero_grad()
    loss.backward()   # gradients flow all the way back into the embedding table too
    optimizer.step()
    if step % 50 == 0 or step == 499:
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == targets).float().mean().item()
        print(f"step {step:4d} | loss={loss.item():.4f} | next-token accuracy={accuracy:.3f}")


# --- Inspect what the embeddings learned ------------------------------------
def cosine_similarity(a, b):
    """Measures how aligned two vectors are, independent of their magnitude:
    1.0 = pointing the same direction, 0.0 = unrelated, -1.0 = opposite."""
    a, b = a.detach().numpy(), b.detach().numpy()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def emb(token):
    return embedding_table.weight[token_to_id[token]]


print("\n--- Cosine similarity between trained embeddings ---")
pairs_to_check = [
    ("fox", "dog"),       # both animals, both often appear after "the"/"lazy"/"quick"
    ("quick", "lazy"),    # both adjectives, both often precede an animal word
    ("the", "a"),         # both articles/determiners
    ("fox", "the"),       # unrelated grammatical roles — expect lower similarity
]
for tok_a, tok_b in pairs_to_check:
    sim = cosine_similarity(emb(tok_a), emb(tok_b))
    print(f"  similarity('{tok_a}', '{tok_b}') = {sim:.3f}")

print("\nCaveat: this corpus is tiny (a few sentences), so don't expect clean, "
      "textbook word-analogy behavior — the point is that the mechanism runs "
      "and produces SOME structure, purely from next-token prediction, with "
      "no one telling the model what a noun or adjective is. At real scale "
      "(billions of words), this exact mechanism is what produces embeddings "
      "where 'king' - 'man' + 'woman' lands near 'queen'.")