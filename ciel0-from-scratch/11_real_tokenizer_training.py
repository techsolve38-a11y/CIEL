"""
Real Tokenizer Training — Track D, item 16
================================================

Goal: train BPE on REAL data (not four toy sentences from item 10), and
directly measure the trade-off every production tokenizer has to
navigate: MORE merges -> bigger vocabulary, but SHORTER token sequences
for the same text. Real tokenizers (GPT-2/3/4, Claude) use tens of
thousands of merges precisely because they've found where this trade-off
pays off at their scale — we'll see the same shape of trade-off here, at
a scale we can actually train on CPU in reasonable time.

Why does the trade-off matter practically? Every token costs compute —
attention is roughly quadratic in sequence length, so a tokenizer that
halves your average sequence length roughly quarters attention's compute
cost for the same text. But push vocabulary too large and you're spending
more parameters just on the embedding table and output head, with
diminishing returns on very rare subwords that barely ever appear.
"""

import time

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Use a 200K-character slice for TRAINING the tokenizer (large enough to
# see real English statistics, small enough to train in reasonable time
# with our pure-Python implementation — production tokenizers train on
# gigabytes, using much faster implementations than what we're building
# by hand here).
train_text = full_text[:200_000]
# Hold out a DIFFERENT slice to test compression on text the tokenizer
# didn't train on — this matters, same reasoning as train/val splits
# elsewhere in this course.
test_text = full_text[800_000:820_000]

print(f"Training on {len(train_text):,} characters, testing on {len(test_text):,} characters (held out).")


def get_pair_counts(words):
    counts = {}
    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(words, pair):
    merged = pair[0] + pair[1]
    new_words = []
    for word in words:
        nw, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                nw.append(merged)
                i += 2
            else:
                nw.append(word[i])
                i += 1
        new_words.append(nw)
    return new_words


def train_bpe(words, n_merges, log_every=100):
    words = [list(w) for w in words]
    merges = []
    start = time.time()
    for i in range(n_merges):
        pair_counts = get_pair_counts(words)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        words = merge_pair(words, best_pair)
        merges.append(best_pair)
        if (i + 1) % log_every == 0:
            print(f"  merge {i+1}/{n_merges} | {time.time()-start:.1f}s elapsed")
    return merges


train_words = [w for w in train_text.split(" ") if w]
N_MERGES = 500
print(f"\n--- Training BPE, {N_MERGES} merges ---")
merges = train_bpe(train_words, N_MERGES)
print(f"Done. Learned {len(merges)} merges.")


# --- Measure the trade-off: vocab size vs sequence compression -------------
def encode_with_n_merges(text, all_merges, n, base_vocab):
    """Apply only the FIRST n merges (in learned order) to text — lets us
    see what compression looks like at various points along training,
    without retraining from scratch each time."""
    active_merges = all_merges[:n]
    words = text.split(" ")
    total_tokens = 0
    vocab = set(base_vocab)
    for w in words:
        tokens = list(w)
        for a, b in active_merges:
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
        total_tokens += len(tokens)
        vocab.update(tokens)
    return total_tokens, vocab


base_vocab = set(train_text)
n_test_chars = len(test_text.replace(" ", ""))   # exclude spaces from the character count for a fair ratio

print(f"\n--- Vocab size vs compression trade-off (measured on held-out text) ---")
print(f"{'merges':>8} | {'vocab size':>10} | {'tokens (test)':>14} | {'chars/token':>12}")
checkpoints = [0, 50, 100, 200, 300, 500]
for n in checkpoints:
    n_tokens, vocab = encode_with_n_merges(test_text, merges, n, base_vocab)
    compression = n_test_chars / n_tokens
    print(f"{n:>8} | {len(vocab):>10} | {n_tokens:>14} | {compression:>12.2f}")

print(f"\nAt 0 merges (character-level): every character is its own token — "
      f"vocab is tiny, but sequences are as long as the raw text.")
print(f"At 500 merges: vocab has grown, but each token now represents "
      f"~{n_test_chars / encode_with_n_merges(test_text, merges, 500, base_vocab)[0]:.1f} "
      f"characters on average — meaningfully shorter sequences for the SAME text.")
print(f"\nFor reference: GPT-2's real tokenizer uses ~50,000 merges on a corpus of "
      f"billions of words — same trade-off, pushed much further because the training "
      f"data and compute budget justify a much larger vocabulary.")