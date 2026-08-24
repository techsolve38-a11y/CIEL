"""
Byte-Pair Encoding Tokenizer — Track C, item 10
====================================================

Goal: convert text into a sequence of integers a network can process,
using a vocabulary that's learned FROM DATA rather than hand-designed.

Why not just use characters directly?
  - Vocabulary is tiny (~100 symbols) and never has "unknown word" problems.
  - BUT: sequences become very long (one token per character), and each
    individual token carries almost no meaning on its own — the network
    has to work much harder to build up "word-level" understanding from
    scratch, layer by layer.

Why not just use whole words?
  - Each token carries lots of meaning, sequences are short.
  - BUT: vocabulary must be huge to cover a language, AND you'll still
    hit words at inference time that never appeared in training data —
    the "out of vocabulary" problem, with no good fallback.

BPE's answer: start at the character level (no OOV problem — anything
can be spelled out character by character in the worst case), then
iteratively merge the most frequently-occurring ADJACENT PAIR of symbols
into a new single symbol. Repeat many times. Common words end up as a
single token (learned automatically, because their component pairs
occur often); rare words fall back to being spelled out in smaller
pieces. This is exactly the algorithm behind GPT's and Claude's
tokenizers — what you're about to build isn't a simplified toy version,
it's the real algorithm, just run on a much smaller corpus.
"""

# A small corpus. Real tokenizers train on gigabytes of text; a few
# sentences is enough to see the algorithm work and understand it, but
# not enough to build a genuinely useful vocabulary yet — that's fine,
# understanding the mechanism is this step's goal, not covering English.
corpus = """
the quick brown fox jumps over the lazy dog
the dog barks at the fox
the fox runs into the forest
a quick fox and a lazy dog become friends
"""

print(f"Corpus length: {len(corpus)} characters")
print(f"Unique characters: {sorted(set(corpus))}")


# --- Step 1: Represent the corpus as a list of "words," each a list of ------
# individual characters. We keep word boundaries (spaces) as their own
# tokens rather than merging across them — this is a simplification real
# tokenizers also broadly follow (they don't usually merge across words).
words = corpus.split(" ")
words = [list(w) for w in words if w]   # each word -> list of its characters
print(f"\nFirst 3 words as character lists: {words[:3]}")


# --- Step 2: Count all adjacent PAIRS of symbols across the whole corpus ---
def get_pair_counts(words):
    """Count how often each adjacent (symbol, symbol) pair occurs, across
    every word. This is the frequency signal BPE uses to decide what to
    merge next."""
    counts = {}
    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


pair_counts = get_pair_counts(words)
most_common_pair = max(pair_counts, key=pair_counts.get)
print(f"\nMost frequent adjacent pair: {most_common_pair} "
      f"(appears {pair_counts[most_common_pair]} times)")


# --- Step 3: Merge the most frequent pair everywhere it occurs -------------
def merge_pair(words, pair):
    """Replace every occurrence of `pair` (two adjacent symbols) with a
    single new merged symbol, across every word."""
    merged_symbol = pair[0] + pair[1]
    new_words = []
    for word in words:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_words.append(new_word)
    return new_words


words_after_one_merge = merge_pair(words, most_common_pair)
print(f"First 3 words after ONE merge: {words_after_one_merge[:3]}")


# --- Step 4: Repeat — this IS training a BPE tokenizer ---------------------
# Each iteration: find the most frequent pair, merge it everywhere, record
# the merge rule. The list of recorded merge rules, IN ORDER, is the
# trained tokenizer — it's the only thing we need to save to reproduce
# this exact tokenization later.
def train_bpe(words, n_merges):
    words = [list(w) for w in words]   # don't mutate the caller's data
    merges = []   # ordered list of (pair, merged_symbol) — this IS the "model"
    for i in range(n_merges):
        pair_counts = get_pair_counts(words)
        if not pair_counts:
            break   # no more pairs left to merge (fully merged already)
        best_pair = max(pair_counts, key=pair_counts.get)
        words = merge_pair(words, best_pair)
        merges.append(best_pair)
    return words, merges


words = [list(w) for w in corpus.split(" ") if w]
n_merges = 15
final_words, merges = train_bpe(words, n_merges)

print(f"\n--- Trained {len(merges)} merges ---")
for i, (a, b) in enumerate(merges):
    print(f"  merge {i+1}: '{a}' + '{b}' -> '{a+b}'")

print(f"\nFinal tokenization of first 5 words: {final_words[:5]}")

vocab = set()
for word in final_words:
    vocab.update(word)
print(f"\nFinal vocabulary size: {len(vocab)} (started at 28 characters, "
      f"added {len(merges)} merged subword tokens)")


# --- Step 5: Build a usable tokenizer — vocab, encode, decode --------------
# Assign every distinct symbol (character or merged subword) an integer ID.
# This ID list is what actually gets fed into a neural network — the
# network never sees text, only these integers, looked up into embedding
# vectors (that's next session's topic).
vocab_list = sorted(set(corpus) | set(a + b for a, b in merges))
# BUG FIX #2: vocab must include every BASE character that ever appeared
# in the corpus (set(corpus) — this also covers space, since we're using
# the raw corpus string here, not the space-stripped `words`), PLUS every
# symbol PRODUCED by a merge rule, even intermediate ones that got
# consumed by a later merge (e.g. 'th' got folded into 'the' and no
# longer appears standalone in the training output — but it's still a
# valid, reachable token, and dropping it would make the tokenizer
# fail on any new text where 'th' occurs without a following 'e').
token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
id_to_token = {i: tok for tok, i in token_to_id.items()}


def encode(text, merges, token_to_id):
    """Split on spaces (word boundaries), apply learned merges within each
    word, then stitch words back together with explicit space tokens."""
    all_ids = []
    text_words = text.split(" ")
    for wi, w in enumerate(text_words):
        tokens = list(w)
        for a, b in merges:
            merged = a + b
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        for tok in tokens:
            if tok in token_to_id:
                all_ids.append(token_to_id[tok])
            else:
                for ch in tok:
                    all_ids.append(token_to_id.get(ch, -1))
        if wi < len(text_words) - 1:
            all_ids.append(token_to_id[" "])   # word boundary
    return all_ids


def decode(ids, id_to_token):
    return "".join(id_to_token[i] for i in ids)


test_text = "the quick fox"
encoded = encode(test_text, merges, token_to_id)
decoded = decode(encoded, id_to_token)

print(f"\n--- Encode/decode round trip ---")
print(f"Original:  {test_text!r}")
print(f"Token IDs: {encoded}")
print(f"Decoded:   {decoded!r}")
print(f"Round trip lossless: {decoded == test_text}")

# Try text with a word NOT in the training corpus, to see the graceful
# fallback in action.
novel_text = "the sleepy cat"
encoded_novel = encode(novel_text, merges, token_to_id)
decoded_novel = decode(encoded_novel, id_to_token)
print(f"\n--- Novel text (contains unseen words) ---")
print(f"Original:  {novel_text!r}")
print(f"Token IDs: {encoded_novel}")
print(f"Decoded:   {decoded_novel!r}")
print(f"Round trip lossless: {decoded_novel == novel_text}")
print("Notice: 'the' compresses to one token (it was learned), while "
      "'sleepy' and 'cat' fall back to individual characters (never seen) — "
      "but decoding still perfectly reconstructs the original text either way.")