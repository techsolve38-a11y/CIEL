"""
The Data Pipeline — Track D, item 17
=========================================

Real training corpora don't arrive clean. They're scraped from the web
(or assembled from many messy sources), and come with: exact duplicates
(the same page mirrored/crawled twice), near-duplicates (boilerplate,
templated pages), junk fragments (HTML remnants, encoding errors, ads),
and wildly inconsistent formatting. A model trained on uncleaned data
wastes capacity memorizing duplicated text and learning from garbage.

We'll deliberately corrupt our clean Shakespeare text to simulate this
realistically, then build the three pipeline stages that fix it:
  1. Cleaning      — normalize formatting, filter out garbage documents
  2. Deduplication — remove exact AND near-duplicate documents
  3. Sharding      — split the cleaned result into multiple files, the
                      way real training pipelines do at scale
"""

import hashlib
import random

random.seed(42)

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into "documents" — paragraphs, roughly, the way a real pipeline
# would treat separate scraped pages as separate documents.
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
print(f"Starting with {len(paragraphs)} clean paragraphs from Shakespeare.")

# --- Deliberately corrupt the data to simulate a real messy scrape --------
documents = list(paragraphs[:2000])   # take a manageable subset

# 1. Inject EXACT duplicates (simulating re-crawled/mirrored pages)
duplicates = random.sample(documents, 300)
documents.extend(duplicates)

# 2. Inject NEAR-duplicates (same content, trivial formatting differences —
#    simulating templated pages or reformatted mirrors)
near_dup_sources = random.sample(paragraphs[:2000], 150)
for doc in near_dup_sources:
    corrupted = doc.replace(" ", "  ")   # double spaces — same content, different formatting
    documents.append(corrupted)

# 3. Inject junk documents (HTML remnants, garbage — simulating scraping artifacts)
junk_samples = [
    "<div class='ad'>CLICK HERE FOR MORE</div>",
    "###### 404 NOT FOUND ######",
    "asdkjhaskjdh aksjdh aksjdhaksjdh",
    "",
    "   ",
    "a",
    "!!!!!!!!!!!!!!!!!!!!!!",
]
documents.extend(junk_samples * 20)   # junk tends to repeat a lot in real scrapes too

# 4. Inject GENUINE near-duplicates: shared boilerplate appended to otherwise
#    distinct documents — this is what templated/mirrored pages actually
#    look like, and (unlike our first near-dup attempt) this survives
#    whitespace normalization, so it needs REAL near-duplicate detection.
boilerplate = " Visit our archive for more classic works."
boilerplate_sources = random.sample(paragraphs[:2000], 100)
for doc in boilerplate_sources:
    documents.append(doc + boilerplate)

random.shuffle(documents)
print(f"After corrupting: {len(documents)} documents "
      f"(added {len(duplicates)} exact dupes, {len(near_dup_sources)} weak near-dupes, "
      f"{len(boilerplate_sources)} genuine near-dupes via shared boilerplate, "
      f"{len(junk_samples)*20} junk fragments).")


# --- Stage 1: Cleaning -------------------------------------------------------
def normalize_whitespace(doc):
    """Collapse any run of whitespace (multiple spaces, tabs, etc.) into a
    single space, and strip leading/trailing whitespace. This alone fixes
    our injected near-duplicates' formatting difference."""
    return " ".join(doc.split())


def is_low_quality(doc, min_length=20, min_alpha_ratio=0.6):
    """Heuristic quality filter — this is exactly the kind of rule real
    pipelines use (simplified): too short, or too few actual letters
    relative to symbols/punctuation, is a strong signal of junk rather
    than genuine prose. Not perfect — no heuristic is — but catches most
    of what we injected without needing to hand-list every junk pattern."""
    if len(doc) < min_length:
        return True
    alpha_count = sum(c.isalpha() or c.isspace() for c in doc)
    alpha_ratio = alpha_count / len(doc)
    return alpha_ratio < min_alpha_ratio


cleaned = []
removed_low_quality = 0
for doc in documents:
    normalized = normalize_whitespace(doc)
    if is_low_quality(normalized):
        removed_low_quality += 1
        continue
    cleaned.append(normalized)

print(f"\n--- After cleaning ---")
print(f"Removed {removed_low_quality} low-quality/junk documents.")
print(f"Remaining: {len(cleaned)} documents.")


# --- Stage 2a: Exact deduplication via hashing ------------------------------
# Hash each document's content; if we've seen this exact hash before, it's
# an exact duplicate. Hashing (rather than comparing full strings) is what
# makes this practical at scale — comparing every document to every other
# document directly would be far too slow for millions of documents.
def exact_dedupe(docs):
    seen_hashes = set()
    unique = []
    for doc in docs:
        h = hashlib.sha256(doc.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(doc)
    return unique


after_exact_dedup = exact_dedupe(cleaned)
print(f"\n--- After exact deduplication ---")
print(f"Removed {len(cleaned) - len(after_exact_dedup)} exact duplicates.")
print(f"Remaining: {len(after_exact_dedup)} documents.")


# --- Stage 2b: Near-duplicate detection via shingling + Jaccard similarity -
# A "shingle" is a small sliding window of words (here, 5-word chunks).
# Two documents that share MANY shingles are very likely near-duplicates,
# even if they're not byte-identical — this is exactly what catches our
# boilerplate-footer case, which exact hashing cannot.
def get_shingles(doc, k=5):
    words = doc.split()
    if len(words) < k:
        return {doc}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def near_dedupe(docs, threshold=0.5):
    """O(n^2) comparison — fine for a few thousand documents in a
    teaching exercise. Real pipelines use approximate techniques
    (MinHash + LSH) specifically to avoid this quadratic cost at
    billions-of-documents scale — worth knowing the naive version exists
    conceptually before reaching for the approximate one."""
    shingle_sets = [get_shingles(d) for d in docs]
    keep = [True] * len(docs)
    for i in range(len(docs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(docs)):
            if not keep[j]:
                continue
            if jaccard_similarity(shingle_sets[i], shingle_sets[j]) >= threshold:
                keep[j] = False   # drop the later one, keep the first occurrence
    return [d for d, k in zip(docs, keep) if k]


after_near_dedup = near_dedupe(after_exact_dedup, threshold=0.5)
print(f"\n--- After near-duplicate detection (Jaccard >= 0.5 on 5-word shingles) ---")
print(f"Removed {len(after_exact_dedup) - len(after_near_dedup)} near-duplicates.")
print(f"Remaining: {len(after_near_dedup)} documents.")


# --- Stage 3: Sharding -------------------------------------------------------
# Why split into multiple files instead of one big cleaned file?
#   - At real scale (many GB-TB), a single file can't fit in memory —
#     shards let training code load and release chunks one at a time.
#   - Shuffling: shuffling within and across many small shards approximates
#     a full random shuffle without needing the entire dataset in memory
#     simultaneously.
#   - Fault tolerance: if one shard is corrupted or lost, you lose a small
#     fraction of data, not the whole training set.
#   - Parallelism: multiple data-loading workers can each own different
#     shards, reading simultaneously rather than contending for one file.
import os

shard_dir = "shards"
os.makedirs(shard_dir, exist_ok=True)
n_shards = 5
shard_size = len(after_near_dedup) // n_shards + 1

for i in range(n_shards):
    shard_docs = after_near_dedup[i * shard_size: (i + 1) * shard_size]
    if not shard_docs:
        continue
    path = os.path.join(shard_dir, f"shard_{i:03d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(shard_docs))
    print(f"Wrote {path}: {len(shard_docs)} documents")

print(f"\n--- Pipeline summary ---")
print(f"Started:              {len(documents)} documents (post-corruption)")
print(f"After cleaning:       {len(cleaned)} documents")
print(f"After exact dedup:    {len(after_exact_dedup)} documents")
print(f"After near-dedup:     {len(after_near_dedup)} documents")
print(f"Final shards:         {n_shards} files in {shard_dir}/")