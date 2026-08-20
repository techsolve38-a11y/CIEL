# Building CIEL From First Principles
### A course for an intermediate Python programmer, zero ML background

## How this course is structured

Five tracks, in order. Each track ends with something you *build*, not
just read about — you don't move on until the thing runs. Tracks A–C are
non-negotiable prerequisites for the from-scratch model (CIEL-0). Track
D is where you actually build CIEL-0. Track E is the software
architecture you already saw me scaffold — once you finish C and D,
you'll be able to read that scaffold and understand *why* every design
decision was made, not just what it does.

Rough total time investment for a working engineer studying part-time:
3–5 months to a working CIEL-0. That's not a discouragement — it's an
honest estimate so you can plan around it.

---

## TRACK A — Mathematical Foundations
*Goal: read and derive the math in a transformer paper without flinching.*

1. **Linear algebra**: vectors, matrices, matrix multiplication, dot
   product, transpose, the geometric meaning of a matrix as a
   transformation. This is the single most important math prerequisite —
   every layer in a neural network is matrix multiplication.
2. **Calculus for ML**: derivatives, partial derivatives, the chain rule.
   You need the chain rule cold — it's the entire mechanism behind
   backpropagation.
3. **Probability basics**: probability distributions, expected value,
   entropy, cross-entropy. Cross-entropy is the loss function language
   models are trained on — you need to understand *why* it's the right
   loss for "predict the next token."
4. **Gradient descent**: how you use a derivative to iteratively improve
   a guess. This is the optimization algorithm underneath all of deep
   learning.

**Build checkpoint**: implement gradient descent from scratch in numpy
to fit a line to noisy data (linear regression), with no ML library.
If you can write that loop and explain every line, Track A is done.

---

## TRACK B — Neural Networks From Scratch
*Goal: build and train a multi-layer neural network using only numpy — no PyTorch/TensorFlow.*

5. **The single neuron**: weighted sum + bias + activation function.
   Build one, feed it inputs by hand.
6. **The multi-layer perceptron (MLP)**: stacking neurons into layers,
   forward pass.
7. **Backpropagation**: the chain rule applied layer-by-layer to compute
   how every weight should change. This is the conceptual heart of deep
   learning — worth doing slowly.
8. **Loss functions and optimizers**: MSE, cross-entropy; SGD, momentum,
   Adam.
9. **Move to PyTorch**: once you've built it by hand, learn PyTorch as
   "the same thing, automated." This ordering matters — PyTorch feels
   like magic if you skip straight to it, and feels obvious if you don't.

**Build checkpoint**: train a numpy-only MLP to classify handwritten
digits (MNIST), then rebuild the exact same model in PyTorch in <30
lines and confirm you get similar accuracy.

---

## TRACK C — Language Models & Attention
*Goal: understand and build a transformer from scratch.*

10. **Tokenization**: how text becomes numbers. Build a simple
    byte-pair-encoding (BPE) tokenizer from scratch.
11. **Embeddings**: how token IDs become vectors that capture meaning.
12. **Why RNNs weren't enough** (brief, historical — you need this to
    appreciate *why* attention was invented, not to build an RNN).
13. **Self-attention**: the core mechanism — build scaled dot-product
    attention from scratch in numpy/PyTorch, by hand, before using any
    built-in attention layer.
14. **The full transformer block**: multi-head attention, feedforward
    layers, layer normalization, residual connections, positional
    encoding.
15. **Build a tiny GPT**: assemble the above into a small decoder-only
    transformer and train it on a toy dataset (e.g. Shakespeare's text,
    character-level) until it generates plausible-looking text.

**Build checkpoint**: this is Andrej Karpathy's "nanoGPT / build GPT
from scratch" exercise — a several-hundred-line model that trains on a
single GPU (or even CPU, slowly) and demonstrably learns to generate
text. This is the single highest-leverage exercise in the entire
curriculum. Do not skip it or substitute a tutorial that lets a library
build the transformer for you.

---

## TRACK D — Training CIEL-0 For Real
*Goal: apply Track C at the scale needed for a usable (not powerful) CIEL-0.*

16. **Real tokenizer training**: train a BPE tokenizer on a real corpus
    (not toy data), understand vocabulary size trade-offs.
17. **Dataset pipeline**: sourcing, cleaning, deduplication, formatting
    text data into training-ready shards.
18. **Training infrastructure**: batching, gradient accumulation, mixed
    precision, checkpointing, resuming — the unglamorous engineering
    that makes a training run survivable.
19. **Evaluation**: perplexity, held-out loss, qualitative sampling —
    how you know if the model is actually improving.
20. **Instruction tuning**: supervised fine-tuning (SFT) on
    instruction/response pairs so the base model becomes conversational.
    (RLHF/DPO-style preference tuning is worth knowing about
    conceptually but is not necessary for a first CIEL-0.)

**Build checkpoint**: this *is* CIEL-0 — a small model with its own
tokenizer, trained from scratch on data you curated, that can hold a
basic instruction-following conversation. It will not be smart. That's
correct and expected per the Foundation Spec — its job at this stage is
to prove the pipeline, not to be capable.

---

## TRACK E — The Systems Architecture (you have a head start)

Once C and D are done, go back and read every file in the scaffold I
already built — `ciel/constitution/`, `ciel/memory/`, `ciel/skills/`,
`ciel/orchestrator/`. At that point you'll understand not just *what*
each module does but *why* it's shaped the way it is — e.g. why the
orchestrator calls the model behind a single function boundary
(`Orchestrator.process`), so CIEL-0 slots in without touching anything
else. Then swap the interim Claude call for your own CIEL-0.

---

## Suggested resources per track (starting points, not exhaustive)

- **A**: 3Blue1Brown's "Essence of Linear Algebra" and "Essence of
  Calculus" video series — intuition-first, exactly what's needed here.
- **B**: Michael Nielsen's *Neural Networks and Deep Learning* (free
  online book) — builds an MLP from scratch in the way this course
  wants.
- **C**: Andrej Karpathy's "Neural Networks: Zero to Hero" video series,
  specifically the "Let's build GPT from scratch" video — this is
  Track C's spine.
- **D**: Karpathy's `nanoGPT` repo as a reference implementation once
  you've built your own — compare, don't copy first.

---

## How we'll use this together

Tell me when you want to start a session on a specific numbered item
(e.g. "let's do #7, backpropagation") and I'll teach it properly —
derive the math, write the code with you, and make sure you can explain
it back before we move on. I won't hand you finished code to copy;
you'll write it, I'll check it and explain the parts that don't click.
