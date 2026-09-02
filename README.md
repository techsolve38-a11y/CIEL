# CIEL

A personal intelligence, sovereignty & legacy system — see
`docs/CIEL_Foundation_Specification.docx` and
`docs/CIEL_Three_Phase_Development_Plan.docx` for the full design spec.

This repo has two parts, deliberately kept separate:

## `system/`
The Phase I software architecture — constitution, memory, user model,
skills, tools, orchestrator, evaluation. Runs today using an existing
LLM (Claude) as the interim reasoning engine, behind a single function
boundary (`Orchestrator.process`) so it can be swapped for a from-scratch
model without touching anything else. See `system/README.md` for details
and setup instructions.

## `ciel0-from-scratch/`
The learning + build track for CIEL-0, the eventual from-scratch language
model, following `docs/CIEL_Learning_Curriculum.md`. Numbered scripts,
in build order, each one a working checkpoint (not a finished library) —
this is deliberately built by hand, not imported from a package, because
the point is understanding the internals well enough to design CIEL-0's
architecture, not just to run one.

Current progress: `01_gradient_descent.py` — gradient descent from
scratch in numpy (Track A checkpoint).

## Status

Early. `system/` runs end-to-end against Claude as the interim brain.
`ciel0-from-scratch/` has just started (Track A of the curriculum).

Information About Track 7
Why RNNs weren't enough. Before attention, the standard approach to sequences was the Recurrent Neural Network: process tokens one at a time, left to right, carrying a single "hidden state" vector forward as a summary of everything seen so far. Two structural problems fall out of that design:

Sequential bottleneck. Token 50 can only be processed after token 49 is done, which is done only after token 48 — no parallelism across the sequence. On modern hardware built for massively parallel matrix multiplication, this is a severe waste, and it's a huge part of why training used to take so much longer per token of data.
Information gets squeezed through one vector. Everything the model has seen — potentially thousands of tokens back — has to be compressed into that single fixed-size hidden state before token 50 can use it. Early information reliably gets diluted or overwritten by the time you're deep into a long sequence — this is the "long-range dependency" problem, and it's the direct motivation for attention.