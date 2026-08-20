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
