# MoE Routing Project — Handoff (Resume Here)

**Purpose:** read this first when picking the MoE routing project back up in a
new session — it's the short "where things stand" pointer. Full derivation
trail, sourced numbers, and the reasoning behind every choice lives in
`notes.md`; don't re-derive anything already logged there.

**Status as of this handoff:** Phase 1 fully complete (1a workload shape, 1b
uniform comms volume, 1c load-imbalance modeling). **Next up: Phase 2 —
predict the ideal system architecture.** Phase 0 (ASTRA-sim setup) is a
separate, non-blocking tooling task, still not executed.

---

## Start here: Phase 2 — predict the ideal system architecture

Per `spec.md`, Phase 2 asks for a hand-derived hypothesis
(not yet validated in a tool — that's Phase 3) covering:
- **Topology**: given the dispatch/combine communication pattern
  (data-dependent, all-to-all-ish), what interconnect topology fits, and
  why — grounded in which properties of *this specific* pattern (bisection
  bandwidth? latency? worst-case contention under imbalance?) matter most.
- **Bandwidth allocation**: uniform across links, or does anything from
  Phase 1c argue for something else (e.g. extra headroom for popular-expert
  links)?
- **Buffering/SRAM**: how much on-chip buffering does a chip need to avoid
  stalling under the imbalance model, given Phase 1's volumes?

**The one Phase 1 result that should directly shape this phase's
hypothesis**: imbalance turned out to barely move the needle (headroom only
eroded from 6.7× to ~5× across the full uniform-to-99%-stall range) — a
hard, imbalance-proof floor exists in the FLOPs/byte ratio regardless of
routing skew. That's a real reason to walk into Phase 2 *skeptical* of
exotic imbalance-driven topology/bandwidth choices (e.g. reserving extra
capacity for hot-expert links) rather than assuming they're obviously
worth it — test that skepticism explicitly rather than importing it as a
given. The lever that actually mattered in Phase 1 was **dispatch
precision** (crossover to comms-bound at ≈2.5 bytes/element) — worth
keeping in mind if Phase 2's reasoning brushes up against numerics at all.

---

## Collaboration mode — read before responding to anything

Same self-directed-learning constraint as the attention project (see its own
`../attention/handoff.md` for the fuller version): check reasoning, ask
questions that expose gaps, don't hand over derivations the user hasn't
produced themselves.

**Exceptions the user has explicitly invoked in this project:**
- Pure arithmetic plug-in once the user has stated the formula — fine to
  hand over, don't make them be a human calculator.
- Factual/reference lookups (hardware specs, paper equations, imbalance
  literature) — verify via search, cite the source, don't trust memory.
  Several Phase 1c numbers were found this way, including one correction of
  a previously-logged wrong assumption (see Phase 1 results below).
- Chip/precision *selection* specifically — "just recommend it directly,"
  Socratic pushback isn't wanted there.
- Watch calibration generally: don't let a sub-derivation go deeper than its
  actual role warrants (happened once with the attention-weight footprint
  under MLA/TP in Phase 1a). Also watch for accidentally resolving an open
  question the user explicitly wants to work out themselves, even when
  answering an adjacent factual-recall request — happened once in Phase 1c
  (asked to recall formulas, also answered the still-open question of where
  the imbalance multiplier applies) and the user caught it.

---

## Phase 1 results (full derivation in `notes.md`)

**1b (uniform/ideal case)**: 105 MiB/layer dispatch+combine, whole system,
one decode step. ≈28,087 FLOPs/byte vs. a TPU 8i ICI ridge point of
≈4,208 — decisively compute-bound, 6.7× margin.

**1c (load imbalance)**: three-point range (uniform / CF=1.0 dropping-on /
uncapped dropping-off), grounded in two real cited sources (Gini≈0.70
across DeepSeek-V3/Qwen3-MoE/Mixtral; 70% GPU stall time on real
Mixtral-8×7B serving → implies a 3.3× device-load multiplier). **All three
land compute-bound.** A structural floor of ≈21,065 FLOPs/byte exists
regardless of imbalance severity — no amount of routing skew can flip this
workload comms-bound on this hardware at FP4. The lever that *can* flip it
is dispatch precision (crossover ≈2.5 bytes/element, between FP8 and
BF16).

One correction worth knowing: an earlier logged assumption
("token-dropping is training-only") was **wrong**, caught by re-reading
the actual paper text — DeepSeek-V2 explicitly supports optional
inference-time dropping at CF=1.0, device-level, lowest-affinity-first.

---

## Phase 0: ASTRA-sim still not built (separate, non-blocking)

Plan (see `notes.md` § Phase 0 status for full reasoning): local
Docker build on the Mac first (repo's Dockerfile is unpinned
`ubuntu:22.04`, should build natively on arm64). Checkpoint is running the
stock all-reduce example. Farmshare is fallback only, not default. **Still
not attempted — confirm with the user whether it's been done outside this
chat.** Not needed until Phase 3 (simulation) — doesn't block Phase 2's
hand-derivation, so no rush to interleave it.

---

## Where everything lives

- `spec.md` — original spec, all phases.
- `notes.md` — full derivation trail for 1a/1b/1c, Prediction/Log
  style. Read it for the "why," not just the "what." Keep appending here
  for Phase 2.
- This file — update/overwrite at the next natural pause point.
- Sibling project: `../attention/spec.md` / `../attention/notes.md` /
  `../attention/handoff.md` — the attention project this one extends.
  Phase 4 of this spec asks for a synthesis connecting the two.
