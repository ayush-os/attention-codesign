# MoE Routing Project — Handoff (Resume Here)

**Purpose:** read this first when picking the MoE routing project back up in a
new session — it's the short "where things stand" pointer. Full derivation
trail, sourced numbers, and the reasoning behind every choice lives in
`moe_phase1_notes.md`; don't re-derive anything already logged there.

**Status as of this handoff:** Phase 1a (workload shape) and Phase 1b
(uniform-routing comms volume + compute-to-comms ratio) complete. Phase 0
(ASTRA-sim setup) decided but **not yet executed**. Phase 1c (load-imbalance
modeling) not started — that's the next real work.

---

## Collaboration mode — read before responding to anything

Same self-directed-learning constraint as the attention project (see its own
`handoff.md` for the fuller version): check reasoning, ask questions that
expose gaps, don't hand over derivations the user hasn't produced themselves.

**Exceptions the user has explicitly invoked in this project:**
- Pure arithmetic plug-in once the user has stated the formula — user
  explicitly said this is fine to hand over, don't make them be a human
  calculator.
- Factual/reference lookups (hardware specs, paper equations) — verify via
  search, cite the source, don't trust memory.
- Chip/precision *selection* specifically was called out by the user as a
  "just recommend it directly" category, distinct from conceptual
  derivation — Socratic pushback isn't wanted there.
- Watch calibration: earlier in this project, a sub-derivation (attention
  weight footprint under MLA/TP) went deeper than its actual role warranted
  (it was only in service of one input number, batch size). The user caught
  this themselves when it started to feel like the project might be "too
  advanced" — it wasn't; the fix was recalibrating depth, not the workload
  or the user's readiness. Worth staying alert to this pattern recurring in
  1c (imbalance modeling is flagged in the spec as the phase most likely to
  balloon).

---

## Phase 1b result (complete — full derivation in `moe_phase1_notes.md`)

Uniform/ideal-case dispatch+combine comms volume: **105 MiB/layer**, whole
system, one decode step (8,192 tokens, 13,440 bytes/token — 2,560-byte FP4
payload × 2 for dispatch+combine × 2.625 expected remote devices/token, the
last number coming from the device-limited-routing fan-out worked out
against the real paper mechanism, not the spec's naive per-expert formula).
Expert FLOPs (all 8/token) for the same scope: ≈3.09 TFLOPs/layer. Ratio
≈28,087 FLOPs/byte vs. a TPU 8i ICI ridge point of ≈4,208 FLOPs/byte —
**decisively compute-bound, ~6.7× above ridge point, in the ideal case.**
Caveat carried forward: this is a system-wide average under perfect
balance — doesn't guarantee no local comms-bound moments once Phase 1c's
imbalance model is applied.

A real conceptual correction happened mid-derivation worth knowing about if
picking this back up: batch (1,024/device, 8,192 system-wide) and seq_len
(8192) got conflated at one point — resolved by recognizing this whole
analysis is a **decode-step** derivation (one new token per active sequence
per forward pass), so seq_len does not multiply into per-step token count.
A prefill-regime version (tokens = sequences × seq_len) would be a distinct,
not-yet-done extension if ever wanted later.

## Two outstanding items — do the first before deep-diving the second

### 1. Phase 0: ASTRA-sim isn't built yet

Plan (see `moe_phase1_notes.md` § Phase 0 status for full reasoning): try a
local Docker build on the Mac first (repo's Dockerfile is unpinned
`ubuntu:22.04`, should build natively on arm64, unlike Chipyard). Checkpoint
is running the stock all-reduce example. Farmshare is the fallback only if
the NS-3 backend specifically fails to build, or for long Phase 3 sweeps —
not the default venue. **This hasn't been attempted yet — confirm with the
user whether it's been done outside this chat before assuming it's still
pending.**

### 2. Phase 1c: derive comms volume under load imbalance (the real case)

This is the next real 🧠 work now that 1b's uniform control case is done.
Open items flagged but not yet resolved: the three balance-loss coefficients
(α1/α2/α3 — expert/device/communication-level, not yet researched which is
the right lever) and that token-dropping is training-only per the paper
(likely rules it out as an imbalance-mitigation assumption for this
inference-focused analysis). Full detail in `moe_phase1_notes.md`'s "Open
items" section — don't re-derive, just pick up from there. **Watch
calibration here** (see below) — this phase is explicitly flagged in the
spec as the one most likely to balloon in scope.

---

## Where everything lives

- `moe_routing_project_spec.md` — original spec, all phases.
- `moe_phase1_notes.md` — full Phase 1a derivation trail (workload sourcing,
  chip/precision choice reasoning, MLA weight structure, the TP-sharding
  detour and its resolution, batch-size math) plus Phase 0 status and open
  items for 1b/1c. This is the detailed reference — read it when you need
  the "why," not just the "what."
- This file — update/overwrite at the next natural pause point (e.g., end of
  1b), same convention as the attention project's `handoff.md`.
- Sibling project: `codesign_project_spec.md` / `phase1_notes.md` /
  `handoff.md` — the attention project this one extends. Phase 4 of this
  spec asks for a synthesis connecting the two; worth knowing that context
  exists even though it's not relevant until then.
