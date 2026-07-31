# MoE Routing Project — Handoff (Resume Here)

**Purpose:** read this first when picking the MoE routing project back up in a
new session — it's the short "where things stand" pointer. Full derivation
trail, sourced numbers, and the reasoning behind every choice lives in
`moe_phase1_notes.md`; don't re-derive anything already logged there.

**Status as of this handoff:** Phase 1 fully complete — 1a (workload shape),
1b (uniform-routing comms volume), and 1c (load-imbalance modeling) all
done. Phase 0 (ASTRA-sim setup) decided but **still not yet executed**.
Phase 2 (system architecture hypothesis) is the next real work.

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

## Phase 1c result (complete — full derivation in `moe_phase1_notes.md`)

**Headline: this workload is robust to realistic load imbalance on TPU 8i
at FP4 — every scenario stays decisively compute-bound.** Imbalance
magnitude grounded in two real, cited sources (Gini≈0.70 across
DeepSeek-V3/Qwen3-MoE/Mixtral; 70% GPU stall time on real Mixtral-8×7B
serving → implies a 3.3× device-load multiplier). Key structural finding:
splitting per-device FLOPs into a fixed shared-expert component (driven by
home tokens, doesn't scale with routing popularity) and a scaling
routed-expert component (does scale) reveals a **hard, imbalance-proof
floor of ≈21,065 FLOPs/byte** — no imbalance severity, however extreme, can
push this below the ≈4,208 ridge point. The three-point range (uniform /
CF=1.0 dropping-on / uncapped dropping-off) all land compute-bound, ~5-6.7×
margin throughout.

**The one lever that actually matters: dispatch precision, not imbalance.**
The floor scales inversely with payload precision — FP4 gives ~5× margin,
BF16 only ~1.25× (thin), and the exact crossover to comms-bound is
**≈2.5 bytes/element**. If this workload ever tips comms-bound, it'll be
from a numerics decision, not routing skew.

Also worth knowing if picking this back up: an earlier logged assumption
("token-dropping is training-only") was **wrong** and got corrected
mid-session — the paper actually allows optional inference-time dropping at
CF=1.0 exactly, device-level, lowest-affinity-first. Don't re-assert the old
framing.

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

### 2. Phase 2: predict the ideal system architecture

This is the next real 🧠 work now that all of Phase 1 (uniform + imbalanced
comms volume) is done. Per the spec: topology, bandwidth allocation, and
buffering/SRAM sizing, informed by Phase 1's findings. One instinct flagged
in `moe_phase1_notes.md` worth testing rather than assuming: since imbalance
turned out to barely move the needle (headroom only eroded from 6.7× to
~5×), exotic imbalance-driven topology/bandwidth choices (e.g. extra
capacity earmarked for popular-expert links) may matter less than they
would have going into this phase without that result in hand.

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
