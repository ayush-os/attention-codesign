# Handoff — Resume Here

**Purpose:** this doc exists so a new chat session can pick up this project exactly where the last one left off, without needing to re-read the full Phase 1 derivation trail first. Read this doc first; use `prefill_notes.md` as the detailed Phase 1 reference when you need it.

**Status as of this handoff:** **Phase 1 (prefill) is complete.** Next step: **Phase 2 (decode)** — repeat the full 1a→1d loop for decode-phase attention, the memory-leaning counterpart to Phase 1's compute-leaning prefill.

---

## Critical collaboration constraint — read before responding to anything

This is a **self-directed learning project**. The user is going from workload characteristics to hardware design decisions, validated in RTL, and is doing every derivation themselves. Your job is to **check reasoning, ask questions that expose gaps, flag missing considerations, and help structure/log conclusions — not to solve steps, hand over derivations, or supply numbers/answers the user hasn't produced themselves.**

- When the user shares a hypothesis or derivation: probe assumptions, ask what they checked vs. asserted, don't supply the missing piece.
- Exception: pure arithmetic plug-in *after* the user has established the formula themselves — the user has explicitly said they're fine delegating that (they don't want to be a human calculator once the reasoning is done). Still don't derive the formula itself for them.
- Exception: factual/reference lookups (real hardware specs, tool defaults, Gemmini/Chisel source behavior, "did we already log X") — verify via web search/source reading rather than trusting memory, but don't extend this into doing their conceptual work. This exception got real use in Phase 1d: resolving whether Gemmini has a native softmax unit, and whether its transposer supports the Phase 1b axis-routing assumption, were both legitimate direct lookups against Gemmini's actual GitHub source — see `prefill_notes.md` §4.5/§4.7 for how that looked in practice.
- Phase 0/tooling setup is explicitly marked 🔧 in the project spec (boilerplate, not learning-bearing) — fine to help directly and concretely there, unlike the 🧠-marked conceptual phases. This extends to Farmshare toolchain/build debugging (env vars, Makefile regeneration, TIMEOUT_CYCLES, tmux) — drive that directly, only flag back when something is a genuine conceptual/architectural finding rather than a build error.
- The user works well with concrete, quantified counterexamples/questions (e.g., "would an 8192×8192 array be physically buildable?", "how many cycles would the full workload take at this array's peak throughput, at Verilator's realistic kHz-range speed?") rather than abstract pushback — ground Socratic questions in numbers where possible.
- When something surfaces mid-investigation that's cheaper to check by reading existing files/logs than by reasoning it out or running something new, do that lookup directly rather than asking the user to re-derive it. Doesn't extend to the actual conceptual interpretation of what's found.
- **Scope is a legitimate, discussable topic, not just something to push through.** Phase 1d was deliberately descoped mid-project (see `prefill_notes.md` §4.1) after the user judged Phase 1c's signal-to-time ratio poor. When a phase's marginal learning value looks low relative to its time cost, it's fine — expected, even — to surface that directly and propose a recommendation, the way "is Phase 1d even worth doing" and "should we chase the softmax-granularity question" were both handled: give a real recommendation with reasoning, not just an open-ended question back.
- **Logging convention**: keep a live working log in a fresh `notes.md`, in Prediction/Log style (prediction stated before checking, then the log entry once resolved), the same way Phase 1's derivation was tracked. At Phase 2's natural completion point, refactor/polish that into a standalone `decode_notes.md` (mirroring what was just done for Phase 1 → `prefill_notes.md`) — a finished writeup-style document, not a chronological journal, but with no substantive content lost. Update this `handoff.md` file (overwrite, don't append) at each natural pause point.

---

## Project structure (from `spec.md`)

Phase 0 (Timeloop/Accelergy + Chipyard/Gemmini setup) → Phase 1 (prefill, compute-leaning — **complete**) → **Phase 2 (decode, memory-leaning)** → Phase 3 (numerics/precision) → optional Phase 4 (real-hardware check). Phase 1/2 each follow the same loop: 1a hand-derive FLOPs/bytes/AI/ridge-point → 1b hypothesize PE array/dataflow/scratchpad → 1c sweep Timeloop, compare → 1d configure Gemmini, read generated RTL, run via Verilator, explain every gap vs. Timeloop.

**Fallback framing, worth keeping in mind**: per the spec, Phase 1 alone (which is now done) is already "a complete, defensible artifact." Phase 2/3/4 are additive, not required — so scope decisions for Phase 2 can be made with the same directness Phase 1d's scope-down was, without treating the original spec's ambition as non-negotiable.

---

## Phase 1 (prefill): complete — summary

**Full derivation, all four sub-phases, in `prefill_notes.md`.** Do not re-derive any of this — it's a finished, validated record. Skimming it (especially §5 "Cross-Phase Synthesis" and §6 "Open Threads Carried Forward") before starting Phase 2 is worth the few minutes.

The compressed version, just enough to orient Phase 2:

- **Workload**: Llama 3-70B prefill (batch=32, seq_len=8192, n_heads=64, n_kv_heads=8 GQA, d_head=128, int8), from *How to Scale Your Model*.
- **1a**: Total FLOPs 2⁴⁶ (MHA = GQA exactly). Regime (compute- vs. memory-bound) is governed almost entirely by whether the softmax/P intermediate is fused on-chip, not by workload shape — fused is decisively compute-bound (AI 8192/14564 ≫ ridge 480.5), unfused is decisively memory-bound (AI ≈126/126.9 ≪ ridge). GQA's benefit is regime-dependent in two distinct ways (Amdahl's-Law-style, and roofline-position-style — fused prefill gets zero throughput benefit from GQA at all, since compute-bound time is set by FLOPs alone).
- **1b**: 128×128 PE array (Trainium precedent), weight-stationary with K/V as the stationary operand (driven by GQA's 8× group-reuse), `tile_k`=1024, `tile_q`=32, fine-grained online-softmax. Major open finding: `tile_q`=32 forces Q-tile outer to K/V-chunk, breaking the "fetch K/V once per group" reuse assumption (~256× re-fetch instead of once) — deliberately left unresolved for Timeloop.
- **1c**: Confirmed this Timeloop setup can only characterize the *unfused* regime (structural limitation, not a mapper finding). Ridge point had to be recomputed for the actual modeled architecture (≈80.08, not TPU v5e's 480.5) — AI≈126 is still compute-bound for *this* system. Dataflow sensitivity is itself regime-dependent (barely matters near an architecture's own ridge, matters a lot further above it). A single mapper run's "winner" is a local optimum, not a guaranteed ceiling.
- **1d (deliberately scoped down** — no custom attention kernel, no real-workload-scale run, given a back-of-envelope check showed that would be genuinely multi-day for a marginal-at-best payoff): configured Gemmini at 32×32/WS-only (scaled down from the 128×128 hypothesis), validated on Farmshare via Verilator. Confirmed Gemmini has a native softmax hardware unit whose structure mirrors the online-softmax mechanism independently hand-derived in 1b. Confirmed the axis-routing assumption from 1b via Gemmini's real transposer hardware. Found that a "WS-only" config restricts behavior via a compile-time control constant, not by generating structurally smaller hardware — revising, not overturning, the 1b hardware-cost framing.

---

## Immediate next step: Phase 2 (decode)

Per the spec, **before repeating 1a-1d**, think through from first principles why decode should push hardware conclusions in a *different* direction than prefill — larger effective memory-bandwidth need relative to compute, a different reuse pattern (small batch, single new token, dominated by reading the growing KV cache rather than large matmuls), possibly a different ridge-point crossover. **Write that prediction down before touching any numbers** — don't just copy Phase 1's config and assume it transfers; that assumption is exactly what Phase 2 is testing. This mirrors Phase 1a's own opening instruction, and the user has already pre-registered one piece of it directly: Phase 1a Key Takeaway #7 predicts GQA's throughput win — invisible in fused, compute-bound prefill — should show up for real in decode's memory-bound regime. Whether that prediction holds is a natural first thing to check.

Concrete first decisions for Phase 2a (workload characterization — still fully 🧠, not predetermined here):
- Pick a concrete decode-phase shape (batch, KV-cache length representing an already-filled context, single new token per step) — same Llama 3-70B GQA config as a natural default for continuity with Phase 1, but this is the user's call to make and justify, not something to assume.
- Derive FLOPs/bytes/AI/ridge-point for *this* shape from scratch — the reuse pattern is fundamentally different (no large Q·K^T/·V matmuls over a full sequence; instead, one query token attending over a long, already-materialized KV cache), so the Phase 1a formulas don't just carry over with different numbers plugged in.

**Deliverable for Phase 2** (per spec): same structure as Phase 1 — prediction, Timeloop result, Gemmini/RTL result, gap explanation — plus an explicit comparison to Phase 1: how did the "ideal" array shape, dataflow, and scratchpad sizing differ between prefill and decode, and does that match what you'd expect from the two workloads' arithmetic intensity?

---

## Where everything lives

- `spec.md` — original project spec, all phases.
- `prefill_notes.md` — the complete, polished Phase 1 record (all four sub-phases, cross-phase synthesis, open threads, toolchain appendix). Reference only — nothing here should be re-derived.
- `notes.md` — **create this fresh** for Phase 2's live derivation trail, same Prediction/Log style Phase 1 used. Refactor into `decode_notes.md` at Phase 2's natural completion point.
- `timeloop-accelergy-exercises/workspace/attention_1c/` (separate repo/Docker environment, not this one) — Phase 1c's actual Timeloop configs/outputs. A parallel `attention_2c/`-style directory is the natural place for Phase 2's Timeloop work, for consistency.
- Farmshare (`~/chipyard`) — same Chipyard/Gemmini checkout Phase 1d used; the `AttentionPrefillRocketConfig` config and its generated RTL are still there and are a real, working reference point for however Phase 2's Gemmini config gets built.
- This file (`handoff.md`) — update/overwrite at the next natural pause point rather than leaving stale.
